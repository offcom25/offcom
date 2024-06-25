import json
import logging
import os.path
import random
import time
import traceback
import copy
import warnings
from abc import abstractmethod
from typing import List, Optional, Set, Tuple, Type

import z3

from offcom.fusion.tvm.op_pattern_kind import *
from offcom.abstract.arith import *
from offcom.abstract.dtype import *
from offcom.abstract.op import (
    AbsOpBase,
    AbsTensor,
    Expand,
    Placeholder,
    concretize_op,
    rank_all,
)
from offcom.error import ConstraintError, SanityCheck
from offcom.gir import GraphIR, InstExpr, InstIR
from offcom.logging import MGEN_LOG, SMT_LOG
from offcom.util import HAS_PYGRAPHVIZ, set_seed, viz_dot


class RequiredDimNotFound(Exception):
    pass


def concretize_graph(ir: GraphIR, model: z3.ModelRef) -> GraphIR:
    return ir.concretize(model)


class BaseGen:
    def __init__(
            self,
            opset: List[AbsOpBase],
            seed: Optional[int] = 42,
            forward_prob: Optional[float] = None,
            concr_ph_dim_rng: Tuple[int, int] = (1, 64),
            max_elem_per_tensor: int = 2 ** 16,
            rank_choices=None,
            dtype_choices=None,
    ):
        assert len(opset) > 0, "opset must not be empty"
        if seed is not None:
            set_seed(seed)

        self.seed = seed
        self.op_candidates = opset
        self.ir = GraphIR()
        self.monotonic_placeholder_id = 0

        # Names of current placeholders
        self.placeholders: List[str] = []
        # for all (including newly created tmp) placeholders
        self.forward_prob = 0.5 if forward_prob is None else forward_prob
        self.concr_ph_dim_rng = concr_ph_dim_rng
        self.max_elem_per_tensor = max_elem_per_tensor
        self.rank_choices = rank_choices if rank_choices else rank_all()
        # analyze the dtypes used by the opset
        dtype_top = set()
        for op in opset:
            dtype_top.update({dt for dtc in op.in_dtypes + op.out_dtypes for dt in dtc})

        self.dtype_choices = (
            [
                dt if isinstance(dt, DType) else DType.from_str(dt)
                for dt in dtype_choices
            ]
            if dtype_choices
            else DTYPE_GEN_ALL
        )

        self.dtype_choices = list(dtype_top.intersection(self.dtype_choices))
        assert len(self.dtype_choices) > 0, "dtype_choices must not be empty"
        assert len(self.rank_choices) > 0, "rank_choices must not be empty"

    def random_rank(self):
        return random.choice(self.rank_choices)

    def tensor_type_constraints(
            self, atensor: AbsTensor
    ) -> List[Union[z3.BoolRef, bool]]:
        return [atensor.nelement() <= self.max_elem_per_tensor]

    @abstractmethod
    def assume(self, c: Union[z3.BoolRef, bool]):
        pass

    def make_symbolic_placeholder(self, rank, dtype=None) -> Placeholder:
        syms = self.new_syms(
            [f"ph{self.monotonic_placeholder_id}_{k}" for k in range(rank)]
        )
        ph = Placeholder(
            AbsTensor(
                shape=syms,
                dtype=dtype if dtype is not None else self.random_dtype_gen(),
            )
        )
        self.monotonic_placeholder_id += 1
        return ph

    def make_random_concrete_placeholder(self, rank, dtype=None):
        l, r = self.concr_ph_dim_rng
        shape = []
        product = 1
        for _ in range(rank):
            v = random.randint(l, r)
            if product * v > self.max_elem_per_tensor:
                v = 1
            shape.append(v)
            product *= v

        random.shuffle(shape)  # shuffle

        ph = Placeholder(
            AbsTensor(
                shape=shape,
                dtype=dtype if dtype is not None else self.random_dtype_gen(),
            )
        )
        return ph

    def random_dtype_gen(self):
        # more floats than ints.
        # ~ in DTYPE_GEN_ALL and in self.dtype_choices
        dtypes = [dt for dt in DTYPE_GEN_ALL if dt in self.dtype_choices]
        assert (
                len(dtypes) > 0
        ), "Empty INTERSECT(DTYPE_GEN_ALL, dtype_choices). Please relax dtype_choices."

        wts = [1] * len(dtypes)
        for dt in DTYPE_GEN_FLOATS:  # 这里是为了尝试生成更多的floats
            if dt in dtypes:
                wts[dtypes.index(dt)] = 4
        return random.choices(dtypes, weights=wts)[0]

    def new_sym(self, name):
        return z3.Int(name)

    def new_syms(self, names):
        return [self.new_sym(name) for name in names]

    def insert_init_ph_node(self, ph: Placeholder) -> InstIR:
        inst = self.forward_insert_node(ph, [])

        for c in ph.ttype.sym_gt_conc_ge_zero():
            self.assume(c)

        return inst

    @abstractmethod
    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def make_concrete(self) -> GraphIR:
        raise NotImplementedError

    def extra_exit_check(self, max_node_size) -> bool:
        """
        Returns:
            bool: add more checks to determine whether to exit the generation.
        """
        return False

    def num_op(self) -> int:
        # exclude placeholders.
        return self.ir.n_compute_inst()

    def try_insert(self):
        node_t = self.pick_next_op_type()
        print(node_t)
        print('debug')
        self.try_insert_node_type(node_t)

    def abstract_gen(self, max_node_size=10, max_gen_millisec=2000):
        z3.set_param("timeout", max_gen_millisec // 3)

        assert max_node_size > 0, "max_node_size must be positive"

        init_time = time.time()

        # starts generation.
        while (
                time.time() - init_time < max_gen_millisec / 1000
                and self.num_op() < max_node_size
        ):
            if self.extra_exit_check(max_node_size):
                break
            self.try_insert()

        # init graph placeholders
        SanityCheck.gt(len(self.placeholders), 0)

        def determine_ph_type(ph: str, to_input: bool):
            SanityCheck.true(ph in self.placeholders)
            ph_inst_id, _ = InstIR.var_inst_idx(ph)
            ph_inst = self.ir.find_inst_by_id(ph_inst_id)
            if to_input:
                ph_inst.iexpr.op = ph_inst.iexpr.op.input()
            else:
                ph_inst.iexpr.op = ph_inst.iexpr.op.const()

        determine_ph_type(self.placeholders[0], True)  # At lease make one input.
        for ph in self.placeholders[1:]:
            determine_ph_type(ph, random.randint(0, 1))

    def pick_next_op_type(self):
        return random.choice(self.op_candidates)

    def forward_insert_node(self, node: AbsOpBase, input_vars: List[str]) -> InstIR:
        new_inst = self.ir.add_inst(InstExpr(op=node, args=input_vars))

        if isinstance(node, Placeholder):
            # Add placeholder's return varname.
            self.placeholders.append(new_inst.retval())

        return new_inst

    def backward_insert_node(
            self, node, input_vars: List[str], ph_to_replace: List[str]
    ) -> InstIR:
        new_inst = self.forward_insert_node(node, input_vars=input_vars)

        # replace all uses of ph_to_replace
        # and delete the unused placeholders.
        for ph, rv in zip(ph_to_replace, new_inst.retvals()):
            self.ir.replace_alluse(ph, rv)
            ph_inst_id, _ = InstIR.var_inst_idx(ph)
            ph_inst = self.ir.find_inst_by_id(ph_inst_id)
            self.ir.remove_unused(ph_inst)
            self.placeholders.remove(ph)

        return new_inst

    def try_forward_insert(self, op: AbsOpBase) -> bool:
        n_inp = len(op.inp_ranks)
        dim_spec_list = []

        if op.same_inp_dims:  # find `n_inp` under the same input shapes.
            rank_set = set(op.inp_ranks[0])

            for ranks in op.inp_ranks[1:]:
                rank_set.intersection_update(set(ranks))

            SanityCheck.ge(len(rank_set), 1)

            final_dim = random.choice(list(rank_set))
            dim_spec_list = [(final_dim,)] * n_inp
        else:  # inputs have different dimension sizes.
            dim_spec_list = op.inp_ranks

        varnames = self.pick_var_group(
            dim_spec_list, op.in_dtypes, ndim_relation=op.irank_relation
        )

        if self.try_forward_insert_at(op, varnames):
            return True

        return False

    def try_backward_insert(self, op: AbsOpBase):
        # we know that: Y = op(X)
        # S1 - select Y: Y must be a placeholder; (this also means the graph must start w/ a placeholder)
        phvars = self.pick_var_group(
            op.out_ranks,
            op.out_dtypes,
            var_candidates=[
                name
                for name in self.placeholders
                if not isinstance(op, Expand)
                   or self.ir.vars[name].ndims < op.expand_last_dim
            ],
            ndim_relation=op.orank_relation,
        )

        if self.try_occupy_placeholder(op, phvars):
            return True

        return False

    def try_insert_node_type(
            self, node_t: Type[AbsOpBase], max_tensor_pick_time=3
    ) -> bool:
        MGEN_LOG.debug(
            f"@[Node #{self.ir.n_inst()}] <-- trying to insert node type {node_t.__name__}"
        )

        try:
            for _ in range(max_tensor_pick_time):
                # should recreate a new instance since some attributes (like axis) should be initialized for each pick
                op_param_n = node_t.get_num_var_param()
                op_id = self.ir.n_inst()
                op_params = [
                    self.new_sym("op%s_%s" % (op_id, k)) for k in range(op_param_n)
                ]

                op: AbsOpBase = node_t(*op_params)

                if random.uniform(0, 1) < self.forward_prob:
                    if self.try_forward_insert(op):
                        return True
                else:
                    if self.try_backward_insert(op):
                        return True
        except ConstraintError:
            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug(traceback.format_exc())
            return False

        return False

    def filter_rank_dtype(self, ndims, dtype, candidates: List[str]) -> List[str]:
        cans = candidates

        cans = list(
            filter(  # filter with ndim
                lambda vname: self.ir.vars[vname].ndims in ndims, cans
            )
        )
        if len(cans) == 0:
            raise ConstraintError(f"Cannot find candidate to sat rank of {ndims}.")

        if dtype is not None:
            cans = list(
                filter(  # filter with dtype
                    lambda vname: self.ir.vars[vname].dtype == dtype, cans
                )
            )
            if len(cans) == 0:
                raise ConstraintError(
                    f"Cannot find candidate to sat rank of {ndims} and dtype {dtype}."
                )

        return cans

    def pick_var_group(
            self,
            ndim_list: List[Set[int]],
            dtype_combs: List[Tuple[DType, ...]],
            var_candidates: Optional[List[str]] = None,
            ndim_relation=None,
    ) -> List[str]:
        """Randomly pick a group of variables that satisfy one of the `dtype_combs` and `ndim_list`.
        Returns:
            List[str]: Satisfiable group of variable names.
        """

        if var_candidates is None:
            var_candidates = list(self.ir.vars.keys())

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            for cand in var_candidates:
                MGEN_LOG.debug(
                    f"Candidate: {cand}: {self.ir.vars[cand].dtype} ~ {self.ir.vars[cand].ndims}"
                )
            MGEN_LOG.debug(f"Input data ranks candidates: {ndim_list}")
            MGEN_LOG.debug(f"Input data types candidates: {dtype_combs}")

        ir_dtypes = []
        for i, ndims in enumerate(ndim_list):
            ir_dtypes.extend(
                [
                    self.ir.vars[vname].dtype
                    for vname in self.filter_rank_dtype(
                    ndims=ndims, dtype=None, candidates=var_candidates
                )
                ]
            )

        # possibility check: must be generatable dtypes.
        if set(ir_dtypes).isdisjoint(set(DTYPE_GEN_ALL)):
            raise ConstraintError(f"Unsupported dtypes: {ir_dtypes}")

        # only use dtypes currently available after ndim filtering
        dcombs = [c for c in dtype_combs if all(dt in ir_dtypes for dt in c)]
        if len(dcombs) == 0:
            raise ConstraintError(
                f"No candidate w/ rank@{ndim_list} & dtype@{dtype_combs}"
            )

        # randomized enumeration over dtype_combs
        random.shuffle(dcombs)
        for dcomb in dcombs:
            if ndim_relation is None:
                ret = []
                for i, ndims in enumerate(ndim_list):
                    candidates = self.filter_rank_dtype(
                        ndims=ndims, dtype=dcomb[i], candidates=var_candidates
                    )
                    ret.append(random.choice(candidates))
                return ret
            else:
                # candidates for 0-indexed tensor
                topcands = self.filter_rank_dtype(
                    ndims=ndim_list[0], dtype=dcomb[0], candidates=var_candidates
                )
                random.shuffle(topcands)
                for tcand in topcands:
                    ret = [tcand]
                    for i, ndims in enumerate(ndim_list[1:]):
                        required_ndim = ndim_relation[i + 1](self.ir.vars[tcand].ndims)
                        if required_ndim not in ndim_list[i + 1]:
                            break
                        self.filter_rank_dtype(
                            ndims=[required_ndim],
                            dtype=dcomb[i + 1],
                            candidates=var_candidates,
                        )
                    if len(ret) == len(ndim_list):
                        return ret

        raise ConstraintError("Cannot find desired combinations of tensor variables.")


def check_sat(solver: z3.Solver, *assumptions) -> z3.CheckSatResult:
    start = time.time()

    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        if solver.assertions():
            SMT_LOG.debug(
                f"existing constraints: {', '.join(map(str, solver.assertions()))}"
            )
        if assumptions:
            SMT_LOG.debug(f"new constraints: {', '.join(map(str, assumptions))}")

    cres = solver.check(*assumptions)

    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        SMT_LOG.debug(
            f"{cres} <-- checking time: {int((time.time() - start) * 1000)} ms"
        )

        if cres == z3.unsat:
            SMT_LOG.debug(f"Unsat core: {solver.unsat_core()}")

    return cres


def set_z3_state(seed=None):
    z3.set_param(
        "smt.phase_selection",
        5,
        "smt.arith.random_initial_value",
        True,
        "smt.random_seed",
        seed,
        "sat.random_seed",
        seed,
        "sat.phase",
        "random",
        "memory_max_size",
        50 * 1024,  # MB
    )


class SymbolicGen(BaseGen):
    def __init__(
            self,
            opset,
            seed=None,
            init_fp=False,
            symbolic_init=True,
            **kwargs,
    ):
        super().__init__(opset, seed, **kwargs)
        if seed is not None:
            set_z3_state(seed)

        self.solver = z3.Solver()
        self.last_solution: Optional[z3.ModelRef] = None

        # Insert the first node.
        if symbolic_init:
            ph = self.make_symbolic_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        else:
            ph = self.make_random_concrete_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )

        self.insert_init_ph_node(ph)
        for pred in self.tensor_type_constraints(ph.ttype):
            self.assume(pred)

    def assume(self, c: z3.BoolRef):
        self.solver.add(c)

    def check_sat(self, *assumptions):
        cres = check_sat(self.solver, *assumptions)
        if cres == z3.sat:
            self.last_solution = self.solver.model()
        return cres

    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        itensors = [self.ir.vars[vname] for vname in input_vars]
        constraints = node.checked_requires(itensors)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        # make a copy
        otensors = node.checked_type_transfer(itensors)

        for aten in otensors:
            for c in aten.gt_zero():
                constraints.append(c)

        # limit output tensor size
        for aten in otensors:
            constraints.extend(self.tensor_type_constraints(aten))

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.assume(c)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {itensors}")
            MGEN_LOG.debug(f"\toutputs: {otensors}")

        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        self.forward_insert_node(node, input_vars)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {phvars} for node {node}"
            )
        # S2 - create X: X can be
        #                   - a new placeholder (fallback)
        #                   - an existing alive shape

        otensors = [self.ir.vars[name] for name in phvars]

        # S2.2: try to reuse some existing outputs;
        # TODO: allow reuse existing alive shapes
        # n_inps = len(node.inp_ranks)
        # max_try = 2
        # n_reuse = n_inps - 1
        # while n_reuse > 0 and max_try > 0:
        #     # TODO...
        #     max_try -= 1
        #     n_reuse -= 1

        # S2.2: reusing outputs failed. as a fallback, promote all free vars to placeholders.
        phs_as_op_inputs: List[Placeholder] = []
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(otensors):
            # oversample rank 4 tensors as they may be more important
            ph = self.make_symbolic_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(ph.ttype.gt_zero())
            constraints.extend(self.tensor_type_constraints(ph.ttype))

        itensors = [p.ttype for p in phs_as_op_inputs]
        constraints.extend(node.checked_requires(itensors))
        inferred_otensors = node.checked_type_transfer(itensors)

        for i, shape in enumerate(inferred_otensors):
            constraints.extend(shape.eq(otensors[i]))
            constraints.extend(shape.gt_zero())

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {phs_as_op_inputs}")

        for c in constraints:
            self.assume(c)

        # succ.
        input_vars = []

        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        node.bind_input_like(itensors)
        node.bind_output_like(inferred_otensors)

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def make_concrete(self) -> GraphIR:
        SanityCheck.gt(len(self.ir.insts), 0, "Empty graph!")
        SanityCheck.not_none(self.last_solution, "Run check_sat first!")
        self.ir.concretize(self.last_solution)
        return self.ir


class SymboliSingleIOGen(SymbolicGen):
    """Generate a model which has only one input and one output tensor"""

    def __init__(self, *args, **kwargs):
        forward_prob = kwargs.pop("forward_prob", None)
        if forward_prob is not None:
            # only warn once
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                warnings.warn(
                    "`forward_prob` is not supported in SymboliSingleIOGen which is always 1."
                    "Why: the implementation first just generates forward graph and then cuts backward."
                )
        kwargs["forward_prob"] = 1.0
        super().__init__(*args, **kwargs)

    def eliminate_extra_outputs(self):
        """Find the minimal cut to make the graph has only one output tensor."""
        prev_size = None
        while prev_size != self.ir.n_inst():
            prev_size = self.ir.n_inst()
            cuts = self.ir.leaf_cut_chains()
            cuts = sorted(cuts, key=lambda x: len(x))
            for cut in cuts[:-1]:
                for inst in cut:
                    self.ir.remove_unused(inst)
            self.ir.assert_wellform()
        SanityCheck.eq(len(self.ir.leaf_var()), 1, "Failed to eliminate extra outputs!")

    def abstract_gen(self, **kwargs):
        SymbolicGen.abstract_gen(self, **kwargs)
        self.eliminate_extra_outputs()


class ConcolicGen(BaseGen):
    """Different from SymbolicGen, the graph after an insertion is `concrete` in ConcolicGen.
    However, each step when inserting a node, we symbolically find a satisfiable solution for it.
    """

    def __init__(
            self,
            opset,
            seed=None,
            init_fp=False,
            **kwargs,
    ):
        super().__init__(opset, seed, **kwargs)
        if seed is not None:
            set_z3_state(seed)

        # Insert the first node.
        self.insert_init_ph_node(
            self.make_random_concrete_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        )

    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        solver = z3.Solver()

        itensors = [self.ir.vars[vname] for vname in input_vars]
        constraints = node.checked_requires(itensors)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        # make a copy
        otensors = node.checked_type_transfer(itensors)

        for aten in otensors:
            for c in aten.sym_gt_conc_ge_zero():
                constraints.append(c)

        check_res = check_sat(solver, *constraints)

        if check_res != z3.sat:
            return False

        # materialize otensors and attributes.
        node = concretize_op(node, solver.model())
        otensors = node.checked_type_transfer(itensors)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {itensors}")
            MGEN_LOG.debug(f"\toutputs: {otensors}")

        # Shape checker.
        # NOTE: No need to check input shape as they are already checked for being in the graph.
        for i, ten in enumerate(otensors):
            if not all(self.tensor_type_constraints(ten)):
                MGEN_LOG.debug(f"{i}-th output type constraint failed: {ten}")
                return False

        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        self.forward_insert_node(node, input_vars)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {phvars} for node {node}"
            )

        # TODO: In backward insertion, reusing existing tensors is not implemented.

        # Concrete tensors.
        solver = z3.Solver()

        otensors = [self.ir.vars[name] for name in phvars]
        phs_as_op_inputs: List[Placeholder] = []
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(otensors):
            # oversample rank 4 tensors as they may be more important
            ph = self.make_symbolic_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(ph.ttype.sym_gt_conc_ge_zero())

        itensors = [p.ttype for p in phs_as_op_inputs]
        constraints.extend(node.checked_requires(itensors))
        inferred_otensors = node.checked_type_transfer(itensors)

        for i, shape in enumerate(inferred_otensors):
            constraints.extend(shape.eq(otensors[i]))
            constraints.extend(shape.sym_gt_conc_ge_zero())

        check_res = check_sat(solver, *constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {phs_as_op_inputs}")

        model = solver.model()
        # succ.
        itensors = []
        for i, ph in enumerate(phs_as_op_inputs):
            # materialize ph
            phs_as_op_inputs[i] = concretize_op(ph, model)
            itensors.append(phs_as_op_inputs[i].ttype)

        # Input Shape checker.
        # NOTE: No need to check output because they are already in the graph thus valid.
        for i, ten in enumerate(itensors):
            if not all(self.tensor_type_constraints(ten)):
                MGEN_LOG.debug(f"{i}-th input type constraint failed: {ten}")
                return False

        node = concretize_op(node, model)
        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        # Apply insertion.
        input_vars = []
        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def assume(self, c: bool):
        # semantically equivalent to `assert c`.
        ConstraintCheck.true(c, "Assumption failed")

    def make_concrete(self) -> GraphIR:
        return self.ir


class TVMFusedGen(BaseGen):
    def __init__(
            self,
            opset: List[Type[AbsOpBase]],
            nodes_allocation: List = None,
            seed=42,
            init_fp=True,
            allow_nnsmith_insert=False,
            **kwargs,
    ):
        super().__init__(opset, seed, **kwargs)
        self.nodes_allocation = nodes_allocation

        self.fuse_op_now_placeholders: List[str] = []
        self.last_backward_insert_new_placeholders: List[str] = []
        self.last_backward_insert_new_placeholders_cpy: List[str] = []
        self.generate_start_placeholder_list: List[str] = []

        self.allow_nnsmith_insert = allow_nnsmith_insert

        self.last_solution: Optional[z3.ModelRef] = None  # last z3 solution
        self.init_ph_alive = True  # if the init placeholder is alive

        self.fuse_output_var_list = []
        self.fuse_input_var_list = []
        self.non_fuse_normal_var_list = []
        self.randomized_sat_vars = []

        generate_key_name_val_op_dict(opset)
        self.opset = opset
        self.solver = z3.Solver()

        self.insert_init_ph_node(
            self.make_symbolic_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        )

    def insert_init_ph_node(self, ph: Placeholder) -> InstIR:
        # insert the init ph here
        inst = self.forward_insert_node(ph, [])


        self.non_fuse_normal_var_list.extend(inst.retvals())  # note that the init ph is not a fused op

        for c in ph.ttype.sym_gt_conc_ge_zero():
            self.solver.add(c)

        return inst

    def create_placeholder(self, rank, dtype=None) -> Placeholder:


        syms = self.new_syms(
            [f"v{self.monotonic_placeholder_id}_{k_rank}" for k_rank in range(rank)]
        )

        ph = Placeholder(
            AbsTensor(
                shape=syms, dtype=dtype if dtype is not None else self.random_dtype_gen()
            )
        )

        self.monotonic_placeholder_id += 1
        return ph

    def abstract_gen(self, max_gen_millisec=5000, max_node_size=None):
        z3.set_param("timeout", max_gen_millisec // 3)

        max_node_size = sum(self.nodes_allocation) if self.allow_nnsmith_insert else sum(self.nodes_allocation[:-1])
        # print(f"the max node size is {max_node_size}. all_nnsmith_insert={self.allow_nnsmith_insert}")
        init_time = time.time()
        success_flag = False

        var_candidates = self.fuse_input_var_list + self.fuse_output_var_list + self.non_fuse_normal_var_list
        placeholder_candidates = [i for i in var_candidates if i in self.placeholders]
        # print(f"placeholders: {self.placeholders}")
        # print(f"var: {var_candidates}")

        while not success_flag:
            random.seed(time.time())
            tvm_phase = random.randint(1, 4)

            # print(f"the phase is {tvm_phase}")
            tvm_phase = 1

            if tvm_phase == 1:
                success_flag, _ = self.try_fuse_phase_1(end_placeholder=random.choice(placeholder_candidates),
                                                        nodes_allocation=self.nodes_allocation,
                                                        each_retry_time=10)
            elif tvm_phase == 2:
                success_flag, _ = self.try_fuse_phase_2(end_placeholder=random.choice(placeholder_candidates),
                                                        nodes_allocation=self.nodes_allocation,
                                                        each_retry_time=10)
            elif tvm_phase == 3:
                success_flag, _ = self.try_fuse_every_phase(end_placeholder=random.choice(placeholder_candidates),
                                                            nodes_allocation=self.nodes_allocation,
                                                            each_retry_time=10)
            else:
                success_flag, _ = self.try_fuse_every_phase(end_placeholder=random.choice(placeholder_candidates),
                                                            nodes_allocation=self.nodes_allocation,
                                                            each_retry_time=10)
            # print("fused graph generation fin!")

            with open("/root/.cache/offcom/tvm_phase.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            if success_flag:
                data[str(tvm_phase)]["succ"] += 1
            else:
                data[str(tvm_phase)]["fail"] += 1

            with open("/root/.cache/offcom/tvm_phase.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        if self.allow_nnsmith_insert:
            print("start random insert")
            while (
                    time.time() - init_time < max_gen_millisec / 1000
                    and self.num_op() < max_node_size
            ):
                if self.extra_exit_check(max_node_size):
                    break
                self.try_insert()

        SanityCheck.gt(len(self.placeholders), 0)

        def determine_ph_type(ph: str, to_input: Union[int, bool]):
            SanityCheck.true(ph in self.placeholders)
            ph_inst_id, _ = InstIR.var_inst_idx(ph)
            ph_inst = self.ir.find_inst_by_id(ph_inst_id)
            if to_input:
                ph_inst.iexpr.op = ph_inst.iexpr.op.input()
            else:
                ph_inst.iexpr.op = ph_inst.iexpr.op.const()

        determine_ph_type(self.placeholders[0], True)
        for ph in self.placeholders[1:]:
            determine_ph_type(ph, random.randint(0, 1))



    def try_fuse_phase_1(self, end_placeholder: str, nodes_allocation, each_retry_time=40) -> (bool, List[str]):
        # print("TVM fuse phase 1 start")

        main_branch_node_size, extra_branch_node_size, _ = nodes_allocation


        end_inst_id, _ = InstIR.var_inst_idx(end_placeholder)
        end_inst = self.ir.find_inst_by_id(end_inst_id)

        self.fuse_op_now_placeholders.append(end_inst.retval())
        self.last_backward_insert_new_placeholders_cpy.append(end_inst.retval())


        end_op_list = get_pattern_kind_operator_list(TVMOpPatternKind.kElemWise)


        if not self.try_insert_fusable_op('end', end_op_list, end_placeholder, each_retry_time):
            return False, None


        mid_node_cnt = 0

        mid_op_list = get_pattern_kind_operator_list(
            TVMOpPatternKind.kBroadcast) + end_op_list

        for now_node_idx in range(1, main_branch_node_size - 1):
            self.last_backward_insert_new_placeholders_cpy = copy.deepcopy(self.last_backward_insert_new_placeholders)
            self.last_backward_insert_new_placeholders = []

            if self.try_insert_fusable_op('mid', mid_op_list, end_placeholder, each_retry_time):
                mid_node_cnt += 1

            if mid_node_cnt != now_node_idx:
                return False, None

        if mid_node_cnt != main_branch_node_size - 2:
            return False, None

        k_out_e_wise_fusable_op_list = get_pattern_kind_operator_list(TVMOpPatternKind.kOutEWiseFusable)
        self.last_backward_insert_new_placeholders_cpy = copy.deepcopy(self.last_backward_insert_new_placeholders)
        self.last_backward_insert_new_placeholders = []
        if not self.try_insert_fusable_op('init', k_out_e_wise_fusable_op_list, end_placeholder, each_retry_time):
            return False, None


        self.generate_start_placeholder_list = copy.deepcopy(self.last_backward_insert_new_placeholders)
        self.fuse_input_var_list.extend(self.generate_start_placeholder_list)


        extra_op_list = mid_op_list
        extra_cnt = 0
        for now_extra_idx in range(0, extra_branch_node_size):


            self.last_backward_insert_new_placeholders_cpy = copy.deepcopy(
                self.fuse_op_now_placeholders)
            self.last_backward_insert_new_placeholders_cpy = [i for i in self.last_backward_insert_new_placeholders_cpy
                                                              if i not in self.generate_start_placeholder_list]


            if not self.last_backward_insert_new_placeholders_cpy:
                return True, self.generate_start_placeholder_list
            if self.try_insert_fusable_op('extra', extra_op_list, end_placeholder,
                                          each_retry_time):
                extra_cnt += 1
            if extra_cnt != now_extra_idx + 1:
                return False, None

        if extra_cnt != extra_branch_node_size:
            return False, None

        return True, self.generate_start_placeholder_list

    def try_fuse_phase_2(self, end_placeholder: str, nodes_allocation, each_retry_time=40,
                         toppest_must_ktuple=False) -> (bool, List[str]):
        # print("TVM fuse phase 2 start")
        main_branch_node_size, extra_branch_node_size, _ = nodes_allocation


        end_inst_id, _ = InstIR.var_inst_idx(end_placeholder)
        end_inst = self.ir.find_inst_by_id(end_inst_id)
        # try:
        self.fuse_op_now_placeholders.append(end_inst.retval())
        # finally:
        # print(f"end_inst = {end_inst}, end_inst_id = {end_inst_id}")
        self.last_backward_insert_new_placeholders_cpy.append(end_inst.retval())

        k_element_wise_op_list = get_pattern_kind_operator_list(TVMOpPatternKind.kElemWise)
        k_broadcast_op_list = get_pattern_kind_operator_list(TVMOpPatternKind.kBroadcast)
        k_injective_op_list = get_pattern_kind_operator_list(TVMOpPatternKind.kInjective)

        end_op_list = k_element_wise_op_list + k_broadcast_op_list + k_injective_op_list


        if not self.try_insert_fusable_op("end", end_op_list, end_placeholder, each_retry_time):
            return False, None



        mid_node_cnt = 0
        if toppest_must_ktuple:
            mid_branch_cnt = main_branch_node_size - 2
        else:
            mid_branch_cnt = main_branch_node_size - 1

        mid_op_list = [op for op in end_op_list if op != "core.GetTupleElem"]
        for now_branch_idx in range(1, mid_branch_cnt):
            self.backup()
            if self.try_insert_fusable_op("mid", mid_op_list, end_placeholder,
                                          each_retry_time):
                mid_node_cnt += 1


            if mid_node_cnt != now_branch_idx:
                return False, None

        if toppest_must_ktuple:
            now_branch_idx = main_branch_node_size - 2
            self.backup()

            if self.try_insert_fusable_op("mid", mid_op_list, end_placeholder, each_retry_time):
                mid_node_cnt += 1

            if mid_node_cnt != now_branch_idx:

                return False, None

        if mid_node_cnt != main_branch_node_size - 2:
            return False, None


        k_tuple_op_list = get_pattern_kind_operator_list(TVMOpPatternKind.kTuple)
        if not toppest_must_ktuple:
            top_op_list = k_injective_op_list + k_tuple_op_list
        else:
            top_op_list = k_tuple_op_list

        self.backup()

        if not self.try_insert_fusable_op("top", top_op_list, end_placeholder, each_retry_time):
            return False, None

        self.generate_start_placeholder_list = copy.deepcopy(self.last_backward_insert_new_placeholders)
        self.fuse_input_var_list.extend(self.generate_start_placeholder_list)

        extra_cnt = 0
        extra_op_list = end_op_list
        for now_extra_idx in range(0, extra_branch_node_size):
            self.last_backward_insert_new_placeholders_cpy = copy.deepcopy(
                self.fuse_op_now_placeholders)
            # print("self.fuse_op_now_placeholders", self.fuse_op_now_placeholders)
            # print("generate_start_placeholder_list", self.generate_start_placeholder_list)

            self.last_backward_insert_new_placeholders_cpy = [i for i in self.last_backward_insert_new_placeholders_cpy
                                                              if i not in self.generate_start_placeholder_list]


            if self.last_backward_insert_new_placeholders_cpy == []:
                return True, self.generate_start_placeholder_list


            if self.try_insert_fusable_op("extra", extra_op_list, end_placeholder, each_retry_time):
                extra_cnt += 1

            if extra_cnt != now_extra_idx + 1:
                return False, None

        if extra_cnt != extra_branch_node_size:
            return False, None


        return True, self.generate_start_placeholder_list

    def try_fuse_phase_3(self, end_placeholder: str, nodes_allocation, each_retry_time=40) -> (bool, List[str]):
        # print("TVM fuse phase 3 start")
        main_branch_node_size, extra_branch_node_size, random_branch_node_size = nodes_allocation

        phase_2_main_branch_cnt = random.randint(1, main_branch_node_size - 3)
        phase_1_main_branch_cnt = main_branch_node_size - phase_2_main_branch_cnt
        phase_2_other_branch_node_cnt = random.randint(0, extra_branch_node_size - 2)
        phase_1_other_branch_node_cnt = extra_branch_node_size - phase_2_other_branch_node_cnt

        phase_1_success_flag = False
        phase_1_generate_start_placeholder_list = []
        for _ in range(0, each_retry_time):
            phase_1_success_flag, phase_1_generate_start_placeholder_list = \
                self.try_fuse_phase_2(end_placeholder=end_placeholder, nodes_allocation=nodes_allocation,
                                      each_retry_time=each_retry_time,
                                      toppest_must_ktuple=True)
            if phase_1_success_flag:
                break
            if not phase_1_success_flag:
                return False, None



        self.fuse_op_now_placeholders = copy.deepcopy(phase_1_generate_start_placeholder_list)
        for item in self.generate_start_placeholder_list:
            if item in self.fuse_input_var_list:
                self.fuse_input_var_list.remove(item)
        self.generate_start_placeholder_list = []



        mid_node_cnt = 0
        k_element_wise_list = get_pattern_kind_operator_list(TVMOpPatternKind.kElemWise)
        k_broadcast_op_list = get_pattern_kind_operator_list(TVMOpPatternKind.kBroadcast)
        k_injective_op_list = get_pattern_kind_operator_list(TVMOpPatternKind.kInjective)

        mid_op_list = k_element_wise_list + k_broadcast_op_list + k_injective_op_list
        self.last_backward_insert_new_placeholders = copy.deepcopy(
            phase_1_generate_start_placeholder_list)
        for now_branch_idx in range(1, phase_2_main_branch_cnt):
            self.backup()

            if self.try_insert_fusable_op("mid", mid_op_list, end_placeholder, each_retry_time):
                mid_node_cnt += 1

            if mid_node_cnt != now_branch_idx:
                return False, None

        if mid_node_cnt != phase_2_main_branch_cnt - 1:
            return False, None

        top_op_list = k_element_wise_list + k_broadcast_op_list + k_injective_op_list
        self.backup()
        if not self.try_insert_fusable_op("top", top_op_list, end_placeholder, each_retry_time):
            return False, None


        extra_cnt = 0
        extra_op_list = k_element_wise_list + k_broadcast_op_list + k_injective_op_list
        for now_extra_idx in range(0, phase_2_other_branch_node_cnt):
            self.last_backward_insert_new_placeholders_cpy = copy.deepcopy(self.fuse_op_now_placeholders)

            # print(f"self.fuse_op_now_placeholders_cpy: {self.fuse_op_now_placeholders}")
            # print(f"generate_start_placeholder_list: {self.generate_start_placeholder_list}")

            self.last_backward_insert_new_placeholders_cpy = [i for i in self.last_backward_insert_new_placeholders_cpy
                                                              if i not in self.generate_start_placeholder_list]

            if not self.last_backward_insert_new_placeholders_cpy:
                return True, self.generate_start_placeholder_list

            if self.try_insert_fusable_op("extra", extra_op_list, end_placeholder, each_retry_time):
                extra_cnt += 1

        if extra_cnt == phase_2_other_branch_node_cnt:
            return True, self.generate_start_placeholder_list
        else:
            return False, None

    def try_fuse_every_phase(self, end_placeholder: str, nodes_allocation, each_retry_time=40) -> (bool, List[str]):
        # print("TVM fuse phase every start")
        main_branch_node_size, extra_branch_node_size, random_branch_node_size = nodes_allocation
        # kBroadcast / kElementWise -> kElementWise / kBroadcast / kCommReduce / kInjective 中间kind <= kInjective

        end_inst_id, _ = InstIR.var_inst_idx(end_placeholder)
        end_inst = self.ir.find_inst_by_id(end_inst_id)

        self.fuse_op_now_placeholders.append(end_inst.retval())
        self.last_backward_insert_new_placeholders_cpy.append(end_inst.retval())

        # kBroadcast / kElementWise -> kElementWise / kBroadcast / kCommReduce / kInjective


        k_Broadcast_lp_list = get_pattern_kind_operator_list(TVMOpPatternKind.kBroadcast)
        k_element_wise_list = get_pattern_kind_operator_list(TVMOpPatternKind.kElemWise)
        k_common_reduce_list = get_pattern_kind_operator_list(TVMOpPatternKind.kCommReduce)
        k_injective_list = get_pattern_kind_operator_list(TVMOpPatternKind.kInjective)

        end_op_list = k_Broadcast_lp_list + k_element_wise_list + k_common_reduce_list + k_injective_list

        now_chs_dominator_added = False
        if not self.try_insert_fusable_op("end", end_op_list, end_placeholder, each_retry_time):
            return False, None

        mid_op_list = k_element_wise_list + k_Broadcast_lp_list + k_injective_list
        mid_node_cnt = 0
        for now_branch_idx in range(1, main_branch_node_size - 1):
            self.backup()

            if self.try_insert_fusable_op("mid", mid_op_list, end_placeholder, each_retry_time):
                mid_node_cnt += 1

            if mid_node_cnt != now_branch_idx:
                return False, None

        if mid_node_cnt != main_branch_node_size - 2:
            return False, None


        # 插入kBroadcast / kElementWise
        top_op_list = k_element_wise_list + k_Broadcast_lp_list
        self.backup()

        if not self.try_insert_fusable_op("top", top_op_list, end_placeholder, each_retry_time):
            return False, None

        extra_cnt = 0
        extra_op_list = k_element_wise_list + k_Broadcast_lp_list + k_injective_list
        for now_extra_idx in range(0, extra_branch_node_size):
            self.last_backward_insert_new_placeholders_cpy = copy.deepcopy(self.fuse_op_now_placeholders)
            # print(f"self.fuse_op_now_placeholders: {self.fuse_op_now_placeholders}")
            # print(f"generate_start_placeholder_list: {self.generate_start_placeholder_list}")

            self.last_backward_insert_new_placeholders_cpy = [i for i in self.last_backward_insert_new_placeholders_cpy
                                                              if i not in self.generate_start_placeholder_list]
            if not self.last_backward_insert_new_placeholders_cpy:
                return True, self.generate_start_placeholder_list

            if self.try_insert_fusable_op("extra", extra_op_list, end_placeholder, each_retry_time):
                extra_cnt += 1

            if extra_cnt != now_extra_idx + 1:
                return False, None

        if extra_cnt != extra_branch_node_size:
            return False, None

        return True, self.generate_start_placeholder_list

    def try_insert_fusable_op(self, state, op_list: List, end_placeholder: str,
                              try_times: int) -> bool:  # judge whether successful

        is_fuse_end_node = False if state == "end" else True
        is_main_branch_insert = False if state == "extra" else True
        must_through_node = random.choice(
            self.last_backward_insert_new_placeholders_cpy) if state != "end" else end_placeholder
        for idx in range(try_times):
            now_chs_element_opname = random.choice(op_list)

            now_chs_element_op = op_name_to_op_dict[now_chs_element_opname]
            is_state = self.fuse_op_try_backward_insert_node_type(now_chs_element_op,
                                                                  must_through_node=must_through_node,
                                                                  is_fuse_end_node=is_fuse_end_node,
                                                                  is_main_branch_insert=is_main_branch_insert)
            if is_state:
                return True
        print("fail to try the fusable op")
        return False

    def backup(self):
        self.last_backward_insert_new_placeholders_cpy = copy.deepcopy(self.last_backward_insert_new_placeholders)
        self.last_backward_insert_new_placeholders = []

    def fuse_op_try_backward_insert_node_type(self, now_chs_element_op: Type[AbsOpBase], max_tensor_pick_time=5,
                                              must_through_node=None, is_fuse_end_node=False,
                                              is_main_branch_insert=True):

        try:
            for _ in range(max_tensor_pick_time):
                op_param_n = now_chs_element_op.get_num_var_param()
                op_id = self.ir.n_inst()
                op_params = [
                    self.new_sym("op%s_%s" % (op_id, k)) for k in range(op_param_n)
                ]
                op: AbsOpBase = now_chs_element_op(*op_params)


                if is_main_branch_insert:
                    placeholders_candidates = [i for i in self.last_backward_insert_new_placeholders_cpy if
                                               i not in self.generate_start_placeholder_list]
                else:
                    placeholders_candidates = [i for i in self.fuse_op_now_placeholders if
                                               i not in self.generate_start_placeholder_list]

                phvars = self.select_var_group(

                    op.out_ranks,
                    op.out_dtypes,
                    var_candidates=[
                        name
                        for name in placeholders_candidates
                        if not isinstance(op, Expand)
                           or self.ir.vars[name].ndims < op.expand_last_dim
                    ],
                    must_through_node=must_through_node
                )
                otensors = [self.ir.vars[name] for name in phvars]
                phs_as_op_inputs: List[Placeholder] = []
                constraints = []
                for rank, dtype in op.deduct_inp_ranks_and_dtype(otensors):

                    ph = self.create_placeholder(
                        rank if rank != -1 else self.random_rank(), dtype=dtype
                    )
                    phs_as_op_inputs.append(ph)
                    constraints.extend(ph.ttype.gt_zero())
                itensors = [p.ttype for p in phs_as_op_inputs]
                constraints.extend(op.checked_requires(itensors))
                inferred_otensors = op.checked_type_transfer(itensors)
                for i, shape in enumerate(inferred_otensors):
                    constraints.extend(shape.eq(otensors[i]))
                    constraints.extend(shape.gt_zero())
                check_res = self.check_sat(*constraints)
                if check_res != z3.sat:

                    continue
                for c in constraints:
                    self.solver.add(c)
                input_vars = []
                for ph in phs_as_op_inputs:
                    new_inst = self.ir.add_inst(InstExpr(op=ph, args=[]))
                    if isinstance(ph, Placeholder):
                        self.placeholders.append(new_inst.retval())
                        self.fuse_op_now_placeholders.append(new_inst.retval())
                        self.last_backward_insert_new_placeholders.append(new_inst.retval())
                    input_vars.append(new_inst.retval())
                op.bind_input_like(itensors)
                op.bind_output_like(inferred_otensors)
                # self.backward_insert_node(node, input_vars, phvars)
                # new_inst = self.forward_insert_node(op, input_vars=input_vars)
                op.bind_node_position(
                    "main" if is_main_branch_insert else "extra")

                new_inst = self.ir.add_inst(InstExpr(op=op, args=input_vars))

                if isinstance(op, Placeholder):
                    assert isinstance(op, Placeholder) == False
                    # self.fuse_op_now_placeholders.append(new_inst.retval())
                for ph, rv in zip(phvars, new_inst.retvals()):

                    if ph in self.fuse_input_var_list:
                        self.fuse_input_var_list.remove(ph)
                        self.fuse_input_var_list.append(rv)
                    if ph in self.fuse_output_var_list:
                        assert ph not in self.fuse_output_var_list
                        self.fuse_output_var_list.remove(ph)
                    if ph in self.non_fuse_normal_var_list:
                        self.non_fuse_normal_var_list.remove(ph)
                    if is_fuse_end_node:
                        self.fuse_output_var_list.append(rv)



                    self.ir.replace_alluse(ph, rv)
                    ph_inst_id, _ = InstIR.var_inst_idx(ph)
                    ph_inst = self.ir.find_inst_by_id(ph_inst_id)
                    self.ir.remove_unused(ph_inst)
                    self.placeholders.remove(ph)
                    self.fuse_op_now_placeholders.remove(ph)
                    if ph in self.last_backward_insert_new_placeholders:
                        self.last_backward_insert_new_placeholders.remove(ph)


                return True
        except RequiredDimNotFound:
            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug(traceback.format_exc())
            return False
        except ConstraintError:
            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug(traceback.format_exc())
            return False
        return False

    def select_var_group(
            self,
            ndim_list: List[Set[int]],
            dtype_combs: List[Tuple[DType, ...]],
            var_candidates: Optional[List[str]] = None,
            must_through_node: Optional[str] = None,
    ) -> List[str]:
        """Randomly pick a group of variables that satisfy one of the `dtype_combs_spec` and `ndim_list`.

        Returns:
        List[str]: Satisfiable group of variable names.
        """
        if var_candidates is None:
            var_candidates = list(self.ir.vars.keys())

        must_through_node_inst = None
        if must_through_node is not None:
            must_through_node_inst_id, _ = InstIR.var_inst_idx(must_through_node)
            must_through_node_inst = self.ir.find_inst_by_id(must_through_node_inst_id)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f"Input data types candidates: {dtype_combs}")
            if must_through_node is not None:
                MGEN_LOG.debug(f"Must through node: {must_through_node}, Instance: {must_through_node_inst}")

        viable_dtypes = []
        for ndims in ndim_list:
            viable_dtypes.append(
                [
                    self.ir.vars[vname].dtype
                    for vname in self.var_filter(  # 这一步筛选出满足阶数的变量dtype
                    ndims=ndims, dtype=None, candidates=var_candidates
                )
                ]
            )

        dtype_combs = [
            comb for comb in dtype_combs
            if all(dt in viable_dtype for dt, viable_dtype in zip(comb, viable_dtypes))
        ]



        if must_through_node_inst:
            filter_dtype_combs = []
            must_through_node_dtypes = [self.ir.vars[vname].dtype for vname in var_candidates if
                                        vname in must_through_node_inst.retvals()]  # 找到必插点的dtypes
            for dtype_comb in dtype_combs:
                if any(dtype in dtype_comb for dtype in must_through_node_dtypes):
                    filter_dtype_combs.append(dtype_comb)
            dtype_combs = filter_dtype_combs

        if len(dtype_combs) == 0:
            raise ConstraintError(
                "No viable candidates: rank within {} and dtype within {}, with must_through node {}.".format(
                    ndim_list, dtype_combs, must_through_node
                )
            )

        random.shuffle(dtype_combs)
        for dtype_comb in dtype_combs:
            candidate_group = []
            for ndim, dtype in zip(ndim_list, dtype_comb):
                candidates = self.filter_rank_dtype(
                    ndims=ndims, dtype=dtype, candidates=var_candidates
                )
                if not candidates:
                    break
                candidate_group.append(random.choice(candidates))

        if len(candidate_group) == len(ndim_list):
            if must_through_node is None or any(vname in candidate_group for vname in must_through_node_inst.retvals()):
                return candidate_group

        raise ConstraintError("Cannot find desired combinations of tensor variables.")

    def var_filter(self, ndims, dtype, candidates: List[str]):
        cans = candidates
        # try:
        cans = list(
            filter(  # filter with ndim
                lambda vname: self.ir.vars[vname].ndims in ndims, cans
            )
        )
        # finally:
        # print(f"cans: {cans}\nir.vars: {self.ir.vars}")
        # print(f"input_var: {self.fuse_input_var_list}")
        if len(cans) == 0:
            raise RequiredDimNotFound(
                f"No variable with ndims in {ndims} found in {candidates}"
            )

        return cans

    def assume(self, c: z3.BoolRef):
        self.solver.add(c)

    def check_sat(self, *assumptions):
        cres = check_sat(self.solver, *assumptions)
        if cres == z3.sat:
            self.last_solution = self.solver.model()
        return cres

    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        itensors = [self.ir.vars[vname] for vname in input_vars]
        constraints = node.checked_requires(itensors)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        # make a copy
        otensors = node.checked_type_transfer(itensors)

        for aten in otensors:
            for c in aten.gt_zero():
                constraints.append(c)

        # limit output tensor size
        for aten in otensors:
            constraints.extend(self.tensor_type_constraints(aten))

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.assume(c)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {itensors}")
            MGEN_LOG.debug(f"\toutputs: {otensors}")

        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        self.forward_insert_node(node, input_vars)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {phvars} for node {node}"
            )
        # S2 - create X: X can be
        #                   - a new placeholder (fallback)
        #                   - an existing alive shape

        otensors = [self.ir.vars[name] for name in phvars]

        # S2.2: try to reuse some existing outputs;
        # TODO: allow reuse existing alive shapes
        # n_inps = len(node.inp_ranks)
        # max_try = 2
        # n_reuse = n_inps - 1
        # while n_reuse > 0 and max_try > 0:
        #     # TODO...
        #     max_try -= 1
        #     n_reuse -= 1

        # S2.2: reusing outputs failed. as a fallback, promote all free vars to placeholders.
        phs_as_op_inputs: List[Placeholder] = []
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(otensors):
            # oversample rank 4 tensors as they may be more important
            ph = self.make_symbolic_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(ph.ttype.gt_zero())
            constraints.extend(self.tensor_type_constraints(ph.ttype))

        itensors = [p.ttype for p in phs_as_op_inputs]
        constraints.extend(node.checked_requires(itensors))
        inferred_otensors = node.checked_type_transfer(itensors)

        for i, shape in enumerate(inferred_otensors):
            constraints.extend(shape.eq(otensors[i]))
            constraints.extend(shape.gt_zero())

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {phs_as_op_inputs}")

        for c in constraints:
            self.assume(c)

        # succ.
        input_vars = []

        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        node.bind_input_like(itensors)
        node.bind_output_like(inferred_otensors)

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def make_concrete(self) -> GraphIR:
        SanityCheck.gt(len(self.ir.insts), 0, "Empty graph!")

        SanityCheck.not_none(self.last_solution, "Run check_sat first!")
        self.ir.concretize(self.last_solution)
        return self.ir


class TorchFusedGen(BaseGen):
    def __init__(
            self,
            opset,
            seed=None,
            init_fp=False,
            symbolic_init=True,
            **kwargs,
    ):
        super().__init__(opset, seed, **kwargs)
        if seed is not None:
            set_z3_state(seed)

        self.solver = z3.Solver()
        self.last_solution: Optional[z3.ModelRef] = None

        generate_key_name_val_op_dict(opset)

        # Insert the first node.
        if symbolic_init:
            ph = self.make_symbolic_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        else:
            ph = self.make_random_concrete_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )

        self.insert_init_ph_node(ph)
        for pred in self.tensor_type_constraints(ph.ttype):
            self.assume(pred)

    def assume(self, c: z3.BoolRef):
        self.solver.add(c)

    def check_sat(self, *assumptions):
        cres = check_sat(self.solver, *assumptions)
        if cres == z3.sat:
            self.last_solution = self.solver.model()
        return cres

    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        itensors = [self.ir.vars[vname] for vname in input_vars]
        constraints = node.checked_requires(itensors)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        # make a copy
        otensors = node.checked_type_transfer(itensors)

        for aten in otensors:
            for c in aten.gt_zero():
                constraints.append(c)

        # limit output tensor size
        for aten in otensors:
            constraints.extend(self.tensor_type_constraints(aten))

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.assume(c)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {itensors}")
            MGEN_LOG.debug(f"\toutputs: {otensors}")

        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        self.forward_insert_node(node, input_vars)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {phvars} for node {node}"
            )
        # S2 - create X: X can be
        #                   - a new placeholder (fallback)
        #                   - an existing alive shape

        otensors = [self.ir.vars[name] for name in phvars]

        # S2.2: try to reuse some existing outputs;
        # TODO: allow reuse existing alive shapes
        # n_inps = len(node.inp_ranks)
        # max_try = 2
        # n_reuse = n_inps - 1
        # while n_reuse > 0 and max_try > 0:
        #     # TODO...
        #     max_try -= 1
        #     n_reuse -= 1

        # S2.2: reusing outputs failed. as a fallback, promote all free vars to placeholders.
        phs_as_op_inputs: List[Placeholder] = []
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(otensors):
            # oversample rank 4 tensors as they may be more important
            ph = self.make_symbolic_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(ph.ttype.gt_zero())
            constraints.extend(self.tensor_type_constraints(ph.ttype))

        itensors = [p.ttype for p in phs_as_op_inputs]
        constraints.extend(node.checked_requires(itensors))
        inferred_otensors = node.checked_type_transfer(itensors)

        for i, shape in enumerate(inferred_otensors):
            constraints.extend(shape.eq(otensors[i]))
            constraints.extend(shape.gt_zero())

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {phs_as_op_inputs}")

        for c in constraints:
            self.assume(c)

        # succ.
        input_vars = []

        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        node.bind_input_like(itensors)
        node.bind_output_like(inferred_otensors)

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def make_concrete(self) -> GraphIR:
        SanityCheck.gt(len(self.ir.insts), 0, "Empty graph!")
        SanityCheck.not_none(self.last_solution, "Run check_sat first!")
        self.ir.concretize(self.last_solution)
        return self.ir

    def abstract_gen(self, max_node_size=3, max_gen_millisec=2000):
        z3.set_param("timeout", max_gen_millisec // 3)

        init_time = time.time()

        while time.time() - init_time < max_gen_millisec / 1000 and self.num_op() < max_node_size:
            if self.extra_exit_check(max_node_size):
                break

            fuse_sequence_list = [
                # ["conv", "bn"],
                ["conv", "bn", "relu"],
                # ["conv", "relu"],
                # ["linear", "relu"],
                # ["bn", "relu"]
            ]

            conv = ["core.NCHWConv2d"]
            bn = ["core.BatchNorm2d"]
            relu = ["core.ReLU"]
            linear = ["torch.Linear"]

            random.seed(time.time())
            fuse_sequence = random.choice(fuse_sequence_list)
            print(fuse_sequence)
            for now_op_name_before in fuse_sequence:  # insert the fusable ops sequentially
                now_op_name = None
                if now_op_name_before == "conv":
                    now_op_name = random.choice(conv)
                elif now_op_name_before == "bn":
                    now_op_name = random.choice(bn)
                elif now_op_name_before == "relu":
                    now_op_name = random.choice(relu)
                elif now_op_name_before == "linear":
                    now_op_name = random.choice(linear)
                else:
                    assert now_op_name is not None, f"the now_op_name is {now_op_name}"
            now_op = op_name_to_op_dict[now_op_name]

            print(f" {now_op}")
            self.try_insert_node_type(now_op)

            if self.num_op() != len(fuse_sequence):
                continue

                # init graph placeholders
            SanityCheck.gt(len(self.placeholders), 0)

            def determine_ph_type(ph: str, to_input: bool):
                SanityCheck.true(ph in self.placeholders)
                ph_inst_id, _ = InstIR.var_inst_idx(ph)
                ph_inst = self.ir.find_inst_by_id(ph_inst_id)
                if to_input:
                    ph_inst.iexpr.op = ph_inst.iexpr.op.input()
                else:
                    ph_inst.iexpr.op = ph_inst.iexpr.op.const()

            determine_ph_type(self.placeholders[0], True)  # At lease make one input.
            for ph in self.placeholders[1:]:
                determine_ph_type(ph, random.randint(0, 1))


def model_gen(
        opset: Set[Type[AbsOpBase]],
        method: str = "TVMFusion",
        max_nodes=5,
        nodes_allocation=None,
        seed=None,
        allow_nnsmith_insert=False,
        timeout_ms=10000,
        **kwargs,
):
    assert max_nodes > 0, "max_nodes must >= 1"

    symbolic_init = not method.endswith("-cinit")
    if method.startswith("symbolic"):
        gen = SymbolicGen(opset, seed, symbolic_init=symbolic_init, **kwargs)
    elif "concolic" == method:
        gen = ConcolicGen(opset, seed, **kwargs)
    elif method.startswith("single-io"):
        gen = SymboliSingleIOGen(opset, seed, symbolic_init=symbolic_init, **kwargs)
    elif method == "TVMFusion":
        if nodes_allocation is None:
            nodes_allocation = [3, 2, 1]
        gen = TVMFusedGen(opset, nodes_allocation=nodes_allocation, seed=42, allow_nnsmith_insert=allow_nnsmith_insert,
                          **kwargs)
    elif method == "TorchFusion":
        gen = TorchFusedGen(opset)
    else:
        raise NotImplementedError(f"the method {method} is not implemented")
        # raise ValueError(f"Unknown method {method}. Try `symbolic` or `concolic`.")

    gen.abstract_gen(max_node_size=max_nodes)  # max_node_size=max_nodes, max_gen_millisec=timeout_ms

    return gen


def viz(ir: GraphIR, filename: str = None):
    if HAS_PYGRAPHVIZ:
        viz_dot(ir.to_dot(), filename)


def count_op_types(file_path, ir: GraphIR):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    for inst in ir.insts:
        op_name = inst.iexpr.op.name()
        if op_name in data:
            data[op_name] += 1
        else:
            data[op_name] = 1

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
