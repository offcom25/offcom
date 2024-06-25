from enum import Enum, unique

from offcom.abstract.op import AbsOpBase, UnaryOpBase, ElementWiseUnaryOp, BcastBinaryOp, Pool2d, MaxPool2d, Constant


@unique
class TVMOpPatternKind(Enum):  # https://github.com/apache/tvm/blob/main/include/tvm/relay/op_attr_types.h
    # Elementwise operation
    kElemWise = 0
    # Broadcasting operator, can always map output axis to the input in order.
    # for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
    # Note that the axis need to be in order so transpose is not a bcast operator.
    kBroadcast = 1
    # Injective operator, can always injectively map output axis to a single input axis.
    # All injective operator can still be safely fused to injective and reduction.
    kInjective = 2
    # Communicative reduction operator.
    kCommReduce = 3
    # Complex operation, can still fuse elemwise operations into its output.
    # but cannot chain another complex op
    kOutEWiseFusable = 4
    # The pattern for tuple nodes. Can fuse into subsequent injective ops,
    # but treated specially
    kTuple = 7
    # Opaque operation, cannot fuse anything.
    kOpaque = 8


operator_pattern_dict = {
    # "Input": TVMOpPatternKind.kOpaque,
    # "Constant": TVMOpPatternKind.kElemWise,
    "core.ReLU": TVMOpPatternKind.kElemWise,
    "core.GELU": TVMOpPatternKind.kElemWise,
    "core.LeakyReLU": TVMOpPatternKind.kElemWise,
    "core.PReLU": TVMOpPatternKind.kBroadcast,
    "core.Sigmoid": TVMOpPatternKind.kElemWise,
    "core.Sin": TVMOpPatternKind.kElemWise,
    "core.Cos": TVMOpPatternKind.kElemWise,
    "core.Asin": TVMOpPatternKind.kElemWise,
    "core.Acos": TVMOpPatternKind.kElemWise,
    "core.Tan": TVMOpPatternKind.kElemWise,
    "core.Atan": TVMOpPatternKind.kElemWise,
    "core.Abs": TVMOpPatternKind.kElemWise,
    "core.Where": TVMOpPatternKind.kBroadcast,
    "core.Add": TVMOpPatternKind.kBroadcast,
    "core.Sub": TVMOpPatternKind.kBroadcast,
    "core.Mul": TVMOpPatternKind.kBroadcast,
    "core.Div": TVMOpPatternKind.kBroadcast,
    "core.Max": TVMOpPatternKind.kBroadcast,
    "core.Min": TVMOpPatternKind.kBroadcast,
    "core.Equal": TVMOpPatternKind.kBroadcast,
    "core.Greater": TVMOpPatternKind.kBroadcast,
    "core.Less": TVMOpPatternKind.kBroadcast,
    "core.And": TVMOpPatternKind.kBroadcast,
    "core.Or": TVMOpPatternKind.kBroadcast,
    "core.Xor": TVMOpPatternKind.kBroadcast,
    "core.Pow": TVMOpPatternKind.kBroadcast,
    "core.Floor": TVMOpPatternKind.kBroadcast,
    "core.Ceil": TVMOpPatternKind.kElemWise,
    "core.Clip": TVMOpPatternKind.kElemWise,
    "core.Round": TVMOpPatternKind.kElemWise,
    "core.Sqrt": TVMOpPatternKind.kElemWise,
    "core.Log2": TVMOpPatternKind.kElemWise,
    "core.Neg": TVMOpPatternKind.kElemWise,
    "core.Softmax": TVMOpPatternKind.kOutEWiseFusable,
    "core.MaxPool2d": TVMOpPatternKind.kOutEWiseFusable,
    "core.AvgPool2d": TVMOpPatternKind.kOutEWiseFusable,
    "core.Slice": TVMOpPatternKind.kInjective,
    "core.ConstPad": TVMOpPatternKind.kInjective,
    "core.ReflectPad": TVMOpPatternKind.kInjective,
    "core.ReplicatePad": TVMOpPatternKind.kInjective,
    "core.ExpandLast1": TVMOpPatternKind.kBroadcast,
    "core.ExpandLast2": TVMOpPatternKind.kBroadcast,
    "core.ExpandLast3": TVMOpPatternKind.kBroadcast,
    "core.ExpandLast4": TVMOpPatternKind.kBroadcast,
    "core.BatchNorm2d": TVMOpPatternKind.kOutEWiseFusable,
    "core.Conv1d": TVMOpPatternKind.kOutEWiseFusable,
    "core.NCHWConv2d": TVMOpPatternKind.kOutEWiseFusable,
    "core.Reshape": TVMOpPatternKind.kInjective,
    "torch.Flatten": TVMOpPatternKind.kInjective,
    "core.Transpose": TVMOpPatternKind.kInjective,
    "core.NearestInterp": TVMOpPatternKind.kOpaque,
    "core.LinearInterp": TVMOpPatternKind.kOpaque,
    "core.BilinearInterp": TVMOpPatternKind.kOpaque,
    "core.BicubicInterp": TVMOpPatternKind.kOpaque,
    "core.TrilinearInterp": TVMOpPatternKind.kOpaque,
    "core.Squeeze": TVMOpPatternKind.kInjective,
    "torch.TorchReduceSum": TVMOpPatternKind.kCommReduce,
    "core.ReduceMin": TVMOpPatternKind.kCommReduce,
    "core.ReduceMax": TVMOpPatternKind.kCommReduce,
    "core.ReduceMean": TVMOpPatternKind.kCommReduce,
    "core.ArgMin": TVMOpPatternKind.kCommReduce,
    "core.ArgMax": TVMOpPatternKind.kCommReduce,
    "core.Tril": TVMOpPatternKind.kElemWise,
    "core.Triu": TVMOpPatternKind.kElemWise,
    "torch.Linear": TVMOpPatternKind.kOutEWiseFusable,
    "core.Concat1": TVMOpPatternKind.kInjective,
    "core.Concat2": TVMOpPatternKind.kInjective,
    "core.Concat3": TVMOpPatternKind.kInjective,
    "core.Concat4": TVMOpPatternKind.kInjective,
    "core.Concat5": TVMOpPatternKind.kInjective,
    "core.CastBool": TVMOpPatternKind.kElemWise,
    "core.CastF32": TVMOpPatternKind.kElemWise,
    "core.CastF64": TVMOpPatternKind.kElemWise,
    "core.CastI32": TVMOpPatternKind.kElemWise,
    "core.CastI64": TVMOpPatternKind.kElemWise,
    "core.MatMul": TVMOpPatternKind.kOutEWiseFusable,

}

pattern_kind_op_dict = {
    TVMOpPatternKind.kElemWise: [],
    TVMOpPatternKind.kBroadcast: [],
    TVMOpPatternKind.kInjective: [],
    TVMOpPatternKind.kCommReduce: [],
    TVMOpPatternKind.kOutEWiseFusable: [],
    TVMOpPatternKind.kTuple: [],
    TVMOpPatternKind.kOpaque: [],
}

op_name_to_op_dict = {}  # {op_name: op_instance}

for k, v in operator_pattern_dict.items():
    pattern_kind_op_dict[v].append(k)

def get_op_pattern_kind(op: AbsOpBase):
    op_name = op.name()
    assert op_name in operator_pattern_dict
    return operator_pattern_dict[op_name]


def get_pattern_kind_operator_list(pattern_kind: TVMOpPatternKind):
    assert pattern_kind in TVMOpPatternKind
    return [key for key, value in op_name_to_op_dict.items()]


def generate_key_name_val_op_dict(opset):
    for now_op in opset:
        op_name = now_op.name()
        op_name_to_op_dict[op_name] = now_op
