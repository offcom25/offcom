from enum import Enum, unique

import numpy as np


@unique
class DType(Enum):
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    uint8 = "uint8"  # Support quantized models.
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    bool = "bool"
    complex64 = "complex64"
    complex128 = "complex128"

    dtuple_f16 = "dtuple_f16"
    dtuple_f32 = "dtuple_f32"
    dtuple_f64 = "dtuple_f64"
    dtuple_u8 = "dtuple_u8"
    dtuple_i8 = "dtuple_i8"
    dtuple_i16 = "dtuple_i16"
    dtuple_i32 = "dtuple_i32"
    dtuple_i64 = "dtuple_i64"
    dtuple_bool = "dtuple_bool"
    dtuple_complex64 = "dtuple_complex64"
    dtuple_complex128 = "dtuple_complex128"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        s = super().__str__()
        assert s.startswith("DType."), s
        return s[len("DType."):]

    def short(self) -> str:
        return {
            DType.float16: "f16",
            DType.float32: "f32",
            DType.float64: "f64",
            DType.uint8: "u8",
            DType.uint16: "u16",
            DType.uint32: "u32",
            DType.uint64: "u64",
            DType.int8: "i8",
            DType.int16: "i16",
            DType.int32: "i32",
            DType.int64: "i64",
            DType.complex64: "c64",
            DType.complex128: "c128",
            DType.bool: "b",

            DType.dtuple_f16: "dtuple_f16",
            DType.dtuple_f32: "dtuple_f32",
            DType.dtuple_f64: "dtuple_f64",
            DType.dtuple_u8: "dtuple_u8",
            DType.dtuple_i8: "dtuple_i8",
            DType.dtuple_i16: "dtuple_i16",
            DType.dtuple_i32: "dtuple_i32",
            DType.dtuple_i64: "dtuple_i64",
            DType.dtuple_bool: "dtuple_bool",
            DType.dtuple_complex64: "dtuple_complex64",
            DType.dtuple_complex128: "dtuple_complex128"
        }[self]

    def is_float(self):
        return self in [DType.float16, DType.float32, DType.float64]

    @staticmethod
    def from_str(s):
        return {
            "f16": DType.float16,
            "f32": DType.float32,
            "f64": DType.float64,
            "u8": DType.uint8,
            "i8": DType.int8,
            "i32": DType.int32,
            "i64": DType.int64,
            "c64": DType.complex64,
            "c128": DType.complex128,
            "float16": DType.float16,
            "float32": DType.float32,
            "float64": DType.float64,
            "uint8": DType.uint8,
            "uint16": DType.uint16,
            "uint32": DType.uint32,
            "uint64": DType.uint64,
            "int8": DType.int8,
            "int16": DType.int16,
            "int32": DType.int32,
            "int64": DType.int64,
            "complex64": DType.complex64,
            "complex128": DType.complex128,
            "bool": DType.bool,

            "dtuple_f16": DType.dtuple_f16,
            "dtuple_f32": DType.dtuple_f32,
            "dtuple_f64": DType.dtuple_f64,
            "dtuple_u8": DType.dtuple_u8,
            "dtuple_i8": DType.dtuple_i8,
            "dtuple_i16": DType.dtuple_i16,
            "dtuple_i32": DType.dtuple_i32,
            "dtuple_i64": DType.dtuple_i64,
            "dtuple_bool": DType.dtuple_bool,
            "dtuple_complex64": DType.dtuple_complex64,
            "dtuple_complex128": DType.dtuple_complex128
        }[s]

    def numpy(self):
        return {
            DType.float16: np.float16,
            DType.float32: np.float32,
            DType.float64: np.float64,
            DType.uint8: np.uint8,
            DType.uint8: np.uint8,
            DType.uint16: np.uint16,
            DType.uint32: np.uint32,
            DType.uint64: np.uint64,
            DType.int8: np.int8,
            DType.int16: np.int16,
            DType.int32: np.int32,
            DType.int64: np.int64,
            DType.complex64: np.complex64,
            DType.complex128: np.complex128,
            DType.bool: np.bool_,

        }[self]

    def torch(self) -> "torch.dtype":
        import torch

        return {
            DType.float16: torch.float16,
            DType.float32: torch.float32,
            DType.float64: torch.float64,
            DType.uint8: torch.uint8,
            # PyTorch does not support other unsigned int types: https://github.com/pytorch/pytorch/issues/58734

            DType.uint16: torch.uint16,
            DType.uint32: torch.uint32,
            DType.uint64: torch.uint64,
            DType.int8: torch.int8,
            DType.int16: torch.int16,
            DType.int32: torch.int32,
            DType.int64: torch.int64,
            DType.complex64: torch.complex64,
            DType.complex128: torch.complex128,
            DType.bool: torch.bool,

            DType.dtuple_f16: tuple,
            DType.dtuple_f32: tuple,
            DType.dtuple_f64: tuple,
            DType.dtuple_u8: tuple,
            DType.dtuple_i8: tuple,
            DType.dtuple_i16: tuple,
            DType.dtuple_i32: tuple,
            DType.dtuple_i64: tuple,
            DType.dtuple_bool: tuple,
            DType.dtuple_complex64: tuple,
            DType.dtuple_complex128: tuple,

        }[self]

    @staticmethod
    def from_torch(dtype) -> "DType":
        import torch

        return {
            torch.float16: DType.float16,
            torch.float32: DType.float32,
            torch.float64: DType.float64,
            torch.uint8: DType.uint8,
            torch.uint16: DType.uint16,
            torch.uint32: DType.uint32,
            torch.uint64: DType.uint64,
            torch.int8: DType.int8,
            torch.int16: DType.int16,
            torch.int32: DType.int32,
            torch.int64: DType.int64,
            torch.complex64: DType.complex64,
            torch.complex128: DType.complex128,
            torch.bool: DType.bool,
        }[dtype]

    def tensorflow(self) -> "tf.Dtype":
        import tensorflow as tf

        return {
            DType.float16: tf.float16,
            DType.float32: tf.float32,
            DType.float64: tf.float64,
            DType.uint8: tf.uint8,
            DType.uint16: tf.uint16,
            DType.uint32: tf.uint32,
            DType.uint64: tf.uint64,
            DType.int8: tf.int8,
            DType.int16: tf.int16,
            DType.int32: tf.int32,
            DType.int64: tf.int64,
            DType.complex64: tf.complex64,
            DType.complex128: tf.complex128,
            DType.bool: tf.bool,
        }[self]

    @staticmethod
    def from_tensorflow(dtype) -> "DType":
        import tensorflow as tf

        return {
            tf.float16: DType.float16,
            tf.float32: DType.float32,
            tf.float64: DType.float64,
            tf.uint8: DType.uint8,
            tf.uint16: DType.uint16,
            tf.uint32: DType.uint32,
            tf.uint64: DType.uint64,
            tf.int8: DType.int8,
            tf.int16: DType.int16,
            tf.int32: DType.int32,
            tf.int64: DType.int64,
            tf.complex64: DType.complex64,
            tf.complex128: DType.complex128,
            tf.bool: DType.bool,
        }[dtype]

    def sizeof(self) -> int:
        return {
            DType.float16: 2,
            DType.float32: 4,
            DType.float64: 8,
            DType.uint8: 1,
            DType.uint16: 2,
            DType.uint32: 4,
            DType.uint64: 8,
            DType.int8: 1,
            DType.int16: 2,
            DType.int32: 4,
            DType.int64: 8,
            DType.complex64: 8,
            DType.complex128: 16,
            DType.bool: 1,  # Follow C/C++ convention.
        }[self]


# "DTYPE_GEN*" means data types used for symbolic generation.
# "DTYPE_GEN_ALL" is surely a subset of all types but it is
# used to conservatively to avoid unsupported data types while
# applying offcom to various frameworks.
DTYPE_ALL = [
    DType.float32,
    DType.float64,
    DType.int32,
    DType.int64,
    DType.bool,
]

DTYPE_NON_BOOLS = [dtype for dtype in DTYPE_ALL if dtype != DType.bool]

DTYPE_FLOATS = [DType.float32, DType.float64]
DTYPE_INTS = [DType.int32, DType.int64]

DTYPE_TUPLE = [
    DType.dtuple_f32,
    DType.dtuple_f64,
    DType.dtuple_i32,
    DType.dtuple_i64,
    DType.dtuple_bool
]

DTYPE_GEN_FLOATS = [DType.float16, DType.float32, DType.float64]
DTYPE_GEN_INTS = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]
DTYPE_GEN_COMPLEX = [DType.complex64, DType.complex128]


DTYPE_GEN_ALL = DTYPE_GEN_FLOATS + DTYPE_GEN_INTS + DTYPE_GEN_COMPLEX
DTYPE_GEN_TUPLE_ALL = DTYPE_GEN_ALL + DTYPE_TUPLE
DTYPE_GEN_NON_BOOL = [dtype for dtype in DTYPE_GEN_ALL if dtype != DType.bool]
