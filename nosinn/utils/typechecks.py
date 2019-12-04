from typing import Any
from torch import Tensor

__all__ = [
    "is_bool",
    "is_int",
    "is_nan",
    "is_nonnegative_int",
    "is_positive_int",
    "is_power_of_two",
    "is_probability",
]


def is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def is_int(x: Any) -> bool:
    return isinstance(x, int)


def is_probability(x: float) -> bool:
    return 0.0 <= x <= 1.0


def is_positive_int(x: float) -> bool:
    return is_int(x) and x > 0


def is_nonnegative_int(x: float) -> bool:
    return is_int(x) and x >= 0


def is_power_of_two(n: float) -> bool:
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False


def is_nan(tensor: Tensor):
    return tensor != tensor
