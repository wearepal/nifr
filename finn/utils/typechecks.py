def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_probability(x):
    return 0.0 <= x <= 1.0


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False


def is_nan(tensor):
    return tensor != tensor
