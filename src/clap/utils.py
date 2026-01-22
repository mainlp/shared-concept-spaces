import ast


def safe_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    return val


def ulist(lst):
    """
    Returns a list with unique elements from the input list.
    """
    # as of py 3.6 dict.fromkeys preserves order, unlike list(set(lst))
    return list(dict.fromkeys(lst))
