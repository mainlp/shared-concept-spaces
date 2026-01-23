import ast
import json
from pathlib import Path


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


def load_dict(file_name):
    with open(file_name, "r") as f:
        json_dic = json.load(f)
    return json_dic


def is_empty_json_file(file_path: Path) -> bool:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
            if content.strip() == "":
                return True
            data = json.loads(content)
            return len(data) == 0  # Empty dict means empty JSON
    except Exception:
        return False  # If file can't be read or parsed, it's not "empty"
