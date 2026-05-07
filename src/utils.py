import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _remove_boxed(s: str) -> str | None:
    """Strip a leading \\boxed{...} wrapper and return inner content."""
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract the rightmost non-empty \\boxed{...} or \\fbox{...} answer.

    Searches right-to-left so we skip placeholder boxes like ``\\boxed{{}}``.
    """
    candidates = []
    for macro in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(macro, start)
            if idx < 0:
                break
            candidates.append(idx)
            start = idx + 1

    if not candidates:
        return None

    for idx in sorted(candidates, reverse=True):
        i = idx
        while i < len(text) and text[i] != "{":
            i += 1
        if i >= len(text):
            continue

        right_brace_idx = None
        num_left_braces_open = 0
        j = i
        while j < len(text):
            if text[j] == "{":
                num_left_braces_open += 1
            if text[j] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = j
                    break
            j += 1

        if right_brace_idx is None:
            continue

        retval = text[idx : right_brace_idx + 1]
        content = _remove_boxed(retval) if retval.startswith("\\boxed{") else retval

        if (
            content is not None
            and content.strip().replace("{", "").replace("}", "").strip() != ""
        ):
            return _remove_boxed(retval) if retval.startswith("\\boxed{") else content

    return None


# ---------------------------------------------------------------------------
# Hendrycks MATH normalization (from math_normalize.py in Power-SMC)
# ---------------------------------------------------------------------------


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
        string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _normalize_hendrycks(answer: str | None) -> str | None:
    """Hendrycks MATH normalization (math_equivalence)."""
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


# ---------------------------------------------------------------------------
# Grader normalization + sympy equivalence (from math_grader.py in Power-SMC)
# ---------------------------------------------------------------------------

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize_grader(expr: str) -> str | None:
    """Secondary normalization from the Power-SMC math grader."""
    if expr is None:
        return None

    m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def _count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def _should_allow_eval(expr: str):
    if _count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def _are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if _should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def _split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def is_equiv(pred: str, gold: str) -> bool:
    """Check equivalence using the Power-SMC reference grader.

    Two-tier normalization:
    1. Hendrycks MATH normalization (fast string match)
    2. Grader normalization + sympy simplification (fallback)
    """
    if pred is None:
        return False

    # Tier 1: Hendrycks normalization
    gold_normalized_h = _normalize_hendrycks(gold)
    pred_normalized_h = _normalize_hendrycks(pred)
    if gold_normalized_h == pred_normalized_h:
        return True

    # Tier 2: Grader normalization + sympy
    gold_normalized = _normalize_grader(gold)
    pred_normalized = _normalize_grader(pred)

    if gold_normalized is None:
        return False
    if gold_normalized == pred_normalized:
        return True
    if len(pred_normalized) == 0:
        return False

    gold_elems = _split_tuple(gold_normalized)
    pred_elems = _split_tuple(pred_normalized)

    if len(gold_elems) > 1 and (
        gold_normalized[0] != pred_normalized[0]
        or gold_normalized[-1] != pred_normalized[-1]
    ):
        return False
    elif len(gold_elems) != len(pred_elems):
        return False
    else:
        for gold_elem, pred_elem in zip(gold_elems, pred_elems):
            if _is_frac(gold_elem) and _is_frac(pred_elem):
                if gold_elem != pred_elem:
                    return False
            elif _str_is_int(gold_elem) != _str_is_int(pred_elem):
                return False
            else:
                if not _are_equal_under_sympy(gold_elem, pred_elem):
                    return False

    return True


# ---------------------------------------------------------------------------
# Dataset loaders – each returns list[dict] with keys:
#   problem, answer, level (int), subject, unique_id (optional)
# ---------------------------------------------------------------------------

DATASET_REGISTRY_EVAL: dict[str, callable] = {}
DATASET_REGISTRY_TRAIN: dict[str, callable] = {}


def register_dataset_eval(name):
    def wrapper(fn):
        DATASET_REGISTRY_EVAL[name] = fn
        return fn
    return wrapper

def register_dataset_train(name):
    def wrapper(fn):
        DATASET_REGISTRY_TRAIN[name] = fn
        return fn
    return wrapper


@register_dataset_eval("minerva_math")
def load_minerva_math(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("math-ai/minervamath", split="test")
    out = []
    for row in ds:
        out.append({
            "problem": row["question"],
            "answer": row["answer"],
            "solution": "",
            "level": 0,
            "subject": "",
            "unique_id": "",
        })
    return out


@register_dataset_eval("aime_2025")
def load_aime_2025(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("MathArena/aime_2025", split="train")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": "",
            "level": 0,
            "subject": ", ".join(row["problem_type"]),
            "unique_id": f"aime2025_{row['problem_idx']}",
        })
    return out


@register_dataset_eval("hmmt_feb_2025")
def load_hmmt_feb_2025(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("MathArena/hmmt_feb_2025", split="train")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": "",
            "level": 0,
            "subject": ", ".join(row["problem_type"]),
            "unique_id": f"hmmt_feb2025_{row['problem_idx']}",
        })
    return out


@register_dataset_eval("aime24")
def load_aime24(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("math-ai/aime24", split="test")
    out = []
    for row in ds:
        answer = extract_boxed_answer(row["solution"]) or ""
        out.append({
            "problem": row["problem"],
            "answer": answer,
            "solution": row["solution"],
            "level": 0,
            "subject": "",
            "unique_id": f"aime24_{row['id']}",
        })
    return out


@register_dataset_eval("aime25")
def load_aime25(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("math-ai/aime25", split="test")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": "",
            "level": 0,
            "subject": "",
            "unique_id": f"aime25_{row['id']}",
        })
    return out


@register_dataset_eval("aime26")
def load_aime26(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("MathArena/aime_2026", split="train")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": "",
            "level": 0,
            "subject": "",
            "unique_id": f"aime26_{row['problem_idx']}",
        })
    return out


@register_dataset_eval("math500")
def load_math500(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for row in ds:
        level = int(str(row["level"]).removeprefix("Level "))
        if levels and level not in levels:
            continue
        out.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "solution": row["solution"],
            "level": level,
            "subject": row["subject"],
            "unique_id": row.get("unique_id", ""),
        })
    return out


@register_dataset_eval("deepmath")
def load_deepmath_eval(levels: list[int] | None = None) -> list[dict]:
    """Load zwhe99/DeepMath-103K for evaluation (unique questions only)."""
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    out = []
    for i, row in enumerate(ds):
        out.append({
            "problem": row["question"],
            "answer": row["final_answer"],
            "solution": "",
            "level": 0,
            "subject": row.get("subject", ""),
            "unique_id": f"deepmath_{i}",
        })
    return out


@register_dataset_eval("polaris")
def load_polaris_eval(levels: list[int] | None = None) -> list[dict]:
    """Load POLARIS-Project/Polaris-Dataset-53K for evaluation."""
    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
    out = []
    for i, row in enumerate(ds):
        if row["answer"] is None or len(row["answer"].strip()) == 0:
            continue
        out.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "solution": "",
            "level": 0,
            "subject": row.get("difficulty", ""),
            "unique_id": f"polaris_{i}",
        })
    return out


@register_dataset_train("deepmath")
def load_deepmath(
    max_samples: int | None = None,
    seed: int = 42,
    explode_solutions: bool = True,
) -> "Dataset":
    """Load zwhe99/DeepMath-103K.

    When *explode_solutions* is True (default, for SFT), each example is
    tripled: one row per r1_solution_{1,2,3} with columns problem/solution/answer.

    When *explode_solutions* is False (for GRPO/RL), only problem and answer
    are kept (no solution column, no tripling).
    """
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")

    if explode_solutions:
        from datasets import concatenate_datasets

        def _make_split(sol_col):
            return ds.map(
                lambda x: {"problem": x["question"], "solution": x[sol_col], "answer": x["final_answer"]},
                remove_columns=ds.column_names,
                num_proc=4,
            )

        ds = concatenate_datasets([
            _make_split("r1_solution_1"),
            _make_split("r1_solution_2"),
            _make_split("r1_solution_3"),
        ])

        ds = ds.filter(
            lambda x: x["solution"] is not None and len(x["solution"].strip()) > 0,
            num_proc=4,
        )
    else:
        ds = ds.map(
            lambda x: {"problem": x["question"], "answer": x["final_answer"]},
            remove_columns=ds.column_names,
            num_proc=4,
        )
        ds = ds.filter(
            lambda x: x["answer"] is not None and len(x["answer"].strip()) > 0,
            num_proc=4,
        )

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


@register_dataset_train("openthoughts")
def load_openthoughts(
    max_samples: int | None = None,
    seed: int = 42,
) -> "Dataset":
    """Load OpenThoughts-114k (metadata subset), filtered to math domain.

    Maps deepseek_reasoning -> solution, extracts boxed answer from
    deepseek_solution (falls back to ground_truth_solution).
    """
    ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")

    # Filter to math domain
    ds = ds.filter(lambda x: x["domain"] == "math", num_proc=4)

    # Map to standard columns
    def _map_columns(example):
        # Try extracting boxed answer from deepseek_solution first
        answer = extract_boxed_answer(example["deepseek_solution"] or "")
        if not answer and example.get("ground_truth_solution"):
            answer = extract_boxed_answer(example["ground_truth_solution"])
        return {
            "problem": example["problem"],
            "solution": example["deepseek_reasoning"],
            "answer": answer or "",
        }

    ds = ds.map(_map_columns, remove_columns=ds.column_names, num_proc=4)

    # Drop rows with empty solution or answer
    ds = ds.filter(
        lambda x: (
            x["solution"] is not None
            and len(x["solution"].strip()) > 0
            and x["answer"] is not None
            and len(x["answer"].strip()) > 0
        ),
        num_proc=4,
    )

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


@register_dataset_train("numinamath")
def load_numinamath(
    max_samples: int | None = None,
    sources: list[str] | None = None,
    seed: int = 42,
) -> "Dataset":
    """Load NuminaMath-1.5 with optional source filtering."""
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")

    ds = ds.filter(
        lambda x: (
            x["problem_is_valid"] == "Yes"
            and x["answer"] is not None
            and len(x["answer"].strip()) > 0
        ),
        num_proc=4,
    )

    if sources:
        ds = ds.filter(lambda x: x["source"] in sources, num_proc=4)

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


@register_dataset_train("polaris")
def load_polaris(
    max_samples: int | None = None,
    difficulty: list[str] | None = None,
    seed: int = 42,
) -> "Dataset":
    """Load POLARIS-Project/Polaris-Dataset-53K with optional difficulty filtering."""
    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")

    ds = ds.filter(
        lambda x: x["answer"] is not None and len(x["answer"].strip()) > 0,
        num_proc=4,
    )

    if difficulty:
        ds = ds.filter(lambda x: x["difficulty"] in difficulty, num_proc=4)

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds
