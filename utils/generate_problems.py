import random
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLOR_CONST = {
    'red':    [1, 0, 0, 1],
    'green':  [0, 1, 0, 1],
    'blue':   [0, 0, 1, 1],
    'white':  [1, 1, 1, 1],
    'brown':  [0.396, 0.263, 0.129, 1],
    'grey':   [0.5, 0.5, 0.5, 1],
    'yellow': [1, 1, 0, 1],
    'cyan':   [0, 1, 1, 1],
    'magenta': [1, 0, 1, 1],
}


def _partition_even(lst: List[str], k: int) -> List[List[str]]:
    n = len(lst)
    if k <= 0:
        raise ValueError("stack must be >= 1")
    sizes = [n // k + (1 if i < (n % k) else 0) for i in range(k)]
    parts, idx = [], 0
    for s in sizes:
        parts.append(lst[idx:idx+s])
        idx += s
    return parts


def blocksworld_pddl_problem(
    num: int,
    stack: int,
    distractor: int,
    seed: Optional[int] = None,
    domain_name: str = "blocksworld-original",
    problem_name: str = "prob",
) -> str:
    if seed is not None:
        random.seed(seed)

    if stack not in (1, 2, 3):
        raise ValueError("stack must be 1, 2, or 3")

    color_names = list(COLOR_CONST.keys())
    if num < 1:
        raise ValueError("num must be >= 1")
    if num > len(color_names):
        raise ValueError(f"num must be <= {len(color_names)} (available distinct colors)")

    chosen = random.sample(color_names, num)

    init_stacks: List[List[str]] = _partition_even(chosen, stack)

    goal_buckets: List[List[str]] = [[] for _ in range(stack)]
    for name in chosen:
        bucket_idx = random.randrange(stack)
        goal_buckets[bucket_idx].append(name)
    for b in goal_buckets:
        random.shuffle(b)

    objects: List[str] = chosen[:]
    if distractor:
        blacks = [f"black{i+1}" for i in range(distractor)]
        objects.extend(blacks)

    init_facts: List[str] = []
    init_facts.append("(arm-empty)")

    for grp in init_stacks:
        if not grp:
            continue
        base = grp[0]
        init_facts.append(f"(on-table {base})")
        for i in range(1, len(grp)):
            init_facts.append(f"(on {grp[i]} {grp[i-1]})")
        init_facts.append(f"(clear {grp[-1]})")

    if distractor:
        for b in blacks:
            init_facts.append(f"(on-table {b})")

    goal_facts: List[str] = []
    for grp in goal_buckets:
        if not grp:
            continue
        goal_facts.append(f"(on-table {grp[0]})")
        for i in range(1, len(grp)):
            goal_facts.append(f"(on {grp[i]} {grp[i - 1]})")

    def _block(lines: List[str], indent: int = 2) -> str:
        pad = " " * indent
        return "\n".join(pad + ln for ln in lines)

    pddl_lines: List[str] = []
    pddl_lines.append(f"(define (problem {problem_name})")
    pddl_lines.append(f"  (:domain {domain_name})")
    pddl_lines.append("  (:objects")
    pddl_lines.append("    " + " ".join(objects))
    pddl_lines.append("  )")
    pddl_lines.append("  (:init")
    pddl_lines.append(_block(init_facts, indent=4))
    pddl_lines.append("  )")
    pddl_lines.append("  (:goal")
    pddl_lines.append("    (and")
    pddl_lines.append(_block(goal_facts, indent=6))
    pddl_lines.append("    )")
    pddl_lines.append("  )")
    pddl_lines.append(")")

    pddl_str = "\n".join(pddl_lines)

    with open(f"../experiments/blocksworld_pr/problem/{problem_name}.pddl", "w", encoding="utf-8") as f:
        f.write(pddl_str)

    return pddl_str


if __name__ == "__main__":
    print("generate blocksworld problem")
    random.seed(42)
    for num in [3, 4, 5, 6, 7]:
        if num == 3:
            for i in range(1, 6):
                s = blocksworld_pddl_problem(num=num, stack=2, distractor=0, seed=None,
                                            problem_name=f"blocksworld_pr{num}_{i}")
        if num == 4:
            for i in range(1, 5):
                s = blocksworld_pddl_problem(num=num, stack=2, distractor=0, seed=None,
                                             problem_name=f"blocksworld_pr{num}_{i}")
            for i in range(5, 6):
                s = blocksworld_pddl_problem(num=num, stack=3, distractor=0, seed=None,
                                             problem_name=f"blocksworld_pr{num}_{i}")
        if num == 5:
            for i in range(1, 5):
                s = blocksworld_pddl_problem(num=num, stack=2, distractor=0, seed=None,
                                             problem_name=f"blocksworld_pr{num}_{i}")
            for i in range(5, 6):
                s = blocksworld_pddl_problem(num=num, stack=3, distractor=0, seed=None,
                                             problem_name=f"blocksworld_pr{num}_{i}")
        if num == 6:
            for i in range(1, 4):
                s = blocksworld_pddl_problem(num=num, stack=2, distractor=0, seed=None,
                                             problem_name=f"blocksworld_pr{num}_{i}")
            for i in range(4, 6):
                s = blocksworld_pddl_problem(num=num, stack=3, distractor=0, seed=None,
                                             problem_name=f"blocksworld_pr{num}_{i}")
        if num == 7:
            for i in range(1, 3):
                s = blocksworld_pddl_problem(num=num, stack=2, distractor=0, seed=None,
                                             problem_name=f"blocksworld_pr{num}_{i}")
            for i in range(3, 6):
                s = blocksworld_pddl_problem(num=num, stack=3, distractor=0, seed=None,
                                             problem_name=f"blocksworld_pr{num}_{i}")
