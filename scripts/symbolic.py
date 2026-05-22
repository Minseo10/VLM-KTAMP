import os
import subprocess
import time
import logging
from dataclasses import dataclass, field
from typing import Any, List
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.model import *
import re
from typing import Iterable, Tuple
from collections import defaultdict
from pathlib import Path as pathlibPath

logger = logging.getLogger("TAMP")


OBJ = "o_"


@dataclass
class Subgoal:
    """Represents a subgoal with type (Information/TAMP) and state."""
    goal_type: str  # "Information" or "TAMP"
    state: Any     # unified_planning State
    actions: list = field(default_factory=list)  # List of actions leading to this subgoal
    
    def __repr__(self):
        return f"Subgoal({self.goal_type})"


@dataclass
class Expr:
    def __eq__(self, a2):
        return hash(self) == hash(a2)

    def __lt__(self, a2):
        return hash(self) < hash(a2)

    def __gt__(self, a2):
        return hash(self) > hash(a2)


@dataclass
class Action:
    name: str = "default-action"
    args: List[Any] = field(default_factory=lambda: [])

    def __hash__(self):
        return hash(tuple([self.name] + list(self.args)))

    def __eq__(self, b):
        return hash(self) == hash(b)

    def __str__(self):
        return "{}({})".format(self.name, ", ".join(self.args))


def validate(domain_f, problem_f, plan_f):
    """Run pddl plan validator VAL"""
    # This command only works in Linux
    # For windows, you need exe path of the validator
    cmd = f'"Validate" -v "{domain_f}" "{problem_f}" ' f'"{plan_f}"'
    output = subprocess.getoutput(cmd)

    if "Plan valid" in output:
        return True, output
    else:
        return False, output


def fd_planner(plan_f, sas_f, domain_f, problem_f, planner, path):
    """Run Fast Downward planner."""
    cmd = f'"python" "{path}" --alias "{planner}" --search-time-limit 100 --plan-file "{plan_f}" --sas-file "{sas_f}" "{domain_f}" "{problem_f}"'
    start_time = time.time()
    output = subprocess.getoutput(cmd)
    end_time = time.time()
    planning_time = end_time - start_time

    plan = ""
    success = False

    if "Solution found." in output:
        success = True
        # Add planning time as a comment to the end of the plan file
        if os.path.exists(plan_f):
            with open(plan_f, "a") as f:
                f.write(f"\n; Planning time: {planning_time} seconds")
            with open(plan_f, "r") as f:
                plan = f.read()

    if os.path.exists(plan_f):
        # Add planning time as a comment to the end of the plan file
        with open(plan_f, "a") as f:
            f.write(f"\n; Planning time: {planning_time} seconds")
        with open(plan_f, "r") as f:
            plan = f.read()
    # print("Plan: \n", plan)
    # print("output: \n", output)

    return success, plan, output, planning_time


# ('pred', ( (args...), (args...), ... ))
GroupedFacts = Tuple[Tuple[str, Tuple[Tuple[str, ...], ...]], ...]
ATOM_RE = re.compile(r'^\s*Atom\s+([^(]+)\(([^)]*)\)\s*$')
SIGNED_ATOM_RE = re.compile(r'^\s*(Atom|NegatedAtom)\s+([^(]+)\(([^)]*)\)\s*$')


def to_positive_facts(state: Iterable[str]) -> GroupedFacts:
    """
    state: ('Atom p(a,b)', 'NegatedAtom q(x)', 'Atom r()', '', ...)
    Return: (('p', (('a','b'), ...)), ('r', ((),)), ...)
    """
    grouped = {}

    for s in state:
        if not s:
            continue
        m = ATOM_RE.match(s)
        if not m:
            continue

        pred, args_str = m.groups()
        pred = pred.strip()

        if args_str.strip() == "":
            args_t = ()
        else:
            args_t = tuple(a.strip() for a in args_str.split(",") if a.strip())

        grouped.setdefault(pred, set()).add(args_t)

    out: list = []
    for pred in sorted(grouped.keys()):
        args_list = sorted(grouped[pred])
        out.append((pred, tuple(args_list)))

    return tuple(out)


def parse_signed_facts(state: Iterable[str]) -> Tuple[GroupedFacts, GroupedFacts]:
    """Parse DOT state lines into positive and negative grouped facts."""
    positive = {}
    negative = {}

    for s in state:
        if not s:
            continue
        m = SIGNED_ATOM_RE.match(s)
        if not m:
            continue

        sign, pred, args_str = m.groups()
        pred = pred.strip()
        if args_str.strip() == "":
            args_t = ()
        else:
            args_t = tuple(a.strip() for a in args_str.split(",") if a.strip())

        target = positive if sign == "Atom" else negative
        target.setdefault(pred, set()).add(args_t)

    def _freeze(grouped):
        out = []
        for pred in sorted(grouped.keys()):
            out.append((pred, tuple(sorted(grouped[pred]))))
        return tuple(out)

    return _freeze(positive), _freeze(negative)


def apply_action(state, action_str, domprob):
    """Applies an action to the current state and returns the updated state."""
    parts = action_str.replace('(', '').replace(')', '').split()
    action_name = parts[0]
    action_params = parts[1:]

    action = domprob.domain.operators[action_name]
    if not action:
        raise ValueError(f"Action {action_name} not found in domain")

    # Binding action parameters
    bindings = dict(zip(action.variable_list.keys(), action_params))

    def bind(var):
        return bindings.get(var, var)

    state_dict = defaultdict(set, {pred: set(tuples) for pred, tuples in state})

    # Check preconditions
    def check_preconditions():
        # General precondition check
        for precond in action.precondition_pos:
            pred = precond.predicate[0]
            args = precond.predicate[1:]
            if tuple(bind(a) for a in args) not in state_dict.get(pred, set()):
                return False, precond
        return True, None

    satisfied, unmet_precond = check_preconditions()
    if not satisfied:
        raise ValueError(f"Preconditions {unmet_precond} failed for action {action_name}")

    # Apply effects
    new_state = defaultdict(set, {k: set(v) for k, v in state_dict.items()})
    for effect in action.effect_pos:
        pred = effect.predicate[0]
        args = effect.predicate[1:]
        new_state[pred].add(tuple(bind(a) for a in args))
    for effect in action.effect_neg:
        pred = effect.predicate[0]
        args = effect.predicate[1:]
        new_state[pred].discard(tuple(bind(a) for a in args))

    keys_to_remove = [key for key, value in new_state.items() if not value]

    for key in keys_to_remove:
        del new_state[key]

    new_state = tuple(sorted((k, tuple(sorted(v))) for k, v in new_state.items()))
    return new_state


def count_goals_satisfied(state_tuple, goal_atoms):
    """state_tuple: your grouped-facts tuple
       goal_atoms: domprob.goals()"""
    st = {}
    for pred, args_list in state_tuple:
        st[pred] = set(args_list)
    ok = 0
    for at in goal_atoms:
        pred = at.predicate[0]
        args = tuple(at.predicate[1:])
        is_neg = ("negated" in type(at).__name__.lower())
        if is_neg:
            if pred not in st or args not in st[pred]:
                ok += 1
        else:
            if pred in st and args in st[pred]:
                ok += 1
    return ok


def compute_edge_weights_local(name_to_node, edges_by_src, is_goal, goal_atoms,
                               tau=1.0, alpha=0.7, w_min=0.1, w_max=1.0,
                               equal_penalty=0.6, away_penalty=0.0):
    
    import networkx as nx

    goal_nodes = [name for name, node in name_to_node.items()
                  if is_goal(node.discrete_belief)]

    G = nx.DiGraph()
    G.add_nodes_from(name_to_node.keys())
    for u, lst in edges_by_src.items():
        for v, _act in lst:
            G.add_edge(u, v)

    if goal_nodes:
        dist = nx.multi_source_dijkstra_path_length(G.reverse(copy=False), goal_nodes)
    else:
        dist = {}
    INF = float('inf')

    def level_from_score(s: float) -> float:
        """s in [0,1] -> weight in [w_min, w_max]"""
        return float(w_min + (w_max - w_min) * max(0.0, min(1.0, s)))

    weights = {}
    for u, children in edges_by_src.items():
        du = dist.get(u, INF)
        for v, _a in children:
            dv = dist.get(v, INF)

            if du == INF and dv == INF:
                s = away_penalty
            elif du == INF and dv < INF:
                s = 1.0
            elif du < INF and dv == INF:
                s = away_penalty
            else:
                if dv < du:
                    s = 1.0
                elif dv == du:
                    s = equal_penalty
                else:
                    s = away_penalty

            w = level_from_score(s)
            weights[(u, v)] = w

    return weights


def initial_state_tuple(domprob):
    d = defaultdict(set)
    for at in domprob.initialstate():
        pred = at.predicate[0]
        args = tuple(at.predicate[1:])
        d[pred].add(args)
    return tuple(sorted((k, tuple(sorted(v))) for k, v in d.items()))


def goal_atoms(domprob):
    return list(domprob.goals())


def state_satisfies_goal(grouped_state, goal_atoms):
    # grouped_state: (('pred', (('a',), ...)), ...)
    state_dict = {}
    for pred, args_group in grouped_state:
        s = set()
        if not args_group:  # 0-arity
            s.add(())
        else:
            for a in args_group:
                s.add(tuple(a))
        state_dict[pred] = s
    for ga in goal_atoms:
        pred = ga.predicate[0]
        args = tuple(ga.predicate[1:])
        is_neg = ("negated" in type(ga).__name__.lower())
        if is_neg:
            if args in state_dict.get(pred, set()):
                return False
        else:
            if args not in state_dict.get(pred, set()):
                return False
    return True


def generate_problem_pddl(domain_name: str, problem_name: str, original_problem: Any, initial_state: Any, goal_state: Any, subgoal_idx: int) -> str:
    # Extract all objects from problem template (without duplicates)
    objects_set = {}
    for obj in original_problem.all_objects:
        objects_set[obj.name] = obj
    objects = list(objects_set.keys())
    
    # Helper function to convert expressions to PDDL string format
    def convert_to_pddl_format(expr_str):
        """Convert expression to proper PDDL format.
        Examples:
          'on(blue, green)' -> '(on blue green)'
          'on(blue green)' -> '(on blue green)'
          'arm-empty' -> '(arm-empty)'
        """
        expr_str = str(expr_str).strip()
        
        # If already starts with (, assume it's formatted
        if expr_str.startswith('('):
            return expr_str
        
        # Handle predicate(arg1, arg2, ...) format
        if '(' in expr_str and ')' in expr_str:
            # Extract predicate name and arguments
            pred_name = expr_str[:expr_str.index('(')]
            args_part = expr_str[expr_str.index('(') + 1:expr_str.rindex(')')]
            
            # Split arguments by comma or space
            if ',' in args_part:
                args = [arg.strip() for arg in args_part.split(',')]
            else:
                args = [arg.strip() for arg in args_part.split() if arg.strip()]
            
            # Reconstruct as PDDL format: (pred arg1 arg2 ...)
            return f"({pred_name} {' '.join(args)})"
        else:
            # Zero-arity predicate
            return f"({expr_str})"
    
    # Extract initial facts from initial_state
    init_facts = []
    init_fact_set = set()
    for fluent, value in initial_state._values.items():
        if value.constant_value() is True:
            fluent_pddl = str(fluent)
            fact = convert_to_pddl_format(fluent_pddl)
            init_facts.append(fact)
            init_fact_set.add(fact)
    init_facts_str = " ".join(init_facts) + (" " if init_facts else "")

    # Extract goal facts from goal_state
    goal_facts = []
    for fluent, value in goal_state._values.items():
        if value.constant_value() is True:
            fluent_pddl = str(fluent)
            fact = convert_to_pddl_format(fluent_pddl)
            if fact not in init_fact_set:
                goal_facts.append(fact)
    goal_facts_str = " ".join(goal_facts) + (" " if goal_facts else "")
    
    pddl_lines = [
        f"(define (problem {domain_name}{problem_name}_subgoal{subgoal_idx})",
        f"  (:domain {domain_name})",
        "  (:objects",
        f"    {' '.join(objects)}",
        "  )",
        "  (:init",
    ]
    
    if init_facts_str:
        init_line = "  " + init_facts_str
        pddl_lines.append(init_line)

    pddl_lines.extend([
        "  )",
        "  (:goal",
        "    (and",
    ])

    if goal_facts_str:
        goal_line = "  " + goal_facts_str
        pddl_lines.append(goal_line)
    
    pddl_lines.extend([
        "    )",
        "  )",
        ")",
    ])
    
    pddl_content = "\n".join(pddl_lines)
    
    output_dir = pathlibPath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, "problem")))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{domain_name}{problem_name}_subgoal{subgoal_idx}.pddl"
    with open(output_path, 'w') as f:
        f.write(pddl_content)

    return str(output_path)


def grouped_facts_to_upstate(grouped_facts: Tuple, problem_template: Any, negated_facts: Tuple = ()) -> UPState:
    """Convert DOT-derived grouped facts to a unified_planning UPState.

    Start from the full problem initial state, then apply explicit Atom and
    NegatedAtom overrides from the DOT node label. This preserves facts omitted
    from the label, such as known-pose(hook), while still reflecting dynamic
    changes like NegatedAtom known-pose(red).
    """
    values_dict = dict(problem_template.initial_values)

    for pred_name, args_tuples in grouped_facts:
        fluent_obj = problem_template.fluent(pred_name)
        if fluent_obj is None:
            continue
        for args in args_tuples:
            if len(args) == 0:
                # Zero-arity: call with no arguments to create FNode
                fluent_expr = fluent_obj()
            else:
                # Multi-arity: convert string args to Object instances
                obj_args = [problem_template.object(arg) for arg in args]
                fluent_expr = fluent_obj(*obj_args)
            values_dict[fluent_expr] = TRUE()

    for pred_name, args_tuples in negated_facts:
        fluent_obj = problem_template.fluent(pred_name)
        if fluent_obj is None:
            continue
        for args in args_tuples:
            if len(args) == 0:
                fluent_expr = fluent_obj()
            else:
                obj_args = [problem_template.object(arg) for arg in args]
                fluent_expr = fluent_obj(*obj_args)
            values_dict[fluent_expr] = FALSE()
    
    return UPState(values_dict, problem_template)


def upstate_to_grouped_facts(state: UPState) -> Tuple:
    """Convert UPState to grouped facts representation."""
    grouped = defaultdict(set)
    for fluent, value in state._values.items():
        if value.constant_value() is True:
            pred_name = fluent.fluent_name() if hasattr(fluent, 'fluent_name') else str(fluent).split('(')[0]
            args = fluent.args if hasattr(fluent, 'args') else ()
            args_tuple = tuple(str(arg) for arg in args)
            grouped[pred_name].add(args_tuple)
    
    result = []
    for pred in sorted(grouped.keys()):
        result.append((pred, tuple(sorted(grouped[pred]))))
    return tuple(result)


def apply_action_unified(state: UPState, action_str: str, problem, simulator) -> UPState:
    """Apply action to state using SequentialSimulator."""
    parts = action_str.replace('(', '').replace(')', '').split()
    action_name = parts[0]
    action_params = parts[1:] if len(parts) > 1 else []
    
    action = problem.action(action_name)
    if action is None:
        raise ValueError(f"Action {action_name} not found in problem")
    
    param_objects = []
    for param, value in zip(action.parameters, action_params):
        param_objects.append(problem.object(value))
    
    instantiated_action = action(*param_objects)
    new_state = simulator.apply(state, instantiated_action)
    # print(f"Applied action: {instantiated_action}, new state: {new_state}")
    return new_state


def _eval_goal_expr_up(state: UPState, expr) -> bool:
    """Evaluate a unified_planning boolean goal expression on a state."""
    if hasattr(expr, "is_and") and expr.is_and():
        return all(_eval_goal_expr_up(state, arg) for arg in expr.args)
    if hasattr(expr, "is_or") and expr.is_or():
        return any(_eval_goal_expr_up(state, arg) for arg in expr.args)
    if hasattr(expr, "is_not") and expr.is_not():
        return not _eval_goal_expr_up(state, expr.args[0])

    value = state.get_value(expr)
    return value.constant_value() is True


def is_goal_state_up(state: UPState, problem) -> bool:
    """Check if state satisfies all goals using unified_planning."""
    for goal in problem.goals:
        try:
            if not _eval_goal_expr_up(state, goal):
                return False
        except:
            return False
    return True


def initial_state_tuple_up(problem_template) -> Tuple:
    """Get initial state as grouped facts from problem template."""
    initial_state = problem_template.initial_values
    grouped = defaultdict(set)
    for fluent, value in initial_state.items():
        if value:
            fluent_str = str(fluent)
            # Check if fluent has arguments (contains parentheses)
            if '(' in fluent_str and ')' in fluent_str:
                pred_name = fluent_str.split('(')[0]
                args_str = fluent_str.split('(')[1].rstrip(')')
                args = tuple(arg.strip() for arg in args_str.split(',')) if args_str else ()
            else:
                # Zero-arity predicate (no arguments)
                pred_name = fluent_str
                args = ()
            grouped[pred_name].add(args)
    
    result = []
    for pred in sorted(grouped.keys()):
        result.append((pred, tuple(sorted(grouped[pred]))))
    return tuple(result)


def grouped_facts_to_str(state) -> str:
    lines = []
    for pred, args_list in state:
        for args in args_list:
            if not args:
                lines.append(f"({pred})")
            else:
                lines.append(f"({pred} {' '.join(map(str, args))})")
    return "\n".join(lines)


def is_goal_state(state, goal_state_set):
    """
    state: set[tuple], e.g.,
      A) flat facts: {('clear','r4'), ('arm-empty',), ('on','red','green'), ...}
      B) grouped facts: (('arm-empty',((),)), ('clear', (('grey',),)), ('on', (('grey','red'), ...)), ...)
    goal_state_set: iterable of pddlpy Atom / NegatedAtom objects (with .predicate tuple)
    """
    from collections import defaultdict

    state_dict = defaultdict(set)

    is_grouped = (
            isinstance(state, tuple) and len(state) > 0 and
            isinstance(state[0], tuple) and len(state[0]) == 2 and
            isinstance(state[0][0], str) and
            isinstance(state[0][1], tuple)
    )

    if is_grouped:
        for pred, args_group in state:
            if not args_group:
                state_dict[pred].add(())
            else:
                for a in args_group:
                    state_dict[pred].add(tuple(a))
    else:
        for elem in state:
            if not isinstance(elem, tuple) or len(elem) == 0:
                continue
            pred = elem[0]
            args = tuple(elem[1:])
            state_dict[pred].add(args)

    # Check goals
    for atom in goal_state_set:
        if not hasattr(atom, "predicate"):
            return False

        pred = atom.predicate[0]
        args = tuple(atom.predicate[1:])
        cls = type(atom).__name__.lower()

        if "negated" in cls:
            if args in state_dict.get(pred, set()):
                return False
        else:
            if args not in state_dict.get(pred, set()):
                return False

    return True
