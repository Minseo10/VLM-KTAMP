import sys
import os
import logging

logger = logging.getLogger("TAMP")

import utils.utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import prompts, llm_functions as llm, pr2_api as pr2, kuka_api as kuka, franka_api as franka
from utils.utils import *
from collections import defaultdict, deque
from typing import Any, List, Tuple
import hybrid_tree as ht
import json
from forbiditerative import planners
from pathlib import Path as pathlibPath
import pydot
import time
import scripts.action as action
import re
import random
from openai import OpenAI
import genesis as gs
import networkx as nx
from symbolic import *
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *
from unified_planning.model import State as UPState


config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
api_key = config["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)


def diverse_planning(domain_name, method_name, problem_path, plan_number, planner="topk"):
    domain_file = pathlibPath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domains", f"domain_{domain_name}.pddl")))
    problem_file = pathlibPath(problem_path)
    gv_path = pathlibPath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method_name, "plans", f"{pathlibPath(problem_path).stem}.gv")))

    # FI-diverse-agile
    if planner == "diverse_agl":
        plans = planners.plan_diverse_agl(domain_file=domain_file, problem_file=problem_file, number_of_plans_bound=plan_number)
    elif planner == "topk":
        plans = planners.plan_topk(domain_file=domain_file, problem_file=problem_file, number_of_plans_bound=plan_number)

    # create a dot graph out of plans
    import graphviz
    dot_txt = planners.get_dot(domain_file=domain_file, problem_file=problem_file,
                               plans=[plan["actions"] for plan in plans["plans"]])
    gv_path.parent.mkdir(parents=True, exist_ok=True)
    gv_path.write_text(dot_txt, encoding="utf-8")

    src = graphviz.Source(dot_txt)
    src.render(gv_path, view=False)

    return plans['plans'], gv_path  # [{'cost': 24, 'actions': ['unstack b1 b2', ...]}, {'cost': 24, 'actions': ..}, ...]


def build_tree_from_dot(method, dot_path: str, domain, prob_size, prob_idx, problem_template) -> ht.HybridTree:
    """Build hybrid tree from DOT file using unified_planning."""
    graphs = pydot.graph_from_dot_file(dot_path)
    dot = graphs[0]

    def _parse_label(label: str):
        label = (label or "").strip('"').replace('\\n', '\n').replace('\\l', '\n')
        lines = [ln for ln in label.split('\n') if ln.strip()]
        atoms = tuple(lines[1:]) if len(lines) >= 2 else tuple()
        return parse_signed_facts(atoms)

    def _expand_grouped_facts(grouped_facts):
        atoms = set()
        for pred, args_list in grouped_facts:
            for args in args_list:
                atoms.add((pred, args))
        return atoms

    from belief_structs import DiscreteBelief
    
    simulator = SequentialSimulator(problem_template)
    initial_up_state = simulator.get_initial_state()
    initial_positive_atoms = _expand_grouped_facts(upstate_to_grouped_facts(initial_up_state))

    name_to_node = {}
    node_positive_atoms = {}
    node_negative_atoms = {}
    label_atom_universe = set()
    for n in dot.get_nodes():
        name = n.get_name().strip('"')
        if not name or name in ('node', 'graph', 'edge'):
            continue
        label = n.get_attributes().get('label', '')
        disc_tuple, neg_disc_tuple = _parse_label(label)
        node_positive_atoms[name] = _expand_grouped_facts(disc_tuple)
        node_negative_atoms[name] = _expand_grouped_facts(neg_disc_tuple)
        label_atom_universe.update(node_positive_atoms[name])
        label_atom_universe.update(node_negative_atoms[name])
        # print(f"parsed node '{name}' with positive facts: {disc_tuple}, negative facts: {neg_disc_tuple}")
        node = ht.HybridNode(
            discrete_belief=DiscreteBelief(state=None),
            sim_state=None,
            name=name
        )
        name_to_node[name] = node

    edges_by_src = defaultdict(list)
    indeg = defaultdict(int)
    for e in dot.get_edges():
        src = e.get_source().strip('"')
        dst = e.get_destination().strip('"')
        action = e.get_attributes().get('label', '').strip('"')

        if src not in name_to_node:
            node = ht.HybridNode(
                discrete_belief=DiscreteBelief(state=None),
                sim_state=None,
                name=src
            )
            name_to_node[src] = node
        if dst not in name_to_node:
            node = ht.HybridNode(
                discrete_belief=DiscreteBelief(state=None),
                sim_state=None,
                name=dst
            )
            name_to_node[dst] = node

        edges_by_src[src].append((dst, action))
        indeg[dst] += 1
        indeg.setdefault(src, 0)

    
    # Determine root node
    # 1) Find node(s) whose positive atoms match the initial state
    projected_initial_positive = initial_positive_atoms & label_atom_universe

    exact_root_matches = [
        name for name in name_to_node
        if node_positive_atoms.get(name, set()) == projected_initial_positive
    ]

    root_name = exact_root_matches[0] if exact_root_matches else None

    # 2) Indegree = 0
    if root_name is None:
        root_name = next((n for n, d in indeg.items() if d == 0), None)
    
    if root_name is None:
        raise RuntimeError("Could not determine root node from .gv")

    name_to_node[root_name].discrete_belief = DiscreteBelief(state=initial_up_state)

    # Reconstruct every node state by applying actions from the root.
    state_q = deque([root_name])
    while state_q:
        src_name = state_q.popleft()
        src_state = name_to_node[src_name].discrete_belief.state
        if src_state is None:
            continue
        for dst_name, action in edges_by_src.get(src_name, []):
            next_state = apply_action_unified(src_state, action, problem_template, simulator)
            existing_state = name_to_node[dst_name].discrete_belief.state
            if existing_state is None:
                name_to_node[dst_name].discrete_belief = DiscreteBelief(state=next_state)
                # print(f"derived node '{dst_name}' UPState from {src_name} via '{action}': {next_state}")
                state_q.append(dst_name)
            elif existing_state != next_state:
                raise RuntimeError(
                    f"Inconsistent reconstructed state for node '{dst_name}' from action '{action}'."
                )

    edge_w = compute_edge_weights_local(
        name_to_node,
        edges_by_src,
        is_goal=lambda db: is_goal_state_up(db.state, problem_template),
        goal_atoms=problem_template.goals,
        tau=0.6,
        alpha=1.0,
        w_min=0.1,
        w_max=1.0
    )

    tree = ht.HybridTree(name_to_node[root_name])

    q = deque([root_name])
    in_tree = set([root_name])

    while q:
        u_name = q.popleft()
        parent = name_to_node[u_name]

        for v_name, action in edges_by_src.get(u_name, []):
            if v_name == u_name:
                continue

            child = name_to_node[v_name]
            weight = edge_w.get((u_name, v_name), 0.1)

            if child in tree.G:
                if nx.has_path(tree.G, child, parent):
                    continue
                tree.add_node(parent, child, action, weight=weight, name=v_name)
            else:
                tree.add_node(parent, child, action, weight=weight, name=v_name)
                in_tree.add(v_name)
                q.append(v_name)

    tree.visualize_tree(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain, method, "tree", f"discrete{prob_size}_{prob_idx}.png")))
    return tree


def hybrid_tree_expansion(
    method: str,
    json_path: str = None,
    problem_pddl_path: str = None,
    prob_num: int = None,
    prob_idx: int = None,
    trial: int = None,
    repeat: int = None,
    model: str = None,
    domain_name: str = None,
    robot_name: str = None,
    grasp_type: str = None,
    discrete_tree: ht.HybridTree = None,
    problem_template: Any = None,
    sampler: Any = None,
    child_selector: Any = None,
    backtrack_selector: Any = None,
    K: int = 5,
    weight_threshold: float = 0.2,
    num_distractor: int = 0,
) -> Tuple[ht.HybridTree, bool, float, int]:
    """
    From root of discrete_tree, sample K continuous parameters per symbolic edge under VLM guidance,
    generate candidate nodes, select via VLM, and expand hybrid tree by descending to selected child.

    Args:
        discrete_tree: discrete tree built by build_tree_from_dot
        problem_template: unified_planning PDDL problem
        sampler: function mapping (sim_state, action) to continuous parameters
        child_selector: function to select one node from candidate nodes
        K: number of continuous parameters to sample per action
    """
    logger = logging.getLogger(__name__)
    tamp_success = False
    backtrack_count = 0

    simulator = SequentialSimulator(problem_template)

    # Initialize simulator and get initial state
    if robot_name == 'pr2':
            sim_wrapper, root_image_paths = pr2.start_sim(json_path, method, prob_num, prob_idx, trial, repeat)
    elif robot_name == 'kuka':
            sim_wrapper, root_image_paths = kuka.start_sim(json_path, method, prob_num, prob_idx, trial, repeat, num_distractor=num_distractor)
    init_scene = sim_wrapper.scene.sim.get_state()

    start = time.time()
    root = ht.HybridNode(
        discrete_belief=discrete_tree.root.discrete_belief,
        sim_state=init_scene
    )
    root.name = discrete_tree.root.name
    root.image_path = root_image_paths
    hybrid = ht.HybridTree(root)
    hybrid.current_node = root
    hybrid.G.nodes[root]['visited'] = True
    discrete_tree.G.nodes[discrete_tree.root]['visited'] = True

    while True:
        # print(f"current node: {hybrid.current_node.name}")
        goal_ok = is_goal_state_up(hybrid.current_node.discrete_belief.state, problem_template)
        physical_ok, _ = check_state(hybrid.current_node, sim_wrapper, robot_name)
        if goal_ok and physical_ok:
            tamp_success = True
            break

        # Find matching discrete node in discrete plan tree
        matching = [
            n for n in discrete_tree.G.nodes()
            if n.discrete_belief.state == hybrid.current_node.discrete_belief.state
        ]
        if not matching:
            break
        discrete_node = matching[0]
        edges = list(discrete_tree.G.out_edges(discrete_node, keys=True, data=True))
        # print("edges from discrete tree:", [(u.name, v.name, k, d) for u, v, k, d in edges])
        if not edges:
            break

        # For motion replanning (up to 4 times)
        success = False
        attempt = 0

        while attempt < K and not success:
            attempt += 1
            candidates: List[Tuple[ht.HybridNode, bool, Any]] = []
            # accepted_children: List[Tuple[ht.HybridNode, float, str]] = []  # (child, weight, action)

            # Check all symbolic edges from current node in discrete_tree
            for src, dst, action, attrs in edges:
                weight = attrs.get('weight')
                if weight <= weight_threshold:
                    continue
                logger.info(f"\n[hybrid_tree_expansion] sampling attempt {attempt}/{K}, for action {action}")
                child_name = dst.name
                child_name = f"{child_name}_{attempt}"
                # Reset simulator to previous scene state
                sim_wrapper.scene.sim.reset(state=hybrid.current_node.sim_state)
                if domain_name=='kitchen':
                    if hybrid.current_node.detached_object is not None and "putdown" in action.split()[0]:
                        sim_wrapper.close_gripper(object=hybrid.current_node.detached_object)

                    # obj_name = action.split()[1]
                    # target_name = action.split()[2]
                    # obj = sim_wrapper.object_dict[obj_name]
                    #
                    # if target_name in sim_wrapper.attach_dict and target_name in sim_wrapper.object_dict and obj in sim_wrapper.attach_dict[target_name]:
                    #     target = sim_wrapper.object_dict[target_name]
                    #     sim_wrapper.attach_dict[target_name].remove(obj)
                    #     sim_wrapper.detach_constraint(obj, target)

                #Sample and simulate motion execution
                verb = (action or "").strip().split()[0].lower()

                if verb not in {"cook", "clean"}:
                    success, continuous_params = sampler(
                        method, prob_num, prob_idx, trial, repeat, child_name, domain_name, robot_name,
                        sim_wrapper, action, belief=None, left=True, grasp_type=grasp_type,
                    )
                else: # pass sampling for cook and clean
                    success = True
                    continuous_params = {
                        "ok": True,
                        "where": "path",
                        "action": action,
                        "act_type": "",
                        "pre_pose": None,
                        "approach": None,
                        "traj": None,
                        "image_path": hybrid.current_node.image_path,
                        "error": None,
                    }

                new_scene = sim_wrapper.scene.sim.get_state()
                new_state = apply_action_unified(
                    hybrid.current_node.discrete_belief.state, action, problem_template, simulator
                )
                from belief_structs import DiscreteBelief
                new_disc = DiscreteBelief(state=new_state)

                sim_wrapper.scene.clear_debug_objects()
                if domain_name=='kitchen':
                    # print(f"detach_constraint with object {action.split()[1]}")
                    detached_object = sim_wrapper.scene.entities[sim_wrapper.object_dict[action.split()[1]]]
                    sim_wrapper.open_gripper(object=detached_object)

                    # if "putdown" in action.split()[0]:
                    #     attached_object = sim_wrapper.object_dict[action.split()[2]]
                    #     if attached_object is not None:
                    #         sim_wrapper.attach_constraint(detached_object, attached_object)
                    #         sim_wrapper.attach_dict.setdefault(action.split()[2], []).append(detached_object)

                child = ht.HybridNode(
                    discrete_belief=new_disc,
                    sim_state=new_scene
                )
                child.name=child_name
                child.image_path=continuous_params["image_path"]
                if domain_name=='kitchen':
                    child.detached_object = detached_object

                hybrid.add_node(
                    parent_node=hybrid.current_node,
                    child_node=child,
                    action=action,
                    weight=1.0,
                    name=child_name
                )
                # logger.info(f"[hybrid_tree_expansion] child node {child_name} added")

                # Check if the PDDL is satisfied (consistent)
                satisfied, feedback = check_state(child, sim_wrapper, robot_name)
                logger.info(f"[hybrid_tree_expansion] feedback: {feedback}")
                if not satisfied:
                    success = False
                    continuous_params = append_error(continuous_params, feedback)

                hybrid.G.edges[(hybrid.current_node, child, action)]['continuous_params'] = continuous_params
                hybrid.G.edges[(hybrid.current_node, child, action)]['feasible'] = bool(success)

                candidates.append((child, success, action))

            # VLM selects most promising candidate
            ok_candidates = [(c, ok, action) for (c, ok, action) in candidates if ok]
            if ok_candidates:
                if len(ok_candidates) == 1:
                    selected = ok_candidates[0][0]
                    logger.info(f"[hybrid_tree_expansion] selected node: {selected.name}")
                    hybrid.current_node = selected
                    for candidate in ok_candidates:
                        hybrid.G.nodes[candidate[0]]['visited'] = True

                        # find matching discrete nodes
                        matching = [
                            n for n in discrete_tree.G.nodes()
                            if n.discrete_belief.state == candidate[0].discrete_belief.state
                        ]
                        discrete_tree.G.nodes[matching[0]]['visited'] = True
                    success = True
                else:
                    selected = child_selector(domain_name, problem_pddl_path, model, ok_candidates, hybrid.current_node)
                    if selected is not None:
                        logger.info(f"[hybrid_tree_expansion] selected node: {selected.name}")
                        hybrid.current_node = selected
                        for candidate in ok_candidates:
                            hybrid.G.nodes[candidate[0]]['visited'] = True

                            # find matching discrete nodes
                            matching = [
                                n for n in discrete_tree.G.nodes()
                                if n.discrete_belief.state == candidate[0].discrete_belief.state
                            ]
                            discrete_tree.G.nodes[matching[0]]['visited'] = True
                        success = True
                    else:
                        success = False
            else:
                if attempt < K:
                    logger.info(f"[hybrid_tree_expansion] all samples failed in attempt {attempt}, retrying...")
                else:
                    logger.info(f"[hybrid_tree_expansion] all samples failed in {K} attempts, will backtrack.")

        if not success:
            backtrack_count += 1
            back_node = backtrack_selector(domain_name, method, problem_pddl_path, prob_num, prob_idx, model, hybrid.current_node, action, hybrid, discrete_tree, problem_template, trial, repeat)
            if back_node is None:
                back_node = bfs_unvisited_selector(domain_name, method, problem_pddl_path, prob_num, prob_idx, model, hybrid.current_node, action, hybrid, discrete_tree, problem_template, trial, repeat)
            logger.info(f"[hybrid_tree_expansion] backtracking to node: {back_node.name}")
            hybrid.current_node = back_node

    end = time.time()
    time_elapsed = end - start
    sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}")), node_name="tamp_final")

    return hybrid, tamp_success, time_elapsed, backtrack_count


def check_state(current_node, sim_wrapper, robot_name):
    # print("checking state")
    # print("current node:", current_node.name, "\n current discrete belief:", current_node.discrete_belief)
    
    state = current_node.discrete_belief.get_grouped_facts()
    # print("check_state grouped facts:", state, "discrete belief:", current_node.discrete_belief)
    feedback = []
    is_satisfied = True

    # Handle grouped facts format: (('pred', (('arg1', 'arg2'), ...)), ...)
    for pred, args_list in state:
        if pred == 'on':
            for args in args_list:
                obj = args[0]
                underobj = args[1]
                if not check_on(sim_wrapper, obj, underobj):
                    is_satisfied = False
                    feedback.append(f"{obj} is not on {underobj}")
        elif pred == 'on-table':
            for args in args_list:
                obj = args[0]
                if not check_on_table(robot_name, sim_wrapper, obj):
                    is_satisfied = False
                    feedback.append(f"{obj} is not on table")
        elif pred == 'holding':
            for args in args_list:
                if not args or len(args) == 0:
                    continue
                obj = args[0]
                if not check_holding(sim_wrapper, obj):
                    is_satisfied = False
                    feedback.append(f"robot is not holding {obj}")

    return is_satisfied, ", ".join(feedback)


def vlm_selector(domain, problem_pddl_path, model, candidates, current_node) -> ht.HybridNode:
    num_candidates = len(candidates)
    with open(problem_pddl_path) as f:
        problem = f.read()

    domain_pddl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domains", f"domain_{domain}.pddl"))
    with open(domain_pddl_path) as f:
        domain_pddl = f.read()

    system_prompt = prompts.vlm_action_selector(domain, num_candidates, problem, domain_pddl)
    system = [{
        "type": "text",
        "text": system_prompt
    }]

    # current node
    grouped_state = current_node.discrete_belief.get_grouped_facts()
    current_pddl_state = grouped_facts_to_str(grouped_state)
    current_image_paths = current_node.image_path
    current_images = llm.encode_image(current_image_paths)
    current_node_name = current_node.name
    prompt1 = f"For current node {current_node_name}, the symbolic PDDL state is: {current_pddl_state}"
    message_content = [{"type": "text", "text": prompt1}]
    for current_image in current_images:
        message_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_image}"}},
        )

    # candidate nodes
    for i, candidate in enumerate(candidates):
        image_paths = candidate[0].image_path
        images = llm.encode_image(image_paths)
        grouped_state = candidate[0].discrete_belief.get_grouped_facts()
        state = grouped_facts_to_str(grouped_state)
        action = candidate[2]
        node_name = candidate[0].name
        prompt = f"For {node_name} (result of {action}), the symbolic PDDL state is: {state}"
        message_content.append({"type": "text","text": prompt})

        for image in images:
            message_content.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image}"}})

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": message_content}],
        max_completion_tokens=2048,
        temperature=0.0,
    )
    answer = response.choices[0]
    print("reponse: ", response.choices[0])
    answer = answer.message.content
    first_line = answer.strip().splitlines()[0].strip().lower()  # name of selected node

    selected = None
    for candidate in candidates:
        if candidate[0].name == first_line:
            selected = candidate[0]
            break
    return selected


def random_selector(domain, problem_pddl_path, model, candidates, current_node) -> ht.HybridNode:
    pick = random.choice(candidates)
    if isinstance(pick, (tuple, list)):
        return pick[0]
    return pick


def bfs_unvisited_selector(domain, method, problem_pddl_path, prob_num, prob_idx, model, current_node, action, hybrid, discrete, problem_template, trial, repeat) -> ht.HybridNode:

    dq = deque([discrete.root])
    seen = set()

    while dq:
        dn = dq.popleft()
        if dn in seen:
            continue
        seen.add(dn)

        if not discrete.G.nodes[dn].get('visited', False):
            matches = [hn for hn in hybrid.G.nodes()
                       if getattr(hn, "discrete_belief", None) and 
                       hn.discrete_belief.state == dn.discrete_belief.state]
            if matches:
                return random.choice(matches)

        for _, child, _ in discrete.G.out_edges(dn, keys=True):
            dq.append(child)

    return hybrid.root


def vlm_backtrack_selector(domain, method, problem_pddl_path, prob_num, prob_idx, model, current_node, action, hybrid, discrete, problem_template, trial, repeat):
    print("backtracking")
    with open(problem_pddl_path) as f:
        problem = f.read()

    domain_pddl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domains", f"domain_{domain}.pddl"))
    with open(domain_pddl_path) as f:
        domain_pddl = f.read()

    system_prompt = prompts.vlm_backtrack_prompt(domain, problem, domain_pddl)
    system = [{
        "type": "text",
        "text": system_prompt
    }]

    # Use json instead
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain, method, "batch_outputs", f"{prob_num}_{prob_idx}", f"trial{trial}_repeat{repeat}", "backtrack.json"))
    graph_obj = hybrid.export_tree_to_json(hybrid, json_path, problem_template, include_continuous_params=False)

    try:
        json_str = minify_json(graph_obj) if isinstance(graph_obj, (dict, list)) else minify_json(json_path)
    except Exception:
        with open(json_path, "r", encoding="utf-8") as f:
            json_str = f.read()

    message_content = [
        {"type": "text", "text":
            "Here is the hybrid state graph in JSON (minified). "
            "Fields: root,current,nodes[],edges[]. "
            "Node: name/visited/is_goal/discrete_state; "
            "Edge: src/dst/action/feasible. "
            "Pick ONE visited node to backtrack to. First line: node name only."
         }
    ]
    for chunk in chunk_text(json_str, max_chars=12000):
        message_content.append({"type": "text", "text": chunk})

    # current node images
    current_image_paths = current_node.image_path
    current_images = llm.encode_image(current_image_paths)
    current_node_name = current_node.name
    prompt2 = f"For current node ({current_node_name}), simulator-rendered images are:"
    message_content.append({"type": "text", "text": prompt2})
    for current_image in current_images:
        message_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_image}"}},
        )

    # Constraint Violation Feedback
    payload = None
    where_hint = None

    for _, v, k, data in hybrid.G.out_edges(current_node, keys=True, data=True):
        if k == action and not data.get('feasible', True):
            cp = data.get('continuous_params')
            payload = cp.get('error')
            if payload:
                break

    if payload:
        feedback = payload
    else:
        feedback = "A motion path was found, but executing it in simulation produced an incorrect or undesired result."

    prompt3 = f"Constraint Violation Feedback: {feedback}"
    print("Constraint Violation Feedback: ", prompt3)
    message_content.append({"type": "text", "text": prompt3})

    # query vlm
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": message_content}],
        max_completion_tokens=2048,
        temperature=0.0,
    )
    answer = response.choices[0]
    print("reponse: ", response.choices[0])
    answer = answer.message.content
    first_line = answer.strip().splitlines()[0].strip().lower()  # name of selected node

    all_matches = [n for n in hybrid.G.nodes() if n.name == first_line]
    if all_matches:
        return random.choice(all_matches)
    

def extract_final_plan_with_params(hybrid: ht.HybridTree):
    root = hybrid.root
    goal = hybrid.current_node
    if root is None or goal is None:
        return []

    try:
        node_path = nx.shortest_path(hybrid.G, source=root, target=goal)
    except nx.NetworkXNoPath:
        return []

    plan = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edata = hybrid.G.get_edge_data(u, v)
        chosen_key, chosen_attrs = None, None
        for k, attrs in edata.items():
            if 'continuous_params' in attrs:
                chosen_key, chosen_attrs = k, attrs
                break
        if chosen_attrs is None:
            chosen_key, chosen_attrs = next(iter(edata.items()))

        plan.append({
            "action": chosen_key,
            "feasible": bool(chosen_attrs.get("feasible", True)),
            "continuous_params": chosen_attrs.get("continuous_params", {}),
            "from_state": str(u.discrete_belief.state),
            "to_state": str(v.discrete_belief.state),
            "result_images": getattr(v, "image_path", None),
        })
    return plan
