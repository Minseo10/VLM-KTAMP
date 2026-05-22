from graphviz import Digraph
import networkx as nx
from dataclasses import dataclass, field
from typing import Any, List, Optional
import uuid
import json
from pathlib import Path
from utils.utils import *
from belief_structs import DiscreteBelief
from symbolic import is_goal_state_up


@dataclass
class HybridNode:
    def __init__(self, discrete_belief: DiscreteBelief, sim_state: Any, name: str = ""):
        self.discrete_belief = discrete_belief
        self.sim_state = sim_state
        self.parent: Optional["HybridNode"] = None
        self.action: Optional[Any] = None
        self.children: List["HybridNode"] = []
        self.value: float = 0.0
        self.visits: int = 0
        self.depth: int = 0
        self.name: str = name
        self.image_path: List[Any] = None
        self.detached_object: Any = None
        self._id: uuid.UUID = uuid.uuid4()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"Node(name={self.name}, visits={self.visits}, value={self.value:.4f},\n"
            f"     discrete_belief={self.discrete_belief})"
        )
    

class HybridTree:
    def __init__(self, root: HybridNode):
        self.G = nx.MultiDiGraph()
        self.root = root
        self.current_node = root
        self.G.add_node(
            root,
            visits=root.visits,
            value=root.value,
            discrete_belief=root.discrete_belief,
            sim_state=root.sim_state,
            name=root.name,
            depth=root.depth,
        )
        self.filtered_nodes = []
        self.G.nodes[root]['visited'] = True

    def add_node(self,
                 parent_node: HybridNode,
                 child_node: HybridNode,
                 action: str,
                 weight: float,
                 name: str,
                 threshold: float = 0.0):
        """Add a child node to the tree."""
        if weight < threshold:
            self.filtered_nodes.append((parent_node, child_node, weight))
            return None
        
        # Set child metadata
        child_node.parent = parent_node
        child_node.action = action
        child_node.depth = parent_node.depth + 1
        child_node.name = name

        # Add child node
        self.G.add_node(
            child_node,
            visits=child_node.visits,
            value=child_node.value,
            discrete_belief=child_node.discrete_belief,
            sim_state=child_node.sim_state,
            name=name,
            depth=child_node.depth
        )

        # Add edge with action as key
        self.G.add_edge(parent_node, child_node, key=action, weight=weight)

        # Maintain local children list
        if not isinstance(parent_node.children, list):
            parent_node.children = []
        parent_node.children.append(child_node)

        return child_node
    
    def get_child(
        self,
        parent_node: HybridNode,
        action: Any,
    ) -> Optional[HybridNode]:
        """
        Retrieve a child node given (action, observation).
        """
        edge_key = action

        for _, v, k in self.G.out_edges(parent_node, keys=True):
            if k == edge_key:
                return v
        return None

    def path_to_root(self, node: HybridNode) -> List[HybridNode]:
        """
        Return path from root to the given node.
        """
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        return list(reversed(path))

    def visualize_tree(self, image_f):
        """
        Visualize the belief tree with action edges distinguished.
        
        Parameters
        ----------
        image_f : str
            Path to save the visualization (without extension)
        """
        dot = Digraph()
        dot.attr(rankdir='TB')

        # Add all nodes to the graph
        for node in self.G.nodes:
            visits = self.G.nodes[node]['visits']
            value = self.G.nodes[node]['value']
            depth = self.G.nodes[node]['depth']

            # Create label with node info
            label = f"d={depth}\\nV:{visits}\\nR:{value:.3f}"
            
            # Use circle shape for all nodes
            dot.node(str(id(node)), label=label, shape='circle', color='black', style='filled', fillcolor='lightblue')

        # Add all edges to the graph
        for u, v, key, data in self.G.edges(data=True, keys=True):
            weight = data['weight']
            action = str(key).replace('"', '\\"')  # Escape quotes
            edge_label = f"{action}\\nWeight: {weight:.4f}"
            dot.edge(str(id(u)), str(id(v)), label=edge_label, color='black')

        # Add filtered nodes and edges to the graph with gray color
        for parent_node, child_node, weight in self.filtered_nodes:
            dot.node(str(id(child_node)), label="filtered...", color='gray')
            dot.edge(str(id(parent_node)), str(id(child_node)), label=f"{weight:.4f}", color='gray')

        # dot.attr(label=f"Planning time: {planning_time:.4f} seconds", fontsize='12', loc='bottom')

        dot.render(image_f, format='png', cleanup=True)

    def export_tree_to_json(self, tree, out_path, problem_template, include_continuous_params=False):

        def _to_jsonable(x):
            if x is None:
                return None
            if isinstance(x, (str, int, float, bool)):
                return x
            if isinstance(x, (list, tuple, set)):
                return [_to_jsonable(v) for v in x]
            if isinstance(x, dict):
                return {str(k): _to_jsonable(v) for k, v in x.items()}
            if hasattr(x, "tolist"):
                try:
                    return x.tolist()
                except Exception:
                    pass
            try:
                json.dumps(x)
                return x
            except TypeError:
                return str(x)

        nodes = []
        for n in tree.G.nodes():
            n_attrs = tree.G.nodes[n]
            node_obj = {
                "name": n.name,
                "visited": bool(n_attrs.get("visited", False)),
                "is_goal": is_goal_state_up(n.discrete_belief.state, problem_template),
                "discrete_state": str(n.discrete_belief.state),
            }
            nodes.append(node_obj)

        edges = []
        for u, v, k, d in tree.G.edges(keys=True, data=True):
            edge_obj = {
                "src": u.name,
                "dst": v.name,
                "action": str(k),
                "feasible": d.get("feasible", None),
            }
            if include_continuous_params:
                edge_obj["continuous_params"] = _to_jsonable(d.get("continuous_params"))
            edges.append(edge_obj)

        obj = {
            "root": tree.root.name,
            "current": tree.current_node.name,
            "nodes": nodes,
            "edges": edges,
        }

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

        return obj
