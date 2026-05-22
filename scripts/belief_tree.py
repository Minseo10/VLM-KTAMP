from __future__ import annotations

from graphviz import Digraph
import networkx as nx
from dataclasses import dataclass
from typing import Any, Tuple, Optional, List
import uuid
from belief_structs import ContinuousBelief, DiscreteBelief


@dataclass
class BeliefNode:
    def __init__(self, cb: ContinuousBelief, db: DiscreteBelief):
        self.continuous_belief = cb
        self.discrete_belief = db
        self.parent: Optional["BeliefNode"] = None
        self.action: Optional[Any] = None
        self.observation: Optional[Any] = None
        self.children: List["BeliefNode"] = []
        self.value: float = 0.0
        self.visits: int = 0
        self.depth: int = 0
        self.name: str = ""
        self._id: uuid.UUID = uuid.uuid4()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"Node(name={self.name}, visits={self.visits}, \n"
            f"     continuous_belief={self.continuous_belief}, discrete_belief={self.discrete_belief})"
        )
    

class BeliefTree:
    def __init__(self, root: BeliefNode):
        self.G = nx.MultiDiGraph()
        self.root = root
        self.current_node = root
        self.G.add_node(
            root,
            visits=root.visits,
            value=root.value,
            continuous_belief=root.continuous_belief,
            discrete_belief=root.discrete_belief,
            name=root.name,
            depth=root.depth,
        )
        self.G.nodes[root]["visited"] = True

    def add_node(
        self,
        parent_node: BeliefNode,
        child_node: BeliefNode,
        action: Any,
        observation: Any,
        *,
        overwrite: bool = False,
    ) -> BeliefNode:
        """
        Add a child node to the tree using (action, observation) as edge identity.

        If the same (parent, action, observation) already exists:
        - overwrite=False: return existing child
        - overwrite=True: replace existing edge and node

        Parameters
        ----------
        parent_node : BeliefNode
        child_node : BeliefNode
        action : Any
        observation : Any
        overwrite : bool

        Returns
        -------
        BeliefNode
        """
        edge_key = self._edge_key(action, observation)

        # Check if edge already exists
        for _, v, k in self.G.out_edges(parent_node, keys=True):
            if k == edge_key:
                if not overwrite:
                    return v
                else:
                    # Remove existing edge (and optionally node)
                    self.G.remove_edge(parent_node, v, key=k)
                    if self.G.in_degree(v) == 0:
                        self.G.remove_node(v)
                    break

        # Set child metadata
        child_node.parent = parent_node
        child_node.action = action
        child_node.observation = observation
        child_node.depth = parent_node.depth + 1

        # Add child node
        self.G.add_node(
            child_node,
            visits=child_node.visits,
            value=child_node.value,
            continuous_belief=child_node.continuous_belief,
            discrete_belief=child_node.discrete_belief,
            name=child_node.name,
            depth=child_node.depth,
        )

        # Add edge with (action, observation) as key
        self.G.add_edge(
            parent_node,
            child_node,
            key=edge_key,
            action=action,
            observation=observation,
        )

        # Maintain local children list
        if not isinstance(parent_node.children, list):
            parent_node.children = []
        parent_node.children.append(child_node)

        return child_node

    def get_child(
        self,
        parent_node: BeliefNode,
        action: Any,
        observation: Any,
    ) -> Optional[BeliefNode]:
        """
        Retrieve a child node given (action, observation).
        """
        edge_key = self._edge_key(action, observation)

        for _, v, k in self.G.out_edges(parent_node, keys=True):
            if k == edge_key:
                return v
        return None

    def get_children(
        self,
        parent_node: BeliefNode,
        action: Optional[Any] = None,
        observation: Optional[Any] = None,
    ) -> List[BeliefNode]:
        """
        Retrieve children of a node with optional filtering.
        """
        result = []
        action_key = str(action) if action is not None else None
        obs_key = repr(observation) if observation is not None else None

        for _, v, k in self.G.out_edges(parent_node, keys=True):
            a, o = k

            if action_key is not None and a != action_key:
                continue
            if obs_key is not None and o != obs_key:
                continue

            result.append(v)

        return result

    def has_child(
        self,
        parent_node: BeliefNode,
        action: Any,
        observation: Any,
    ) -> bool:
        """
        Check if a child exists for (action, observation).
        """
        return self.get_child(parent_node, action, observation) is not None

    @staticmethod
    def _edge_key(action: Any, observation: Any) -> Tuple[str, str]:
        """Build a hashable edge key for MultiDiGraph.

        Observation objects may contain lists/dicts and are not hashable.
        We canonicalize to string keys while keeping full objects in edge attrs.
        """
        return str(action), repr(observation)

    def path_to_root(self, node: BeliefNode) -> List[BeliefNode]:
        """
        Return path from root to the given node.
        """
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        return list(reversed(path))

    def set_current_node(self, node: BeliefNode) -> None:
        """
        Update the current node pointer.
        """
        if node not in self.G:
            raise ValueError("Node not in tree.")
        self.current_node = node

    def __len__(self) -> int:
        return self.G.number_of_nodes()

    def __repr__(self) -> str:
        return f"BeliefTree(num_nodes={self.G.number_of_nodes()}, root={self.root})"
    
    def visualize_tree(self, image_f: str):
        """
        Visualize the belief tree with action edges and observation edges distinguished.
        
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
            label = f"d={depth}\nV:{visits}\nR:{value:.3f}"
            
            # Use circle shape for all nodes
            dot.node(str(id(node)), label=label, shape='circle', color='black', style='filled', fillcolor='lightblue')
        
        # Add all edges to the graph, distinguishing action and observation
        action_colors = {
            'pickup': 'red',
            'unstack': 'yellow',
            'putdown': 'blue',
            'stack': 'green',
            'pull_towards': 'orange',
            'putdown_sink': 'cyan',
            'putdown_stove': 'cyan',
            'putdown_table': 'cyan',
        }
        
        for u, v, key, data in self.G.edges(data=True, keys=True):
            action, observation = key
            
            # Extract action type from full action string
            if isinstance(action, str):
                action_type = action.split()[0] if ' ' in action else action
            else:
                action_type = str(action)
            
            # Get color for action
            action_color = action_colors.get(action_type, 'gray')
            
            # Format edge labels
            action_label = str(action)[:20] + "..." if len(str(action)) > 20 else str(action)
            obs_label = str(observation)[:15] + "..." if len(str(observation)) > 15 else str(observation)
            
            # Add action edge (solid line, colored by action type)
            dot.edge(str(id(u)), str(id(v)), 
                    label=action_label, 
                    color=action_color, 
                    style='solid',
                    penwidth='2',
                    fontsize='10')
            
            # Add observation edge as a label or sublabel (if needed, can add additional info)
            # For now, we include observation info in the action edge label
            if observation is not None:
                dot.edge(str(id(v)), str(id(v)), 
                        label=f"obs:{obs_label}", 
                        color=action_color, 
                        style='dotted',
                        fontsize='8',
                        constraint='false')
        
        dot.render(image_f, format='png', cleanup=True)
