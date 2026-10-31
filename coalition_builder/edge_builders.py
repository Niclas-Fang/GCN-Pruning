"""
Edge Coalition Builders

Implements various strategies for grouping edges into prior coalitions.

Edge player ID = column index in edge_index (i.e., edge_index[:, player_id])
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import networkx as nx

from .base import CoalitionBuilder, CoalitionResult


class EdgeCoalitionBuilder(CoalitionBuilder):
    """Base class for edge coalition strategies."""
    
    def __init__(self, strategy: str):
        super().__init__(strategy)
    
    def _get_undirected_edge_mapping(
        self,
        edge_index: torch.Tensor
    ) -> Dict[Tuple[int, int], List[int]]:
        """
        Map undirected edges to their directed instances.
        
        For undirected graphs stored as bidirectional edges, this groups
        (u,v) and (v,u) together.
        
        Args:
            edge_index: [2, num_edges] edge index
        
        Returns:
            Dictionary mapping (min(u,v), max(u,v)) -> [edge_player_ids]
        """
        undirected_map = defaultdict(list)
        
        for edge_id in range(edge_index.size(1)):
            u = int(edge_index[0, edge_id])
            v = int(edge_index[1, edge_id])
            
            # Canonical form: (min, max)
            key = (min(u, v), max(u, v))
            undirected_map[key].append(edge_id)
        
        return dict(undirected_map)


class SourceNodeEdgeBuilder(EdgeCoalitionBuilder):
    """Group edges by source node."""
    
    def __init__(self):
        super().__init__('source_node')
    
    def build(
        self,
        edge_index: torch.Tensor,
        **kwargs
    ) -> CoalitionResult:
        """
        Group edges by source node.
        
        Args:
            edge_index: [2, num_edges]
        
        Returns:
            CoalitionResult with edges grouped by source
        """
        source_to_edges = defaultdict(list)
        
        for edge_id in range(edge_index.size(1)):
            source = int(edge_index[0, edge_id])
            source_to_edges[source].append(edge_id)
        
        # Convert to lists
        coalitions = [edges for edges in source_to_edges.values()]
        group_names = [f'source_{src}' for src in source_to_edges.keys()]
        source_nodes = list(source_to_edges.keys())
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            source_nodes=source_nodes
        )


class TargetNodeEdgeBuilder(EdgeCoalitionBuilder):
    """Group edges by target node."""
    
    def __init__(self):
        super().__init__('target_node')
    
    def build(
        self,
        edge_index: torch.Tensor,
        **kwargs
    ) -> CoalitionResult:
        """
        Group edges by target node.
        
        Args:
            edge_index: [2, num_edges]
        
        Returns:
            CoalitionResult with edges grouped by target
        """
        target_to_edges = defaultdict(list)
        
        for edge_id in range(edge_index.size(1)):
            target = int(edge_index[1, edge_id])
            target_to_edges[target].append(edge_id)
        
        coalitions = [edges for edges in target_to_edges.values()]
        group_names = [f'target_{tgt}' for tgt in target_to_edges.keys()]
        target_nodes = list(target_to_edges.keys())
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            target_nodes=target_nodes
        )


class CommunityEdgeBuilder(EdgeCoalitionBuilder):
    """
    Group edges by community structure.
    
    Strategy:
    1. Detect node communities (using Louvain or given)
    2. Group edges as:
       - intra_comm_k: edges within community k
       - inter_comm_i_j: edges between community i and j
    """
    
    def __init__(self):
        super().__init__('community')
    
    def build(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_communities: Optional[np.ndarray] = None,
        method: str = 'louvain',
        **kwargs
    ) -> CoalitionResult:
        """
        Group edges by community structure.
        
        Args:
            edge_index: [2, num_edges]
            num_nodes: Number of nodes
            node_communities: Optional pre-computed community labels [num_nodes]
            method: Community detection method ('louvain' or 'label_propagation')
        
        Returns:
            CoalitionResult with intra-community and inter-community groups
        """
        # Detect communities if not provided
        if node_communities is None:
            node_communities = self._detect_communities(
                edge_index, num_nodes, method
            )
        
        # Group edges by community pattern
        edge_groups = defaultdict(list)
        
        for edge_id in range(edge_index.size(1)):
            u = int(edge_index[0, edge_id])
            v = int(edge_index[1, edge_id])
            
            comm_u = int(node_communities[u])
            comm_v = int(node_communities[v])
            
            if comm_u == comm_v:
                # Intra-community edge
                key = f'intra_comm_{comm_u}'
            else:
                # Inter-community edge (canonical order)
                i, j = min(comm_u, comm_v), max(comm_u, comm_v)
                key = f'inter_comm_{i}_{j}'
            
            edge_groups[key].append(edge_id)
        
        # Convert to coalitions
        coalitions = list(edge_groups.values())
        group_names = list(edge_groups.keys())
        
        # Extract group types
        group_types = []
        comm_pairs = []
        for name in group_names:
            if name.startswith('intra'):
                group_types.append('intra')
                comm_id = int(name.split('_')[-1])
                comm_pairs.append((comm_id, comm_id))
            else:
                group_types.append('inter')
                parts = name.split('_')
                i, j = int(parts[2]), int(parts[3])
                comm_pairs.append((i, j))
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            group_types=group_types,
            community_pairs=comm_pairs,
            num_communities=len(np.unique(node_communities))
        )
    
    def _detect_communities(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        method: str = 'louvain'
    ) -> np.ndarray:
        """
        Detect node communities using NetworkX.
        
        Args:
            edge_index: [2, num_edges]
            num_nodes: Number of nodes
            method: 'louvain' or 'label_propagation'
        
        Returns:
            Community labels [num_nodes]
        """
        # Build undirected NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Add unique undirected edges
        edge_set = set()
        for i in range(edge_index.size(1)):
            u = int(edge_index[0, i])
            v = int(edge_index[1, i])
            edge = (min(u, v), max(u, v))
            edge_set.add(edge)
        
        G.add_edges_from(edge_set)
        
        # Detect communities
        if method == 'louvain':
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(G, seed=42)
        elif method == 'label_propagation':
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.label_propagation_communities(G)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to array
        community_labels = np.zeros(num_nodes, dtype=np.int32)
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                community_labels[node] = comm_id
        
        return community_labels


class MotifEdgeBuilder(EdgeCoalitionBuilder):
    """
    Group edges by local structure / motif participation.
    
    Strategies:
    - 'bridge': Bridge edges vs non-bridge
    - 'triangle': Triangle participation (0, 1, 2+ triangles)
    - 'local_clustering': By local clustering coefficient
    """
    
    def __init__(self, motif_type: str = 'bridge'):
        super().__init__(f'motif_{motif_type}')
        self.motif_type = motif_type
    
    def build(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        **kwargs
    ) -> CoalitionResult:
        """
        Group edges by motif participation.
        
        Args:
            edge_index: [2, num_edges]
            num_nodes: Number of nodes
        
        Returns:
            CoalitionResult with edges grouped by motif property
        """
        if self.motif_type == 'bridge':
            return self._build_bridge_groups(edge_index, num_nodes)
        elif self.motif_type == 'triangle':
            return self._build_triangle_groups(edge_index, num_nodes)
        elif self.motif_type == 'local_clustering':
            return self._build_clustering_groups(edge_index, num_nodes)
        else:
            raise ValueError(f"Unknown motif type: {self.motif_type}")
    
    def _build_bridge_groups(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> CoalitionResult:
        """Group edges as bridge vs non-bridge."""
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Map undirected edges to player IDs
        undirected_map = self._get_undirected_edge_mapping(edge_index)
        
        for (u, v) in undirected_map.keys():
            G.add_edge(u, v)
        
        # Find bridges
        bridges = set(nx.bridges(G))
        
        # Group edges
        bridge_edges = []
        non_bridge_edges = []
        
        for (u, v), edge_ids in undirected_map.items():
            if (u, v) in bridges or (v, u) in bridges:
                bridge_edges.extend(edge_ids)
            else:
                non_bridge_edges.extend(edge_ids)
        
        coalitions = [bridge_edges, non_bridge_edges]
        group_names = ['bridge_edges', 'non_bridge_edges']
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            bridge_counts=[len(bridge_edges), len(non_bridge_edges)]
        )
    
    def _build_triangle_groups(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> CoalitionResult:
        """Group edges by triangle participation count."""
        # Build adjacency for fast triangle counting
        adj = self._build_adjacency_set(edge_index, num_nodes)
        
        # Count triangles per undirected edge
        undirected_map = self._get_undirected_edge_mapping(edge_index)
        edge_triangle_counts = {}
        
        for (u, v) in undirected_map.keys():
            # Count common neighbors
            common = adj[u] & adj[v]
            edge_triangle_counts[(u, v)] = len(common)
        
        # Group edges by triangle count
        # Bins: 0, 1, 2-5, 6+
        bins = {
            '0_triangles': [],
            '1_triangle': [],
            '2_5_triangles': [],
            '6plus_triangles': []
        }
        
        for (u, v), edge_ids in undirected_map.items():
            count = edge_triangle_counts[(u, v)]
            
            if count == 0:
                bins['0_triangles'].extend(edge_ids)
            elif count == 1:
                bins['1_triangle'].extend(edge_ids)
            elif 2 <= count <= 5:
                bins['2_5_triangles'].extend(edge_ids)
            else:
                bins['6plus_triangles'].extend(edge_ids)
        
        # Filter empty bins
        coalitions = [edges for edges in bins.values() if edges]
        group_names = [name for name, edges in bins.items() if edges]
        
        return self._create_result(
            coalitions,
            group_names=group_names
        )
    
    def _build_clustering_groups(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> CoalitionResult:
        """Group edges by local clustering coefficient."""
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        undirected_map = self._get_undirected_edge_mapping(edge_index)
        for (u, v) in undirected_map.keys():
            G.add_edge(u, v)
        
        # Compute local clustering coefficient
        clustering = nx.clustering(G)
        
        # Group edges by average clustering of endpoints
        # Bins: low (0-0.33), medium (0.33-0.67), high (0.67-1.0)
        bins = {
            'low_clustering': [],
            'medium_clustering': [],
            'high_clustering': []
        }
        
        for (u, v), edge_ids in undirected_map.items():
            avg_clustering = (clustering[u] + clustering[v]) / 2
            
            if avg_clustering < 0.33:
                bins['low_clustering'].extend(edge_ids)
            elif avg_clustering < 0.67:
                bins['medium_clustering'].extend(edge_ids)
            else:
                bins['high_clustering'].extend(edge_ids)
        
        coalitions = [edges for edges in bins.values() if edges]
        group_names = [name for name, edges in bins.items() if edges]
        
        return self._create_result(
            coalitions,
            group_names=group_names
        )
    
    def _build_adjacency_set(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Dict[int, Set[int]]:
        """Build adjacency as set for fast membership test."""
        adj = defaultdict(set)
        
        for i in range(edge_index.size(1)):
            u = int(edge_index[0, i])
            v = int(edge_index[1, i])
            adj[u].add(v)
            adj[v].add(u)
        
        return dict(adj)


class FromNodeGroupsEdgeBuilder(EdgeCoalitionBuilder):
    """
    Map edge coalitions from node coalitions.
    
    Given node groups, assign edges based on endpoint group membership:
    - intra_group_k: both endpoints in group k
    - inter_group_i_j: endpoints in different groups i and j
    """
    
    def __init__(self):
        super().__init__('from_node_groups')
    
    def build(
        self,
        edge_index: torch.Tensor,
        node_to_group: Dict[int, int],
        num_node_groups: int,
        **kwargs
    ) -> CoalitionResult:
        """
        Build edge coalitions from node coalitions.
        
        Args:
            edge_index: [2, num_edges]
            node_to_group: Mapping from node ID to group index
            num_node_groups: Number of node groups
        
        Returns:
            CoalitionResult with edges grouped by endpoint groups
        """
        edge_groups = defaultdict(list)
        
        for edge_id in range(edge_index.size(1)):
            u = int(edge_index[0, edge_id])
            v = int(edge_index[1, edge_id])
            
            group_u = node_to_group.get(u, -1)
            group_v = node_to_group.get(v, -1)
            
            if group_u == -1 or group_v == -1:
                # Handle nodes not in any group
                key = 'ungrouped'
            elif group_u == group_v:
                key = f'intra_group_{group_u}'
            else:
                i, j = min(group_u, group_v), max(group_u, group_v)
                key = f'inter_group_{i}_{j}'
            
            edge_groups[key].append(edge_id)
        
        coalitions = list(edge_groups.values())
        group_names = list(edge_groups.keys())
        
        # Extract group types
        group_types = []
        for name in group_names:
            if name == 'ungrouped':
                group_types.append('ungrouped')
            elif name.startswith('intra'):
                group_types.append('intra')
            else:
                group_types.append('inter')
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            group_types=group_types,
            num_node_groups=num_node_groups
        )


# Factory function
def build_edge_coalitions(
    strategy: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    **kwargs
) -> CoalitionResult:
    """
    Factory function to build edge coalitions.
    
    Args:
        strategy: One of ['source_node', 'target_node', 'community', 
                          'motif_bridge', 'motif_triangle', 'motif_clustering',
                          'from_node_groups']
        edge_index: [2, num_edges]
        num_nodes: Number of nodes
        **kwargs: Strategy-specific arguments
    
    Returns:
        CoalitionResult
    """
    if strategy == 'source_node':
        builder = SourceNodeEdgeBuilder()
        return builder.build(edge_index)
    
    elif strategy == 'target_node':
        builder = TargetNodeEdgeBuilder()
        return builder.build(edge_index)
    
    elif strategy == 'community':
        builder = CommunityEdgeBuilder()
        return builder.build(edge_index, num_nodes, **kwargs)
    
    elif strategy.startswith('motif_'):
        motif_type = strategy.split('_', 1)[1]  # Extract type after 'motif_'
        builder = MotifEdgeBuilder(motif_type)
        return builder.build(edge_index, num_nodes, **kwargs)
    
    elif strategy == 'from_node_groups':
        builder = FromNodeGroupsEdgeBuilder()
        return builder.build(edge_index, **kwargs)
    
    else:
        raise ValueError(f"Unknown edge coalition strategy: {strategy}")
