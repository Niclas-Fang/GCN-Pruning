"""
Coalition Builder Module

Provides unified interface for building prior coalitions for Owen-Shapley.

Usage:
    from coalition_builder import build_edge_coalitions, build_channel_coalitions
    
    # Edge coalitions
    result = build_edge_coalitions('community', edge_index, num_nodes)
    
    # Channel coalitions
    result = build_channel_coalitions('uniform_fixed_size', num_channels=64, group_size=8)
"""
from .base import CoalitionResult, CoalitionBuilder
from .edge_builders import (
    build_edge_coalitions,
    SourceNodeEdgeBuilder,
    TargetNodeEdgeBuilder,
    CommunityEdgeBuilder,
    MotifEdgeBuilder,
    FromNodeGroupsEdgeBuilder,
)
from .channel_builders import (
    build_channel_coalitions,
    UniformFixedSizeChannelBuilder,
    LayerWiseChannelBuilder,
    RandomChannelBuilder,
    FunctionalChannelBuilder,
)

__all__ = [
    # Base
    'CoalitionResult',
    'CoalitionBuilder',
    
    # Edge builders
    'build_edge_coalitions',
    'SourceNodeEdgeBuilder',
    'TargetNodeEdgeBuilder',
    'CommunityEdgeBuilder',
    'MotifEdgeBuilder',
    'FromNodeGroupsEdgeBuilder',
    
    # Channel builders
    'build_channel_coalitions',
    'UniformFixedSizeChannelBuilder',
    'LayerWiseChannelBuilder',
    'RandomChannelBuilder',
    'FunctionalChannelBuilder',
]