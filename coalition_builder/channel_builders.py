"""
Channel Coalition Builders

Implements strategies for grouping channels into prior coalitions.

Channel player ID = channel index within a layer or across layers.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional

from .base import CoalitionBuilder, CoalitionResult


class ChannelCoalitionBuilder(CoalitionBuilder):
    """Base class for channel coalition strategies."""
    
    def __init__(self, strategy: str):
        super().__init__(strategy)


class UniformFixedSizeChannelBuilder(ChannelCoalitionBuilder):
    """
    Group channels into uniform fixed-size groups.
    
    Example:
        num_channels = 10, group_size = 4
        Result: [[0,1,2,3], [4,5,6,7], [8,9]]
    """
    
    def __init__(self):
        super().__init__('uniform_fixed_size')
    
    def build(
        self,
        num_channels: int,
        group_size: int = 4,
        **kwargs
    ) -> CoalitionResult:
        """
        Group channels into fixed-size groups.
        
        Args:
            num_channels: Total number of channels
            group_size: Target group size (last group may be smaller)
        
        Returns:
            CoalitionResult with uniform groups
        """
        assert group_size > 0, "group_size must be positive"
        
        coalitions = []
        group_names = []
        
        for start in range(0, num_channels, group_size):
            end = min(start + group_size, num_channels)
            coalition = list(range(start, end))
            coalitions.append(coalition)
            group_names.append(f'channels_{start}_{end-1}')
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            group_size=group_size,
            num_channels=num_channels
        )


class LayerWiseChannelBuilder(ChannelCoalitionBuilder):
    """
    Group channels by layer.
    
    Each layer's channels form one coalition.
    Useful when channels have layer-specific semantics.
    """
    
    def __init__(self):
        super().__init__('layerwise')
    
    def build(
        self,
        layer_channel_counts: List[int],
        layer_names: Optional[List[str]] = None,
        **kwargs
    ) -> CoalitionResult:
        """
        Group channels by layer.
        
        Args:
            layer_channel_counts: Number of channels per layer [64, 64, 32]
            layer_names: Optional layer names
        
        Returns:
            CoalitionResult with one group per layer
        """
        coalitions = []
        group_names = []
        
        offset = 0
        for i, count in enumerate(layer_channel_counts):
            coalition = list(range(offset, offset + count))
            coalitions.append(coalition)
            
            if layer_names and i < len(layer_names):
                name = f'layer_{layer_names[i]}'
            else:
                name = f'layer_{i}'
            group_names.append(name)
            
            offset += count
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            layer_counts=layer_channel_counts,
            num_layers=len(layer_channel_counts)
        )


class RandomChannelBuilder(ChannelCoalitionBuilder):
    """
    Random grouping of channels.
    
    Useful as baseline for comparing structured groupings.
    """
    
    def __init__(self):
        super().__init__('random')
    
    def build(
        self,
        num_channels: int,
        num_groups: int = 4,
        seed: int = 42,
        **kwargs
    ) -> CoalitionResult:
        """
        Randomly assign channels to groups.
        
        Args:
            num_channels: Total number of channels
            num_groups: Number of groups to create
            seed: Random seed
        
        Returns:
            CoalitionResult with random groups
        """
        rng = np.random.RandomState(seed)
        
        # Assign each channel to a random group
        assignments = rng.randint(0, num_groups, size=num_channels)
        
        # Build coalitions
        coalitions = [[] for _ in range(num_groups)]
        for channel_id, group_id in enumerate(assignments):
            coalitions[group_id].append(channel_id)
        
        # Filter empty groups
        coalitions = [c for c in coalitions if c]
        group_names = [f'random_group_{i}' for i in range(len(coalitions))]
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            num_channels=num_channels,
            seed=seed
        )


class FunctionalChannelBuilder(ChannelCoalitionBuilder):
    """
    Group channels by functional roles (based on activation patterns).
    
    This is a more advanced strategy that requires activation data.
    """
    
    def __init__(self):
        super().__init__('functional')
    
    def build(
        self,
        activations: torch.Tensor,
        num_groups: int = 4,
        method: str = 'kmeans',
        **kwargs
    ) -> CoalitionResult:
        """
        Group channels by activation similarity.
        
        Args:
            activations: [num_samples, num_channels] activation patterns
            num_groups: Number of functional groups
            method: Clustering method ('kmeans' or 'spectral')
        
        Returns:
            CoalitionResult with functionally similar channels grouped
        """
        from sklearn.cluster import KMeans, SpectralClustering
        
        # Normalize activations
        act_mean = activations.mean(dim=0, keepdim=True)
        act_std = activations.std(dim=0, keepdim=True) + 1e-8
        act_norm = (activations - act_mean) / act_std
        
        # Cluster channels
        X = act_norm.t().cpu().numpy()  # [num_channels, num_samples]
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=num_groups, random_state=42)
        elif method == 'spectral':
            clusterer = SpectralClustering(n_clusters=num_groups, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        labels = clusterer.fit_predict(X)
        
        # Build coalitions
        coalitions = [[] for _ in range(num_groups)]
        for channel_id, label in enumerate(labels):
            coalitions[label].append(channel_id)
        
        # Filter empty groups
        coalitions = [c for c in coalitions if c]
        group_names = [f'functional_group_{i}' for i in range(len(coalitions))]
        
        return self._create_result(
            coalitions,
            group_names=group_names,
            method=method
        )


# Factory function
def build_channel_coalitions(
    strategy: str,
    num_channels: Optional[int] = None,
    **kwargs
) -> CoalitionResult:
    """
    Factory function to build channel coalitions.
    
    Args:
        strategy: One of ['uniform_fixed_size', 'layerwise', 'random', 'functional']
        num_channels: Total number of channels (required for most strategies)
        **kwargs: Strategy-specific arguments
    
    Returns:
        CoalitionResult
    
    Examples:
        # Uniform fixed size
        result = build_channel_coalitions('uniform_fixed_size', num_channels=64, group_size=8)
        
        # Layerwise
        result = build_channel_coalitions('layerwise', layer_channel_counts=[64, 64, 32])
        
        # Random
        result = build_channel_coalitions('random', num_channels=128, num_groups=8)
    """
    if strategy == 'uniform_fixed_size':
        assert num_channels is not None, "num_channels required for uniform_fixed_size"
        builder = UniformFixedSizeChannelBuilder()
        return builder.build(num_channels, **kwargs)
    
    elif strategy == 'layerwise':
        builder = LayerWiseChannelBuilder()
        return builder.build(**kwargs)
    
    elif strategy == 'random':
        assert num_channels is not None, "num_channels required for random"
        builder = RandomChannelBuilder()
        return builder.build(num_channels, **kwargs)
    
    elif strategy == 'functional':
        builder = FunctionalChannelBuilder()
        return builder.build(**kwargs)
    
    else:
        raise ValueError(f"Unknown channel coalition strategy: {strategy}")
