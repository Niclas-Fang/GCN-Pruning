"""
Coalition Builder Base Classes

Provides unified interface for building prior coalitions for Owen-Shapley.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import torch


@dataclass
class CoalitionResult:
    """
    Standard output format for coalition builders.
    
    Attributes:
        coalitions: List of coalitions, each coalition is a list of player IDs
        player_to_group: Mapping from player ID to group index
        group_metadata: Metadata for each group (name, size, properties)
        strategy: Name of the grouping strategy
    """
    coalitions: List[List[int]]
    player_to_group: Dict[int, int]
    group_metadata: List[Dict[str, Any]]
    strategy: str
    
    def validate(self) -> bool:
        """Validate coalition structure."""
        # Check all players appear exactly once
        all_players = set()
        for coalition in self.coalitions:
            all_players.update(coalition)
        
        # Check player_to_group covers all players
        mapped_players = set(self.player_to_group.keys())
        assert all_players == mapped_players, \
            f"Player set mismatch: {len(all_players)} vs {len(mapped_players)}"
        
        # Check group count matches
        assert len(self.coalitions) == len(self.group_metadata), \
            f"Coalition count mismatch with metadata"
        
        # Check player_to_group values are valid
        for player, group_idx in self.player_to_group.items():
            assert 0 <= group_idx < len(self.coalitions), \
                f"Invalid group index {group_idx} for player {player}"
            assert player in self.coalitions[group_idx], \
                f"Player {player} not in assigned coalition {group_idx}"
        
        return True
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Coalition Strategy: {self.strategy}",
            f"Number of groups: {len(self.coalitions)}",
            f"Total players: {len(self.player_to_group)}",
            f"Group sizes: {[len(c) for c in self.coalitions]}",
        ]
        
        # Show first few groups
        for i, (coalition, meta) in enumerate(zip(self.coalitions[:3], self.group_metadata[:3])):
            lines.append(f"  Group {i}: {meta.get('name', 'unnamed')} (size={len(coalition)})")
        
        if len(self.coalitions) > 3:
            lines.append(f"  ... ({len(self.coalitions) - 3} more groups)")
        
        return "\n".join(lines)


class CoalitionBuilder(ABC):
    """Abstract base class for coalition builders."""
    
    def __init__(self, strategy: str):
        """
        Initialize builder.
        
        Args:
            strategy: Name of the strategy
        """
        self.strategy = strategy
    
    @abstractmethod
    def build(self, *args, **kwargs) -> CoalitionResult:
        """
        Build coalitions.
        
        Returns:
            CoalitionResult with coalitions, mappings, and metadata
        """
        pass
    
    def _create_result(
        self,
        coalitions: List[List[int]],
        group_names: Optional[List[str]] = None,
        **metadata_fields
    ) -> CoalitionResult:
        """
        Helper to create CoalitionResult with standard format.
        
        Args:
            coalitions: List of player ID lists
            group_names: Optional names for groups
            **metadata_fields: Additional metadata fields
        
        Returns:
            CoalitionResult
        """
        # Build player_to_group mapping
        player_to_group = {}
        for g_idx, coalition in enumerate(coalitions):
            for player in coalition:
                player_to_group[player] = g_idx
        
        # Build metadata
        group_metadata = []
        for g_idx, coalition in enumerate(coalitions):
            meta = {
                'group_id': g_idx,
                'size': len(coalition),
                'name': group_names[g_idx] if group_names else f'group_{g_idx}',
            }
            # Add strategy-specific metadata
            for key, values in metadata_fields.items():
                if isinstance(values, (list, tuple)) and len(values) == len(coalitions):
                    meta[key] = values[g_idx]
                else:
                    meta[key] = values
            group_metadata.append(meta)
        
        result = CoalitionResult(
            coalitions=coalitions,
            player_to_group=player_to_group,
            group_metadata=group_metadata,
            strategy=self.strategy
        )
        
        # Validate before returning
        result.validate()
        
        return result
