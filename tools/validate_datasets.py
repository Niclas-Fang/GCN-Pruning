"""Validate that CiteSeer / PubMed / Cora load correctly from the Planetoid cache."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data import load_dataset, get_dataset_info


def main():
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        info = get_dataset_info(name)
        data = load_dataset(name, str(project_root / 'data'))
        print(f"\n=== {name} ===")
        print(f"  expected: {info}")
        print(f"  loaded:   nodes={data.num_nodes}, features={data.num_features}, "
              f"classes={data.num_classes}, edges={data.edge_index.size(1)}")
        print(f"  masks:    train={int(data.train_mask.sum())}, "
              f"val={int(data.val_mask.sum())}, test={int(data.test_mask.sum())}")
        assert data.num_nodes == info['num_nodes']
        assert data.num_features == info['num_features']
        assert data.num_classes == info['num_classes']
        assert int(data.train_mask.sum()) == info['train_size']
        assert int(data.val_mask.sum()) == info['val_size']
        assert int(data.test_mask.sum()) == info['test_size']
        print("  ✓ OK")


if __name__ == '__main__':
    main()
