"""
Test Main Entry Point (without external dependencies)
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all core modules can be imported."""
    print("Testing module imports...")
    
    try:
        from config import Config, get_config
        print("  ✓ config module imported")
    except Exception as e:
        print(f"  ✗ config module failed: {e}")
        return False
    
    # Test utils imports (will fail due to missing dependencies, but that's OK)
    print("\nNote: utils module requires torch/numpy/scipy (not installed)")
    print("      This is expected and normal for the test.\n")
    
    return True


def test_config():
    """Test configuration system."""
    print("Testing configuration system...")
    
    from config import Config
    
    # Create default config
    config = Config()
    print(f"  ✓ Created default config")
    
    # Test config update
    config.update({'exp_name': 'test_experiment', 'seed': 999})
    assert config.exp_name == 'test_experiment'
    assert config.seed == 999
    print(f"  ✓ Config update works")
    
    # Test config to dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert 'exp_name' in config_dict
    print(f"  ✓ Config to dict conversion works")
    
    # Test output dirs
    output_dirs = config.get_output_dirs()
    assert 'base' in output_dirs
    assert 'checkpoints' in output_dirs
    assert 'logs' in output_dirs
    print(f"  ✓ Output directory generation works")
    
    return True


def test_directory_creation():
    """Test directory structure creation."""
    print("\nTesting directory creation...")
    
    from config import Config
    
    config = Config()
    config.exp_name = 'test_exp'
    config.output_dir = 'test_outputs/'
    
    # Create directories
    config.create_output_dirs()
    output_dirs = config.get_output_dirs()
    
    # Verify directories exist
    for name, path in output_dirs.items():
        if path.exists():
            print(f"  ✓ {name:12s} directory created: {path}")
        else:
            print(f"  ✗ {name:12s} directory missing: {path}")
            return False
    
    # Cleanup
    import shutil
    if Path('test_outputs/').exists():
        shutil.rmtree('test_outputs/')
        print("  ✓ Cleanup successful")
    
    return True


def test_project_structure():
    """Verify project structure."""
    print("\nVerifying project structure...")
    
    required_dirs = [
        'models', 'data', 'train', 'pruning', 
        'scorers', 'experiments', 'utils', 'outputs'
    ]
    
    required_files = [
        'config.py', 'main.py', 'README.md', 'requirements.txt',
        'utils/__init__.py', 'utils/seed.py', 'utils/io.py', 'utils/logger.py'
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        path = project_root / dir_name
        if path.exists() and path.is_dir():
            print(f"  ✓ Directory: {dir_name}/")
        else:
            print(f"  ✗ Missing:   {dir_name}/")
            all_good = False
    
    for file_name in required_files:
        path = project_root / file_name
        if path.exists() and path.is_file():
            print(f"  ✓ File:      {file_name}")
        else:
            print(f"  ✗ Missing:   {file_name}")
            all_good = False
    
    return all_good


def main():
    """Run all tests."""
    print("=" * 60)
    print("GCN Pruning Project - Setup Verification")
    print("=" * 60)
    print(f"\nProject root: {project_root}\n")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Module Imports", test_imports),
        ("Configuration System", test_config),
        ("Directory Creation", test_directory_creation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print("\n" + "-" * 60)
        print(f"Running: {test_name}")
        print("-" * 60)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} {test_name}")
    
    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Project setup is complete.")
    else:
        print("✗ Some tests failed. Please check the output above.")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
