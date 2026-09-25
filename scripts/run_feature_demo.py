"""Run from the repository root; all runtime business data remains local."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from entity_resolution.features.demo import main

if __name__ == '__main__':
    main()
