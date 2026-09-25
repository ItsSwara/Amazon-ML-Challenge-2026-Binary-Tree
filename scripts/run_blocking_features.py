"""Run the local mock retrieval-to-features handoff from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from entity_resolution.features.handoff import main

if __name__ == '__main__':
    main()
