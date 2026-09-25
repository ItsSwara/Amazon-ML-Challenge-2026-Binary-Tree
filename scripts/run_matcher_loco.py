"""Stage 5: train on US, pick the F0.5 threshold on India (mock handoff)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from entity_resolution.models.loco import main

if __name__ == '__main__':
    main()
