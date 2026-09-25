"""Stage 8: batched test inference and submission export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from entity_resolution.inference.run_inference import main

if __name__ == '__main__':
    main()
