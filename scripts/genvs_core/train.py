import sys
from pathlib import Path

# Add project root and beta paths to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BETA_PATH = PROJECT_ROOT / "hypersplat" / "beta"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BETA_PATH) not in sys.path:
    sys.path.insert(0, str(BETA_PATH))

if __name__ == "__main__":
    import hypersplat.beta.genvs.train as train

    # Delegate to the training module's own parser so this shim can never
    # drift from hypersplat/beta/genvs/train.py (flags, defaults, --resume).
    parser = train.build_parser()
    args = parser.parse_args()

    trainer = train.GeNVSTrainer(args)
    trainer.train()
