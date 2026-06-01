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
    import argparse
    import hypersplat.beta.genvs.train as train
    
    # Delegate to the training module
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ACE output directory")
    parser.add_argument("--output_dir", type=str, default="results/genvs_train")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--eco_mode", action="store_true", help="Enable Eco Mode")
    parser.add_argument("--eco_sleep", type=float, default=0.1, help="Sleep duration in seconds for Eco Mode")
    
    args = parser.parse_args()
    
    trainer = train.GeNVSTrainer(args)
    trainer.train()
