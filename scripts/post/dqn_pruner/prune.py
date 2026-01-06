import argparse
import sys
import os
import torch
import time

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PrunerConfig
from state_extractor import StateExtractor
from agent import DQNAgent, HeuristicAgent, ContextAwareAgent

def prune_ply(input_path, output_path, model_path=None, context_aware=True):
    """
    Prune floaters from a 3DGS PLY file.
    
    Args:
        input_path: Path to input PLY file
        output_path: Path for output cleaned PLY
        model_path: Optional path to trained DQN model
        context_aware: Use context-aware multi-feature pruning (default: True)
    """
    print(f"[Pruner] Loading config...")
    config = PrunerConfig()
    config.CONTEXT_AWARE = context_aware
    
    print(f"[Pruner] Loading PLY: {input_path}")
    extractor = StateExtractor(config)
    
    start_time = time.time()
    try:
        data = extractor.load_ply(input_path)
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found.")
        return False
        
    print(f"[Pruner] Extracting features (N={data['positions'].shape[0]})...")
    states = extractor.extract_features(data)
    
    # Select Agent (priority: DQN > ContextAware > Heuristic)
    if model_path and os.path.exists(model_path):
        print(f"[Pruner] Loading DQN model from {model_path}")
        agent = DQNAgent(config)
        agent.load(model_path)
        actions = agent.predict(states)
    elif context_aware:
        print(f"[Pruner] Using Context-Aware Agent (multi-feature scoring)")
        agent = ContextAwareAgent(config)
        actions = agent.predict(states, data)
    else:
        print(f"[Pruner] Using Heuristic Agent (basic thresholds)")
        agent = HeuristicAgent(config)
        actions = agent.predict(states, data)
        
    # Apply Actions
    # Action 1 = PRUNE, Action 0 = KEEP
    keep_mask = (actions == 0)
    
    n_original = len(states)
    n_keep = keep_mask.sum().item()
    n_pruned = n_original - n_keep
    percent = (n_pruned / n_original) * 100
    
    print(f"[Pruner] Final: KEEP={n_keep}, PRUNE={n_pruned} ({percent:.2f}%)")
    
    print(f"[Pruner] Saving cleaned PLY to {output_path}")
    extractor.save_ply(data['plydata'], keep_mask, output_path)
    
    elapsed = time.time() - start_time
    print(f"[Pruner] Complete in {elapsed:.2f}s.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS Floater Pruner: Context-Aware Cleanup")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input noisy .ply file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output cleaned .ply file")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to trained DQN .pt file")
    parser.add_argument("--context-aware", dest="context_aware", action="store_true", default=True,
                        help="Use context-aware multi-feature pruning (default)")
    parser.add_argument("--basic", dest="context_aware", action="store_false",
                        help="Use basic heuristic pruning instead")
    
    args = parser.parse_args()
    
    success = prune_ply(args.input, args.output, args.model, args.context_aware)
    if not success:
        sys.exit(1)
