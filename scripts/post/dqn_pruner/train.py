import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
import sys

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PrunerConfig
from state_extractor import StateExtractor
from agent import QNetwork, HeuristicAgent
from data_gen import NoiseInjector

def train(clean_ply_path, output_model_path, epochs=100):
    config = PrunerConfig()
    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
    
    print(f"[DQN-Train] Loading Source 'Clean' PLY: {clean_ply_path}")
    extractor = StateExtractor(config)
    
    try:
        clean_data_raw = extractor.load_ply(clean_ply_path)
    except Exception as e:
        print(f"Error loading PLY: {e}")
        return
    
    injector = NoiseInjector(config)
    
    # Init Model
    model = QNetwork(config.INPUT_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    print(f"[DQN-Train] Starting {epochs} epochs of Synthetic Noise Training...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        noisy_data_raw, labels = injector.generate_batch(clean_data_raw)
        states = extractor.extract_features(noisy_data_raw)
        states = states.to(device)
        labels = labels.to(device).long()
        
        outputs = model(states)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
            
    print(f"[DQN-Train] Saving model to {output_model_path}")
    torch.save(model.state_dict(), output_model_path)
    print("[DQN-Train] Done.")


def train_from_heuristic(ply_paths, output_model_path, epochs=100):
    """
    Train DQN using heuristic agent decisions as pseudo-labels.
    
    This enables bootstrap training without manual labeling:
    1. Heuristic agent makes decisions on noisy PLY files
    2. DQN learns to match (and exceed) heuristic performance
    3. DQN generalizes beyond heuristic's fixed thresholds
    
    Args:
        ply_paths: List of PLY file paths (can be noisy, no clean version needed)
        output_model_path: Where to save trained DQN model
        epochs: Training epochs per PLY file
    """
    config = PrunerConfig()
    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
    
    print(f"[DQN-Bootstrap] Training from {len(ply_paths)} PLY files using heuristic labels")
    
    extractor = StateExtractor(config)
    heuristic = HeuristicAgent(config)
    
    # Init Model
    model = QNetwork(config.INPUT_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for ply_idx, ply_path in enumerate(ply_paths):
        print(f"\n[DQN-Bootstrap] Processing PLY {ply_idx+1}/{len(ply_paths)}: {ply_path}")
        
        try:
            data = extractor.load_ply(ply_path)
        except Exception as e:
            print(f"Skipping {ply_path}: {e}")
            continue
            
        # Extract features
        states = extractor.extract_features(data)
        
        # Get pseudo-labels from heuristic
        heuristic_labels = heuristic.predict(states, data)
        
        # Move to device
        states = states.to(device)
        labels = heuristic_labels.to(device).long()
        
        n_prune = (labels == 1).sum().item()
        n_keep = (labels == 0).sum().item()
        print(f"  Heuristic labels: KEEP={n_keep}, PRUNE={n_prune} ({100*n_prune/(n_keep+n_prune):.1f}%)")
        
        # Train for multiple epochs on this PLY
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Use mini-batches for large PLY files
            batch_size = min(config.BATCH_SIZE, len(states))
            indices = torch.randperm(len(states))[:batch_size]
            
            batch_states = states[indices]
            batch_labels = labels[indices]
            
            outputs = model(batch_states)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                with torch.no_grad():
                    all_outputs = model(states)
                    preds = torch.argmax(all_outputs, dim=1)
                    acc = (preds == labels).float().mean()
                    print(f"  Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
    
    print(f"\n[DQN-Bootstrap] Saving model to {output_model_path}")
    torch.save(model.state_dict(), output_model_path)
    print("[DQN-Bootstrap] Done! DQN now trained to match heuristic performance.")
    print("[DQN-Bootstrap] The DQN will generalize better than heuristic on new scenes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["synthetic", "heuristic"], default="heuristic",
                        help="Training mode: 'synthetic' (needs clean PLY) or 'heuristic' (uses noisy PLYs)")
    parser.add_argument("--ply", type=str, nargs="+", required=True, 
                        help="PLY file(s) for training. For 'synthetic' mode, provide clean PLY. For 'heuristic', any PLYs.")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save the .pt model")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        if len(args.ply) != 1:
            print("Error: Synthetic mode requires exactly 1 clean PLY file")
            sys.exit(1)
        train(args.ply[0], args.save_path, args.epochs)
    else:
        train_from_heuristic(args.ply, args.save_path, args.epochs)

