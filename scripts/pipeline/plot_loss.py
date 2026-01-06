"""Plot ACE-Zero and 3DGS loss curves from pipeline logs."""
import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_losses(log_path: str):
    """Extract losses from pipeline log."""
    ace_losses = []
    dgs_losses = []
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # ACE-Zero loss pattern
            ace_match = re.search(r'Iteration:\s*(\d+)\|.*Loss:\s*([\d.]+),\s*Batch inliers.*:\s*([\d.]+)%', line)
            if ace_match:
                step = int(ace_match.group(1))
                loss = float(ace_match.group(2))
                inliers = float(ace_match.group(3))
                ace_losses.append((step, loss, inliers))
            
            # 3DGS loss pattern (if present)
            dgs_match = re.search(r'step=(\d+)\|.*loss=([\d.e+-]+)', line)
            if dgs_match:
                step = int(dgs_match.group(1))
                loss = float(dgs_match.group(2))
                dgs_losses.append((step, loss))
    
    return ace_losses, dgs_losses

def plot_losses(log_path: str, output_path: str = None):
    """Create loss curve plot."""
    ace_losses, dgs_losses = parse_losses(log_path)
    
    fig, axes = plt.subplots(1, 2 if dgs_losses else 1, figsize=(12, 5))
    if not isinstance(axes, list) and not hasattr(axes, '__iter__'):
        axes = [axes]
    
    # ACE-Zero Loss
    if ace_losses:
        steps = [x[0] for x in ace_losses]
        losses = [x[1] for x in ace_losses]
        inliers = [x[2] for x in ace_losses]
        
        ax1 = axes[0]
        ax1.plot(steps, losses, 'b-', label='Loss', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss', color='blue')
        ax1.set_title('ACE-Zero Training')
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis for inliers
        ax1b = ax1.twinx()
        ax1b.plot(steps, inliers, 'g--', label='Inliers (%)', alpha=0.7)
        ax1b.set_ylabel('Inliers (%)', color='green')
        ax1b.set_ylim(0, 100)
        
        ax1.legend(loc='upper left')
        ax1b.legend(loc='upper right')
    
    # 3DGS Loss
    if dgs_losses and len(axes) > 1:
        steps = [x[0] for x in dgs_losses]
        losses = [x[1] for x in dgs_losses]
        
        ax2 = axes[1]
        ax2.semilogy(steps, losses, 'r-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss (log scale)')
        ax2.set_title('3DGS Training')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    # Print summary
    if ace_losses:
        print(f"\nACE-Zero: {len(ace_losses)} data points")
        print(f"  Loss: {ace_losses[0][1]:.1f} -> {ace_losses[-1][1]:.1f}")
        print(f"  Inliers: {ace_losses[0][2]:.1f}% -> {ace_losses[-1][2]:.1f}%")
    
    if dgs_losses:
        print(f"\n3DGS: {len(dgs_losses)} data points")
        print(f"  Loss: {dgs_losses[0][1]:.4f} -> {dgs_losses[-1][1]:.4f}")

if __name__ == "__main__":
    import sys
    log_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_log.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "loss_curve.png"
    plot_losses(log_path, output_path)
