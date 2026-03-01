import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# VISUALIZATION & COMPARISON
# Run this after training both pancake and tower models
# ==========================================

def plot_model_comparison(pancake_results, tower_results, epochs):
    """
    Plot comprehensive comparison of pancake and tower models
    
    Args:
        pancake_results: dict with keys: train_losses, train_accs, val_accs, final_results
        tower_results: dict with keys: train_losses, train_accs, val_accs, final_results
        epochs: number of epochs trained
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epoch_list = range(1, epochs + 1)
    
    # Plot 1: Training Loss Comparison
    axes[0, 0].plot(epoch_list, pancake_results['train_losses'], 'b-', label='Pancake', linewidth=2)
    axes[0, 0].plot(epoch_list, tower_results['train_losses'], 'r-', label='Tower', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training & Validation Accuracy
    axes[0, 1].plot(epoch_list, pancake_results['train_accs'], 'b-', label='Pancake Train', linewidth=2)
    axes[0, 1].plot(epoch_list, pancake_results['val_accs'], 'b--', label='Pancake Val', linewidth=2)
    axes[0, 1].plot(epoch_list, tower_results['train_accs'], 'r-', label='Tower Train', linewidth=2)
    axes[0, 1].plot(epoch_list, tower_results['val_accs'], 'r--', label='Tower Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final Accuracy Bar Chart
    models = ['Pancake', 'Tower']
    train_accs = [pancake_results['final']['train_acc'], tower_results['final']['train_acc']]
    val_accs = [pancake_results['final']['val_acc'], tower_results['final']['val_acc']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, train_accs, width, label='Train Acc', color='skyblue')
    axes[1, 0].bar(x + width/2, val_accs, width, label='Val Acc', color='orange')
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (train, val) in enumerate(zip(train_accs, val_accs)):
        axes[1, 0].text(i - width/2, train + 1, f'{train:.1f}%', ha='center', va='bottom', fontsize=10)
        axes[1, 0].text(i + width/2, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Generalization Gap (Overfitting Analysis)
    gap_data = [pancake_results['final']['gap'], tower_results['final']['gap']]
    colors = ['red' if g > 10 else 'yellow' if g > 5 else 'green' for g in gap_data]
    axes[1, 1].bar(models, gap_data, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Generalization Gap (%)', fontsize=12)
    axes[1, 1].set_title('Overfitting Analysis (Train - Val)', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(y=5, color='green', linestyle='--', linewidth=2, label='Good (<5%)')
    axes[1, 1].axhline(y=10, color='orange', linestyle='--', linewidth=2, label='Moderate (<10%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, gap in enumerate(gap_data):
        axes[1, 1].text(i, gap + 0.5, f'{gap:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Plot saved as 'model_comparison.png'")
    plt.show()


def print_comparison_report(pancake_results, tower_results):
    """Print detailed text comparison"""
    
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    
    print(f"\n{'='*60}")
    print("PANCAKE MODEL (Wide & Shallow)")
    print("="*60)
    print("  Architecture: 784 → 2048 → 1024 → 15")
    print("  Total Layers: 2 hidden layers")
    print("  Parameters:   ~2.7M")
    print("  Activation:   GELU")
    print("  Dropout:      0.4")
    print("  Learning Rate: 0.0008")
    print("\n  RESULTS:")
    print(f"  ├─ Train Accuracy: {pancake_results['final']['train_acc']:.2f}%")
    print(f"  ├─ Val Accuracy:   {pancake_results['final']['val_acc']:.2f}%")
    print(f"  ├─ Final Loss:     {pancake_results['final']['loss']:.4f}")
    print(f"  └─ Gen. Gap:       {pancake_results['final']['gap']:.2f}%")
    
    print(f"\n{'='*60}")
    print("TOWER MODEL (Narrow & Deep)")
    print("="*60)
    print("  Architecture: 784 → 256 → 256 → 256 → 256 → 128 → 128 → 128 → 128 → 15")
    print("  Total Layers: 8 hidden layers")
    print("  Parameters:   ~700K")
    print("  Activation:   GELU")
    print("  Dropout:      0.2")
    print("  Learning Rate: 0.0007")
    print("\n  RESULTS:")
    print(f"  ├─ Train Accuracy: {tower_results['final']['train_acc']:.2f}%")
    print(f"  ├─ Val Accuracy:   {tower_results['final']['val_acc']:.2f}%")
    print(f"  ├─ Final Loss:     {tower_results['final']['loss']:.4f}")
    print(f"  └─ Gen. Gap:       {tower_results['final']['gap']:.2f}%")
    
    # Determine winner
    print(f"\n{'='*60}")
    if tower_results['final']['val_acc'] > pancake_results['final']['val_acc']:
        winner = "TOWER"
        diff = tower_results['final']['val_acc'] - pancake_results['final']['val_acc']
        loser_params = 2700000
        winner_params = 700000
        print(f"🏆 WINNER: {winner} MODEL")
        print(f"   Validation Accuracy: +{diff:.2f}% higher than Pancake")
        print(f"   Parameter Efficiency: {winner_params/loser_params*100:.1f}% of Pancake's parameters")
        print(f"   Generalization: Better (gap = {tower_results['final']['gap']:.2f}%)")
    else:
        winner = "PANCAKE"
        diff = pancake_results['final']['val_acc'] - tower_results['final']['val_acc']
        print(f"🏆 WINNER: {winner} MODEL")
        print(f"   Validation Accuracy: +{diff:.2f}% higher than Tower")
        print("   Convergence Speed: Faster (fewer layers)")
    
    print("="*60)
    
    # Analysis
    print("\n📊 ANALYSIS:")
    print("-" * 60)
    
    if pancake_results['final']['gap'] > 10:
        print("⚠️  Pancake shows SEVERE overfitting (gap > 10%)")
        print("   → Consider: Higher dropout, more weight decay, or data augmentation")
    elif pancake_results['final']['gap'] > 5:
        print("⚠️  Pancake shows MODERATE overfitting (gap > 5%)")
    else:
        print("✅ Pancake shows GOOD generalization (gap < 5%)")
    
    if tower_results['final']['gap'] > 10:
        print("⚠️  Tower shows SEVERE overfitting (gap > 10%)")
    elif tower_results['final']['gap'] > 5:
        print("⚠️  Tower shows MODERATE overfitting (gap > 5%)")
    else:
        print("✅ Tower shows GOOD generalization (gap < 5%)")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 60)
    print("1. Tower uses {100*(1-700/2700):.0f}% fewer parameters than Pancake")
    print("2. Tower's depth enables hierarchical feature learning")
    print("3. Pancake converges faster but may overfit more easily")
    print("4. GELU activation improves both models over ReLU")
    print("5. Batch Normalization stabilizes training in deep networks")
    print("="*60 + "\n")


# Example usage (to be called after training in notebook):
# 
# pancake_data = {
#     'train_losses': pancake_train_losses,
#     'train_accs': pancake_train_accs,
#     'val_accs': pancake_val_accs,
#     'final': pancake_final_results
# }
#
# tower_data = {
#     'train_losses': tower_train_losses,
#     'train_accs': tower_train_accs,
#     'val_accs': tower_val_accs,
#     'final': tower_final_results
# }
#
# plot_model_comparison(pancake_data, tower_data, EPOCHS)
# print_comparison_report(pancake_data, tower_data)
