import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(log_file, save_path):
    # Parse log file
    epochs, train_loss, train_acc, test_loss, test_acc = [], [], [], [], []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
        
            # Skip empty lines or header
            if not line or line.startswith("epoch"):
                continue
        
            parts = line.split(',')
        
            epochs.append(int(parts[0]))
            train_loss.append(float(parts[1]))
            train_acc.append(float(parts[2]))
            test_loss.append(float(parts[3]))
            test_acc.append(float(parts[4]))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, train_loss, label='Train')
    ax1.plot(epochs, test_loss, label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_acc, label='Train')
    ax2.plot(epochs, test_acc, label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)