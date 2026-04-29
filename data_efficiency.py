import torch
import random
from torch.utils.data import Subset, DataLoader
from models.vit import VisionTransformer
from models.hybrid import HybridViT
from configs.cifar_config import CIFAR10Config
from utils.data import get_cifar10_loaders
import matplotlib.pyplot as plt

# Fix randomness
random.seed(42)
torch.manual_seed(42)

def train_with_subset(model, subset_loader, test_loader, device, epochs=10):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # fixed LR
    criterion = torch.nn.CrossEntropyLoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        
        for images, labels in subset_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


fractions = [0.1, 0.25, 0.5, 1.0]
vit_accs = []
hybrid_accs = []

config = CIFAR10Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_cifar10_loaders(config.batch_size)

dataset = train_loader.dataset
total_size = len(dataset)

# Generate ONE fixed shuffled index list
all_indices = list(range(total_size))
random.shuffle(all_indices)

for frac in fractions:
    
    subset_size = int(total_size * frac)
    indices = all_indices[:subset_size]   # SAME subset for both models
    
    subset = Subset(dataset, indices)
    
    subset_loader = DataLoader(
        subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print(f"\nTraining with {int(frac*100)}% data...")
    
    vit_model = VisionTransformer(config)   
    hybrid_model = HybridViT(config)
    
    vit_acc = train_with_subset(vit_model, subset_loader, test_loader, device)
    hybrid_acc = train_with_subset(hybrid_model, subset_loader, test_loader, device)
    
    vit_accs.append(vit_acc)
    hybrid_accs.append(hybrid_acc)

# Print correctly
print("ViT Accuracies:", vit_accs)
print("Hybrid Accuracies:", hybrid_accs)

# Plot
plt.figure(figsize=(8,6))
plt.plot(fractions, vit_accs, marker='o', label='ViT')
plt.plot(fractions, hybrid_accs, marker='s', label='Hybrid ViT')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/data_efficiency.png')