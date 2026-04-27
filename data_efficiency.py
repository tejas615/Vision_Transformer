# Train both models on subsets: 10%, 25%, 50%, 100% of CIFAR-10
# Plot accuracy vs. dataset size

import torch
import random
from torch.utils.data import Subset, DataLoader
from models.vit import VisionTransformer
from models.hybrid import HybridViT
from configs.cifar_config import CIFAR10Config
from utils.data import get_cifar10_loaders
import matplotlib.pyplot as plt

def train_with_subset(model, train_loader, test_loader, subset_fraction, device, epochs=5):
    
    dataset = train_loader.dataset
    total_size = len(dataset)
    subset_size = int(total_size * subset_fraction)
    
    # Random subset indices
    indices = random.sample(range(total_size), subset_size)
    
    subset = Subset(dataset, indices)
    
    subset_loader = DataLoader(
        subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Fresh optimizer each time
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.to(device)
    
    # Train few epochs only
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
    
    accuracy = 100 * correct / total
    
    return accuracy


fractions = [0.1, 0.25, 0.5, 1.0]
vit_accs = []
hybrid_accs = []

config = CIFAR10Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_cifar10_loaders(config.batch_size)

for frac in fractions:
    # Train both models
    
    vit_model = VisionTransformer(config)   
    hybrid_model = HybridViT(config)
    
    vit_acc = train_with_subset(vit_model, train_loader, test_loader, frac, device)
    hybrid_acc = train_with_subset(hybrid_model, train_loader, test_loader, frac, device)
    
    vit_accs.append(vit_acc)
    hybrid_accs.append(hybrid_acc)

# Plot
plt.plot(fractions, vit_accs, marker='o', label='ViT')
plt.plot(fractions, hybrid_accs, marker='s', label='Hybrid ViT')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('data_efficiency.png')