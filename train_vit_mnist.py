import torch
from models.vit import VisionTransformer
from utils.data import get_mnist_loaders
from utils.train import train_epoch, evaluate
from configs.config import Config

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, test_loader = get_mnist_loaders(config.batch_size)
    
    # Model
    model = VisionTransformer(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, 
                                weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'experiments/vit_mnist_best.pth')
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()