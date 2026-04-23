import torch
from models.hybrid import HybridViT
from utils.data import get_cifar10_loaders
from utils.train import train_epoch, evaluate,WarmupCosineScheduler
from configs.cifar_config import CIFAR10Config

def main():
    config = CIFAR10Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(config.batch_size)
    
    # Model
    model = HybridViT(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, 
                                weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    #Cosine Scheduler
    scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=5,                # you can tune this
    total_epochs=config.epochs,
    base_lr=config.lr
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate AFTER epoch
        current_lr = scheduler.step()
        
        print(f"LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        with open('/content/drive/MyDrive/vit_project/logs/hybrid_cifar10_metrics.txt', 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.2f},{test_loss:.4f},{test_acc:.2f}\n")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'experiments/hybrid_cifar10_best.pth')
            
        torch.save(
        model.state_dict(),
        f'/content/drive/MyDrive/vit_project/experiments/hybrid_cifar10_epoch_{epoch+1}.pth'
        )
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()