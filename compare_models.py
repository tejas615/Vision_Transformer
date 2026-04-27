import torch
from models.vit import VisionTransformer
from models.hybrid import HybridViT
from utils.data import get_cifar10_loaders
from configs.cifar_config import CIFAR10Config

def evaluate(model, dataloader, device):
    model.eval()  # set model to evaluation mode
    
    correct = 0
    total = 0
    
    with torch.no_grad():  # no gradient calculation
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)  # forward pass
            
            _, predicted = torch.max(outputs, 1)  # get predicted class
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def compare_models():
    config = CIFAR10Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, test_loader = get_cifar10_loaders(config.batch_size)
    
    # Load ViT
    vit = VisionTransformer(config).to(device)
    vit.load_state_dict(torch.load('experiments/vit_cifar10_best.pth'))
    
    # Load Hybrid
    hybrid = HybridViT(config).to(device)
    hybrid.load_state_dict(torch.load('experiments/hybrid_cifar10_best.pth'))
    
    # Evaluate
    vit_acc = evaluate(vit, test_loader, device)
    hybrid_acc = evaluate(hybrid, test_loader, device)
    
    print(f"ViT Accuracy: {vit_acc:.2f}%")
    print(f"Hybrid ViT Accuracy: {hybrid_acc:.2f}%")
    print(f"Improvement: {hybrid_acc - vit_acc:.2f}%")
    
    # Compare parameters
    vit_params = sum(p.numel() for p in vit.parameters())
    hybrid_params = sum(p.numel() for p in hybrid.parameters())
    print(f"\nViT Parameters: {vit_params:,}")
    print(f"Hybrid Parameters: {hybrid_params:,}")

if __name__ == '__main__':
    compare_models()