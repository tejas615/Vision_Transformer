import pandas as pd

results = {
    'Model': ['ViT', 'Hybrid ViT'],
    'Parameters (M)': [2684554/1e6, 2684554/1e6],
    'MNIST Accuracy': [98.62, 98.84],  
    'CIFAR-10 Accuracy': [48.04, 61.05],
}

df = pd.DataFrame(results)
print(df)
df.to_csv('results.csv')