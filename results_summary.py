# results_summary.py
import pandas as pd

results = {
    'Model': ['ViT', 'Hybrid ViT'],
    'Parameters (M)': [vit_params/1e6, hybrid_params/1e6],
    'MNIST Accuracy': [98.2, 98.5],  # Your actual results
    'CIFAR-10 Accuracy': [vit_acc, hybrid_acc],
    'Training Time (hours)': [time_vit, time_hybrid]
}

df = pd.DataFrame(results)
print(df)
df.to_csv('results.csv')