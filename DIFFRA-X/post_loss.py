import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('outputs/metrics.csv')
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(df['Step'], df['Loss'], color='tab:red', label='Loss')
    ax1.set_yscale('log') # Loss 通常看对数坐标更清晰

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sigma', color='tab:blue')
    ax2.plot(df['Step'], df['Sigma'], color='tab:blue', linestyle='--', label='Sigma')

    plt.title('DIFFRA-X Training Progress')
    plt.savefig('outputs/training_curve.png')
    print("Curve saved to outputs/training_curve.png")
except Exception as e:
    print(f"Waiting for data... {e}")