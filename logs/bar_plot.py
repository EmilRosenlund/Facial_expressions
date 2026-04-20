import matplotlib.pyplot as plt

# 1. Data Setup
iterations = ['Iter 1', 'Iter 2', 'Iter 3', 'Iter 4', 'Iter 5', 'Iter 6', 'Iter 7']
accuracies = [0.5362, 0.5402, 0.5507, 0.5811, 0.6509, 0.6903, 0.6974]

# Define colors: Blue for ResNet50, Green for InceptionResNetV1
colors = ['#3498db'] * 4 + ['#2ecc71'] * 3

# 2. Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(iterations, accuracies, color=colors, edgecolor='black', alpha=0.8)

# 3. Add data labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

# 4. Customizing the chart
ax.set_title('Model Accuracy Comparison', fontsize=14, pad=15)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Experiment Iterations', fontsize=12)
ax.set_ylim(0, 0.8)  # Space for labels
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 5. Manual Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#3498db', lw=4, label='ResNet50'),
    Line2D([0], [0], color='#2ecc71', lw=4, label='InceptionResNetV1')
]
ax.legend(handles=legend_elements, loc='upper left', title="Architectures")


# 6. Final Layout
plt.tight_layout()
plt.show()
