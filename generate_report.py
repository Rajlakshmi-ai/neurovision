import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Read your local benchmark results
csv_path = 'model_comparison/comparison.csv'
if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit()

df = pd.read_csv(csv_path)

# Safely find the exact name of the 'model' column
model_col = [col for col in df.columns if col.lower() == 'model'][0]

# 2. Generate and save the Graph (Image)
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Filter out non-metric columns
metrics_to_plot = [col for col in df.columns if col.lower() not in ['model', 'epochs_trained', 'weights']]

# Restructure data for Seaborn
df_melted = df.melt(id_vars=model_col, 
                    value_vars=metrics_to_plot, 
                    var_name='Metric', 
                    value_name='Score')

# Draw the bar chart using the correct model column for colors (hue)
sns.barplot(data=df_melted, x='Metric', y='Score', hue=model_col)

plt.title('NeuroVision Model Comparison')
plt.ylim(0, 1.0)
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

image_path = 'model_comparison_graph.png'
plt.savefig(image_path, dpi=300)
print(f"✅ Graph image saved successfully to: {image_path}")

# 3. Generate the Document (.doc)
doc_path = 'model_comparison_report.doc'
with open(doc_path, 'w') as f:
    f.write("=========================================\n")
    f.write("   NEUROVISION MODEL COMPARISON REPORT   \n")
    f.write("=========================================\n\n")
    f.write("OVERVIEW:\n")
    f.write("This document compares the performance of ResNet50, EfficientNetB0, and MobileNetV2.\n\n")
    f.write("METRICS SUMMARY:\n")
    f.write(df.to_string(index=False))
    f.write("\n\nCONCLUSION:\n")
    
    auc_col = [col for col in df.columns if 'auc' in col.lower()][0]
    best_model = df.loc[df[auc_col].idxmax()][model_col]
    
    f.write(f"Based on the Area Under the Curve (AUC), the best performing architecture is {best_model}.\n")

print(f"✅ Document report saved successfully to: {doc_path}")