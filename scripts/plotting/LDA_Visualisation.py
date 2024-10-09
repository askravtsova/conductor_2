import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X_avg contains averaged features and y contains the labels
# Create a DataFrame for easy visualization
df = pd.DataFrame(X_avg)
df['label'] = y

# Plot distribution of a particular feature (e.g., velocity or acceleration column)
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y=df[0], data=df)  # Assuming column 0 is some key feature (velocity/acceleration)
plt.title('Feature Distribution Across Gestures')
plt.xlabel('Gesture Type')
plt.ylabel('Feature Value')
plt.show()

# Create a correlation matrix for features in X_avg
corr_matrix = pd.DataFrame(X_avg).corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()
