import pandas as pd
import matplotlib.pyplot as plt


# Load CSV into a pandas dataframe
history_df = pd.read_csv('lstm_model_history_all_metrics.csv')


# Display the first few rows to inspect the data
#print(history_df.head())

# Plot accuracy over epochs
plt.plot(history_df['accuracy'], label='Training Accuracy')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss over epochs
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

