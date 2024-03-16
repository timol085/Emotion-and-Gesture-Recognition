import h5py
import matplotlib.pyplot as plt

# Load the model file
model_path = 'emotion_model_without_aug.h5'
with h5py.File(model_path, 'r') as file:
    # Access the model's training history
    history = file['history']
    
    # Retrieve accuracy and loss values
    accuracy = history['accuracy'][:]
    loss = history['loss'][:]
    
    # Get the validation accuracy and loss if available
    if 'val_accuracy' in history:
        val_accuracy = history['val_accuracy'][:]
        val_loss = history['val_loss'][:]

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Training Accuracy')
if 'val_accuracy' in history:
    plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
if 'val_loss' in history:
    plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
