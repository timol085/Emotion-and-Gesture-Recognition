import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the class labels and corresponding emotions
class_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Initialize a list to hold images and labels
images = []
labels = []

# Iterate through the class labels
for label, emotion in class_labels.items():
    # Define the path to the directory containing images for this emotion
    image_dir = os.path.join('data', 'test', emotion)

    # Get a list of all image files in the directory
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # Randomly select 7 images from the list
    selected_images = np.random.choice(image_files, 7, replace=False)

    # Load and append the images along with their labels
    for image_path in selected_images:
        image = cv2.imread(image_path)
        images.append(image)
        labels.append(emotion)

# Create a grid of 7x7 subplots
fig, axes = plt.subplots(7, 7, figsize=(15, 15))

# Iterate through the images and labels, and display them in the subplots
for i, (image, label) in enumerate(zip(images, labels)):
    row = i % 7  # Switched row and column indexing
    col = i // 7  # Switched row and column indexing
    ax = axes[row, col]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

# Set column titles with increased spacing
for col, emotion in enumerate(class_labels.values()):
    axes[0, col].set_title(emotion, pad=30)  # Adjusted pad value (default is 6.0)


# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# Show the plot
plt.show()
