import matplotlib.pyplot as plt
import os


data_folder = 'data/train/'

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

height = []
classifier = []

# Loop through each emotion folder
for emotion in emotions:
   
    emotion_path = os.path.join(data_folder, emotion)
    
    num_images = len(os.listdir(emotion_path))
    
    height.append(num_images)
    
    classifier.append(emotion)


positions = range(len(height))
plt.bar(positions, height)


plt.xticks(positions, classifier)

plt.title("Training data")
plt.xlabel("Emotions")
plt.ylabel("Number of images")

plt.show()
