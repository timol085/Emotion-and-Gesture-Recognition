
# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
import tensorflow as tf
# from tensorflow.keras.layers.experimental import preprocessing
from keras.callbacks import EarlyStopping

# Initialize image data generator with rescaling

# Define some parameters
batch_size = 64
img_height = 48
img_width = 48

# Create the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    color_mode='grayscale',
    validation_split=0.2,
    subset='training',
    seed=42
)

# Create the validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    color_mode='grayscale',
    validation_split=0.2,
    subset='validation',
    seed=42
)

# Data Augmentation Layer
# data_augmentation = Sequential([
#     preprocessing.RandomFlip('horizontal'),
#     preprocessing.RandomZoom(0.2),
#     preprocessing.RandomRotation(0.2),
# ])

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply data augmentation and normalization to the datasets
# augmented_train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
# augmented_train_dataset = augmented_train_dataset.map(lambda x, y: (normalization_layer(x), y))
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))

# Concatenate the augmented dataset with the original dataset
# train_dataset = train_dataset.concatenate(augmented_train_dataset)

validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Train the neural network/model
emotion_model_info = emotion_model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=[early_stopping]
)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model_without_aug.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model_without_aug.h5')

