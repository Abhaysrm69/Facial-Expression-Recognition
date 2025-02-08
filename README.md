# Facial Expression Recognition Using CNN

## Overview
This project implements a deep learning model using Convolutional Neural Networks (CNN) to recognize facial expressions from images. The dataset contains seven classes of expressions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Dataset
- Dataset: Face Expression Recognition Dataset
- Classes: 7 (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- Image Size: 48x48 pixels

## Installation
Ensure you have the required libraries installed:
```bash
pip install numpy pandas seaborn matplotlib keras tensorflow
```

## Importing Libraries
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
```

## Displaying Sample Images
```python
picture_size = 48
folder_path = "../input/face-expression-recognition-dataset/images/"
expression = 'disgust'

plt.figure(figsize=(12,12))
for i in range(1, 10):
    plt.subplot(3,3,i)
    img = load_img(folder_path+"train/"+expression+"/"+
                  os.listdir(folder_path + "train/" + expression)[i], target_size=(picture_size, picture_size))
    plt.imshow(img)   
plt.show()
```

## Data Preparation
```python
batch_size = 128

datagen_train = ImageDataGenerator()
datagen_val = ImageDataGenerator()

train_set = datagen_train.flow_from_directory(folder_path+"train", target_size=(picture_size, picture_size), color_mode="grayscale", batch_size=batch_size, class_mode='categorical', shuffle=True)
test_set = datagen_val.flow_from_directory(folder_path+"validation", target_size=(picture_size, picture_size), color_mode="grayscale", batch_size=batch_size, class_mode='categorical', shuffle=False)
```

## Model Architecture
```python
model = Sequential()

# CNN Layers
model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Fully Connected Layers
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

# Compilation
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

## Training the Model
```python
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("./model.h5", monitor='val_acc', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_delta=0.0001)

callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

epochs = 48

history = model.fit_generator(generator=train_set, steps_per_epoch=train_set.n//train_set.batch_size, epochs=epochs, validation_data=test_set, validation_steps=test_set.n//test_set.batch_size, callbacks=callbacks_list)
```

## Performance Visualization
```python
plt.style.use('dark_background')

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
```

## Conclusion
This model achieves a decent accuracy for facial expression recognition and can be further improved with data augmentation, hyperparameter tuning, or transfer learning.

## Future Work
- Experiment with different architectures like ResNet or MobileNet.
- Implement real-time recognition using OpenCV.
- Optimize performance for mobile or edge devices.

## Acknowledgments
This project uses the Face Expression Recognition dataset from Kaggle.




Dataset link: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
