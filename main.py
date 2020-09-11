import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import discord
import requests

dataFolder = ".\\Data"
classes = os.listdir(dataFolder)

discordToken = open("discordToken.txt").read()

batchSize = 32
imgHeight = 180
imgWidth = 180

trainDataSet = tf.keras.preprocessing.image_dataset_from_directory(
    dataFolder,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(imgHeight, imgWidth),
    batch_size=batchSize)

valDataSet = tf.keras.preprocessing.image_dataset_from_directory(
    dataFolder,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(imgHeight, imgWidth),
    batch_size=batchSize)

classNames = trainDataSet.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE
trainDataSet = trainDataSet.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valDataSet = valDataSet.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(classes)

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(imgHeight, imgWidth, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  trainDataSet,
  validation_data=valDataSet,
  epochs=epochs
)

acc = history.history['accuracy']
valAcc = history.history['val_accuracy']

loss=history.history['loss']
valLoss=history.history['val_loss']

epochsRange = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochsRange, acc, label='Training Accuracy')
plt.plot(epochsRange, valAcc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochsRange, loss, label='Training Loss')
plt.plot(epochsRange, valLoss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Discord Stuff

client = discord.Client()

@client.event
async def on_message(message):
  if message.content.startswith("!identify"):
      if message.attachments:
          for attachment in message.attachments:
              imgData = requests.get(attachment.url).content
              with open("identifyImage.png", "wb") as handler:
                  handler.write(imgData)

              img = keras.preprocessing.image.load_img(
                  ".\\identifyImage.png", target_size=(imgHeight, imgWidth)
              )
              img_array = keras.preprocessing.image.img_to_array(img)
              img_array = tf.expand_dims(img_array, 0)

              predictions = model.predict(img_array)
              score = tf.nn.softmax(predictions[0])

              await message.channel.send(
                  "This image most likely belongs to {} with a {:.2f} percent confidence."
                  .format(classNames[np.argmax(score)], 100 * np.max(score))
              )
              os.remove(".\\identifyImage.png")


@client.event
async def on_ready():
  print("Discord Bot Ready")

client.run(discordToken)

# model.save(".\\")
