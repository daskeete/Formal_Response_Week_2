import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow import keras
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
np.set_printoptions(suppress=True, precision=20)

training_images.shape
training_labels.shape
test_images.shape

training_images = training_images / np.max(np.unique(training_images))
test_images = test_images / np.max(np.unique(test_images))
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])                                  
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)                
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
classifications[0]

np.argmax(classifications[0]) #returns indices of maximum values

plt.imshow(test_images[0])

b = list(classifications[0])
for i in range(len(classifications[0])):
    if b[i] < max(classifications[0]):
        b[i]=0.1
bars = np.unique(test_labels)
color = ['grey' if (x < max(classifications[0])) else 'blue' for x in classifications[0]]
plt.bar(bars,b,color=color)
plt.xticks(bars)
plt.tick_params(labelleft=False)

plt.show()
