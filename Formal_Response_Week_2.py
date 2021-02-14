import numpy as np
np.set_printoptions(suppress=True, precision=15)

#%%

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow import keras
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#%%

training_images = training_images / 255.0
test_images = test_images / 255.0


#%%

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#%%

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)

#%%

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#%%

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
i=0

#%%

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions[i]

#%%

plt.title('My Image Plot')
plt.imshow(test_images[i])
plt.savefig('number_')
plt.show()


#%%

#bars = np.unique(test_labels)
color = ['grey' if (x < max(predictions[i])) else 'blue' for x in predictions[i]]
plt.bar(class_names,predictions[i],color=color)
plt.xticks(class_names)
#plt.tick_params(labelleft=False)
plt.title('My Probability Graph')
plt.savefig('probability_graph')

plt.show()

#%%

predictions[i]


#%%

np.argmax(predictions[i])
