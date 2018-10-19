import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# getting 28x28 images dataset of handwritten digits from 0 to 9
Mnist = tf.keras.datasets.mnist

#Loading the Datset's data
(x_train, y_train), (x_test, y_test) = Mnist.load_data()




#Normalizing the training data for better accuracy
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

#Now, builiding the model for classification
model = tf.keras.models.Sequential()

# The Input layer with Flattened Acceptance Property
model.add(tf.keras.layers.Flatten())

#Hidden Layers
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu,input_dim=784))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu,input_dim=784))
#Output Layer
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
#Model Architecture Created. Now, using the model:-
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Training the Model:-
model.fit(x_train,y_train,epochs=0)
model.load_weights('NumberRecognitionModelWeights.h5')

Predictions = model.predict([x_test])

#Storing a prediction. It is of First Test Image
TestNumber = np.argmax(Predictions[2])
print("The Third test image classifies to ", TestNumber)
plt.imshow(x_test[2])
plt.show()
