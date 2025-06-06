from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#dataset
X = np.random.rand(100, 3)
y = np.random.rand(100, 1)

#simple_model
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10)

model.save("model.h5")
