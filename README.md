# SaemanGeum Rice Yield Prediction

# This research has been used satellite imagery data, weather data, and Deep Neural Network model

## Selected model 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping

# This model shows the best performance in this research
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.001)

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stop])

