import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import time 

# Create regression data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
print (X, y)
time.sleep(50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a regression model using TensorFlow
def build_regression_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Model training
regression_model = build_regression_model(X_train.shape[1])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = regression_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                               epochs=100, callbacks=[early_stopping], batch_size=32)

# Model evaluation
test_loss, test_mae = regression_model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

# Save the model
regression_model.save('regression_model.h5')