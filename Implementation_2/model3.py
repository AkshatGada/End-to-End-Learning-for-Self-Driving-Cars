import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, TimeDistributed, LSTM
from tensorflow.keras.models import Model

time_steps = 10  # You can change this to however many frames per sequence you need.
img_height, img_width, channels = 66, 200, 3

inp = Input(shape=(time_steps, img_height, img_width, channels))


td_conv1 = TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))(inp)
td_conv2 = TimeDistributed(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))(td_conv1)
td_conv3 = TimeDistributed(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))(td_conv2)
td_conv4 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(td_conv3)
td_conv5 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(td_conv4)

# Flatten the CNN output for each time step.
td_flat = TimeDistributed(Flatten())(td_conv5)

# Optional: add fully connected (FC) layers to further process the per-frame features.
td_fc1 = TimeDistributed(Dense(100, activation='relu'))(td_flat)
td_fc2 = TimeDistributed(Dense(50, activation='relu'))(td_fc1)
td_fc3 = TimeDistributed(Dense(10, activation='relu'))(td_fc2)

# --- LSTM layer ---
# The LSTM now processes the sequence of features (one vector per frame).
# Here, return_sequences=False means we only care about the final output after processing the sequence.
lstm_out = LSTM(50, activation='tanh', return_sequences=False)(td_fc3)

y = Dense(1)(lstm_out)

model = Model(inputs=inp, outputs=y)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

print("CMM model with LSTM successfully created.")
model.summary()
