import keras

frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
actions_input = keras.layers.Input((self.n_actions,), name='filter')

conv_1 = keras.layers.convolutional.Convolution2D(
    32, 8, 8, subsample=(4, 4), activation='relu'
)(keras.layers.Lambda(lambda x: x / 255.0)(frames_input))
conv_2 = keras.layers.convolutional.Convolution2D(
    64, 4, 4, subsample=(2, 2), activation='relu'
)(conv_1)
conv_3 = keras.layers.convolutional.Convolution2D(
    64, 3, 3, subsample=(1, 1), activation='relu'
)(conv_2)
conv_flattened = keras.layers.core.Flatten()(conv_3)
hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
output = keras.layers.Dense(self.n_actions)(hidden)
filtered_output = keras.layers.merge([output, actions_input], mode='mul')

self.model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
optimizer = optimizer=keras.optimizers.RMSprop(
    lr=0.00025, rho=0.95, epsilon=0.01
)
self.model.compile(
    optimizer, loss=huber_loss)