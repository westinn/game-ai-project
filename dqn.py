import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam


def huber_loss(y_true, y_pred):
    import tensorflow as tf
    return tf.losses.huber_loss(y_true, y_pred)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        self.model = Sequential()
        # 1st Conv2D after inputs
        # 96x84 pixel input with 4 frames with 4 stride
        self.model.add(Conv2D(
            32,
            (8, 8),
            input_shape=(4, 96, 84),
            subsample=(4, 4), activation='relu'))
        # 2nd Conv2D after inputs
        self.model.add(Conv2D(
            64,
            (4, 4),
            subsample=(2, 2), activation='relu'))
        # 3rd Conv2D after inputs
        self.model.add(Conv2D(
            64,
            (3, 3),
            subsample=(1, 1), activation='relu'))
        # Flatten Conv
        self.model.add(Flatten())
        # Normal 512 node hidden layer
        self.model.add(Dense(512, activation='relu'))
        # Output of possible inputs to environment
        self.model.add(Dense(self.action_size))

        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, BATCH_SIZE):
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
