import random
import numpy
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Input, LeakyReLU, Multiply, Lambda
from keras.optimizers import Adam, RMSprop
import keras.backend as K


def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)


def list2np(in_list):
    return numpy.float32(numpy.array(in_list))


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # logging
        self.total_loss = 0.0

        # define the input shape
        self.width = 84
        self.height = 84
        self.state_length = 4

        self.batch_size = 32

        self.dummy_input = numpy.zeros((1, self.action_size))
        self.dummy_batch = numpy.zeros((self.batch_size, self.action_size))

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # TODO
        # double check this
        self.opt = RMSprop(lr=self.learning_rate, rho=0.99, decay=0.0, epsilon=1e-8)

        # Can hold up to 2000 states at a time
        self.deque_len = 2000
        self.memory = deque(maxlen=self.deque_len + 1)

        # testing variables
        self.initial_replay_size = 10000
        self.target_update_interval = 1000

        # build the network
        self.model = self._build_model()
        self.target_network = self._build_model()
        self.target_network.set_weights(self.model.get_weights())

    # Neural Net for Deep-Q learning Model
    def _build_model(self):
        # 1st Conv2D after inputs
        # 96x84 pixel input with 1 frame with 4 stride

        input_shape = Input(shape=(self.width, self.height, self.state_length))
        action = Input(shape=(self.action_size,))

        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_shape)
        conv2 = Conv2D(64, (4, 4), strides=(4, 4), activation='relu')(conv1)
        conv3 = Conv2D(64, (4, 4), strides=(4, 4), activation='relu')(conv2)

        flat = Flatten()(conv3)
        hidden = Dense(512)(flat)
        lrelu = LeakyReLU()(hidden)
        q_value_prediction = Dense(self.action_size)(lrelu)

        select_q_value_of_action = Multiply()([q_value_prediction, action])

        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True),
                                output_shape=lambda_out_shape)(select_q_value_of_action)

        model = Model(inputs=[input_shape, action], outputs=[q_value_prediction, target_q_value])

        model.compile(loss=['mse', 'mse'], loss_weights=[0.0, 1.0], optimizer=self.opt)

        return model

    def learn(self, last_state, action, reward, next_state, done, frame_number):
        # save state
        self.memory.append((last_state, action, reward, next_state, done))

        if len(self.memory) > self.deque_len:
            self.memory.popleft()

        if frame_number >= self.initial_replay_size:
            if frame_number % 4 == 0:
                self.train()

            if frame_number % self.target_update_interval == 0:
                self.target_network.set_weights(self.model.get_weights())

            save_interval = 300000
            if frame_number % save_interval == 0:
                # TODO
                print('save network')

        # return the max q
        # np.max(self.model.predict([np.expand_dims(last_state, axis=0), self.dummy_input])[0])
        return

    def act(self, state):
        if numpy.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return numpy.argmax(act_values[0])  # returns action

    def train(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        y_batch = []

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        done_batch = numpy.array(done_batch) + 0
        # Q value from target network
        print('Prediction time')
        print(list2np(next_state_batch))
        target_q_values_batch = self.target_network.predict([list2np(next_state_batch), self.dummy_batch])[0]

        y_batch = reward_batch + (1 - done_batch) * self.gamma * numpy.max(target_q_values_batch, axis=-1)

        a_one_hot = numpy.zeros((self.batch_size, self.action_size))

        for i, ac in enumerate(action_batch):
            a_one_hot[i, ac] = 1.0

        loss = self.model.train_on_batch([list2np(state_batch), a_one_hot], [self.dummy_batch, y_batch])
        self.total_loss += loss[1]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)