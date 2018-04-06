import gym
import gym_rle
import ipdb
from dqn import DQNAgent
from image_preprocess import image_preprocess

env = gym.make('ClassicKong-v0')
ipdb.set_trace()
state_size = (env.observation_space.shape[0] * env.observation_space.shape[1] * env.unwrapped.frameskip)
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

for e in range(EPISODES):
    state = env.reset()
    done = False

    while not done:
        if RENDER:
            env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
            break
    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)
