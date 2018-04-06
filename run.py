import gym
import gym_rle
import image_preprocess
from dqn import DQNAgent


env = gym.make('ClassicKong-v0')
EPISODES = 1280
BATCH_SIZE = 32
FRAMES_IN_STATE = env.unwrapped.frameskip
RENDER = True

state_size = (
    env.observation_space.shape[0] * 
    env.observation_space.shape[1] * FRAMES_IN_STATE)
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

for e in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if RENDER:
            env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(
                e, EPISODES, total_reward, agent.epsilon))
            break
    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)
