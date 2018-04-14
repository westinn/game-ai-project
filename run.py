import gym
import gym_rle
import image_preprocess
from dqn import DQNAgent

ROM = 'ClassicKong-v0'
EPISODES = 1280
BATCH_SIZE = 32
RENDER = True

env = gym.make(ROM)
FRAMESKIP = env.unwrapped.frameskip
state_size = (
    env.observation_space.shape[0] * env.observation_space.shape[1])
action_size = env.action_space.n

print('\n=====================================================================================')
print('Game ROM: {}'.format(ROM))
print('State Size: {}'.format(state_size))
print('Action Size: {}'.format(action_size))
print('\n# of Episodes: {}'.format(EPISODES))
print('# of Batches: {}'.format(EPISODES / BATCH_SIZE))
print('Batch Size: {}'.format(BATCH_SIZE))
print('=====================================================================================\n')

print('\nTODO: Current logging records last episode score rather than proper mean/median.\n\n')

agent = DQNAgent(state_size, action_size)
total_batches = 0
batch_scores = {}

for e in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if RENDER:
            env.render()
        # predict best, next action
        next_action = agent.act(state)
        # act using the best action and save results
        next_state, reward, done, _ = env.step(next_action)
        reward = reward if not done else -10
        # save result of taking best action on current state into memory
        agent.remember(state, next_action, reward, next_state, done)
        # iterate forward into next state
        state = next_state
        total_reward += reward
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(
                e, EPISODES, total_reward, agent.epsilon))
            break
    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)
