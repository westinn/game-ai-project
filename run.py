import gym
import gym_rle
import image_preprocess
import numpy as np
from dqn import DQNAgent

ROM = 'ClassicKong-v0'
EPISODES = 1280
BATCH_SIZE = 32
RENDER = False

env = gym.make(ROM)
FRAMESKIP = env.unwrapped.frameskip
# 4 frames of 84x84 gray-scaled + down-sampled
state_size = (FRAMESKIP, 1, 84, 84)
action_size = env.action_space.n

print('\n=====================================================================================')
print('Game ROM: {}'.format(ROM))
print('State Size: {}'.format(state_size))
print('Action Size: {}'.format(action_size))
print('\n# of Episodes: {}'.format(EPISODES))
print('# of Batches: {}'.format(int(EPISODES / BATCH_SIZE)))
print('Batch Size: {}'.format(BATCH_SIZE))
print('=====================================================================================\n')

agent = DQNAgent(action_size, state_size)
preprocessor = image_preprocess.ImagePreprocessors()

total_batches = 0
batch_scores = {}
current_episode = []
# step the first args last frame, reset, render
for e in range(EPISODES):
    state = [preprocessor.pre_process_image(env.reset())] * state_size[0]
    total_reward = 0
    done = False

    while not done:
        if RENDER:
            env.render()
        # predict best action
        next_action = agent.act(state)
        # act using the best action and save results
        next_state, reward, done, _ = env.step(next_action)
        # take the last frame of 4, and preprocess that one as the next state
        next_state = preprocessor.pre_process_image(next_state[-1])
        reward = reward if not done else -10
        # save result of taking best action on current state into memory
        agent.remember(state, next_action, reward, next_state, done)
        # iterate forward into next state
        state = next_state
        total_reward += reward

    print("Episode: {}/{}, Score: {}, e: {}".format(e + 1, EPISODES, total_reward, '~'))
    current_episode.append(total_reward)

    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)

    if (e % BATCH_SIZE) > 0 and ((e + 1) % BATCH_SIZE) == 0:
        print("Finished batch: {}/{}".format(total_batches + 1, int(EPISODES / BATCH_SIZE)))
        print("Mean: {} | Median: {}{}".format(np.mean(current_episode), np.median(current_episode), '\n'))
        batch_scores[total_batches] = sum(current_episode)
        total_batches += 1
        current_episode = []

print("\n\nFinished all batches:\n")
for batch, episodes in batch_scores.items():
    print("| Batch#:{} | Mean:{} | Median:{} | Episodes:{} |".format(
        batch,
        np.mean(episodes), np.median(episodes),
        episodes))
