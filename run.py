import gym
import gym_rle
import image_preprocess
import numpy as np
from dqn import DQNAgent

ROM = 'ClassicKong-v0'
EPISODES = 1280
BATCH_SIZE = 32
RENDER = True

env = gym.make(ROM)
FRAMESKIP = env.unwrapped.frameskip
state_size = 96 * 84 * 1
action_size = env.action_space.n

print('\n=====================================================================================')
print('Game ROM: {}'.format(ROM))
print('State Size: {}'.format(state_size))
print('Action Size: {}'.format(action_size))
print('\n# of Episodes: {}'.format(EPISODES))
print('# of Batches: {}'.format(int(EPISODES / BATCH_SIZE)))
print('Batch Size: {}'.format(BATCH_SIZE))
print('=====================================================================================\n')

agent = DQNAgent(state_size, action_size)
preprocessor = image_preprocess.ImagePreprocessors()

total_batches = 0
batch_scores = {}
current_episode = []
# step the first args last frame, reset, render
for e in range(EPISODES):
    state = preprocessor.pre_process_image(env.reset())
    total_reward = 0
    done = False

    # TODO
    # currently total reward is reset each episode, is that correct?

    # count frame number
    frame_number = 0

    while not done:
        if RENDER:
            env.render()

        # store current state
        last_state = state
        # predict best action
        next_action = agent.act(state)
        # act using the best action and save results
        next_state, reward, done, _ = env.step(next_action)

        # TODO
        # we may not need preprocessing

        # take the last frame of 4, and preprocess that one as the next state
        next_state = preprocessor.pre_process_image(next_state[-1])
        agent.learn(last_state, next_action, reward, done, next_state, frame_number)

        # update reward and frame count
        frame_number += 1
        reward = reward if not done else -10
        total_reward += reward


    print("Episode: {}/{}, Score: {}, e: {}".format(e + 1, EPISODES, total_reward, '~'))
    current_episode.append(total_reward)

    # if len(agent.memory) > BATCH_SIZE:
    #     agent.replay(BATCH_SIZE)

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
