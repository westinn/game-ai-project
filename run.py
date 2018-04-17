import gym
import gym_rle
import argparse
import image_preprocess
import numpy as np
from dqn import DQNAgent

ROM = 'ClassicKong-v0'
EPISODES = 1280
BATCH_SIZE = 32
RENDER = True

env = gym.make(ROM)
FRAMESKIP = env.unwrapped.frameskip
state_size = 84 * 84 * 1
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

def train():
    total_batches = 0
    batch_scores = {}
    current_episode = []
    # step the first args last frame, reset, render
    for e in range(EPISODES):
        state = [preprocessor.pre_process_image(env.reset())] * FRAMESKIP
        total_reward = 0
        done = False

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

            # preprocess
            next_state = list(map(preprocessor.pre_process_image, next_state))


            agent.learn(last_state, next_action, reward, done, next_state, frame_number)

            # update reward and frame count
            frame_number += 1
            reward = reward if not done else -10
            total_reward += reward

        print("Episode: {}/{}, Score: {}, e: {}".format(e + 1, EPISODES, total_reward, '~'))
        current_episode.append(total_reward)

        if (e % BATCH_SIZE) > 0 and ((e + 1) % BATCH_SIZE) == 0:
            print("Finished batch: {}/{}".format(total_batches + 1, int(EPISODES / BATCH_SIZE)))
            print("Mean: {} | Median: {}{}".format(np.mean(current_episode), np.median(current_episode), '\n'))
            agent.save_network("batch_save_num_{}".format(total_batches))
            batch_scores[total_batches] = sum(current_episode)
            total_batches += 1
            current_episode = []

    print("\n\nFinished all batches:\n")
    for batch, episodes in batch_scores.items():
        print("| Batch#:{} | Mean:{} | Median:{} | Episodes:{} |".format(
            batch,
            np.mean(episodes), np.median(episodes),
            episodes))

def test(total_episodes=30):
    # load the h5 file containing the network weights
    print('loading network weights...')
    agent.model.load_weights("saved_networks/kong_batch_save_num_11.h5")

    rewards = []
    for _ in range(total_episodes):
        state = [preprocessor.pre_process_image(env.reset())] * FRAMESKIP
        total_reward = 0
        done = False

        # count frame number
        frame_number = 0

        while not done:
            if RENDER:
                env.render()

            # store current state
            last_state = state
            # predict best action
            next_action = agent.test_act(state)
            # act using the best action and save results
            next_state, reward, done, _ = env.step(next_action)

            # preprocess
            next_state = list(map(preprocessor.pre_process_image, next_state))

            # update reward and frame count
            frame_number += 1
            reward = reward if not done else -10
            total_reward += reward

        print('Run %d episodes'%(total_episodes))
        print('Mean:', np.mean(rewards))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=None, action='store_true', help="train the network, otherwise test it.")
    args = parser.parse_args()
    print(args.train)
    if args.train:
        train()
    else:
        test()
