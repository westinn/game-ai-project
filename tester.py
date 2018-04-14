import gym
import gym_rle
import image_preprocess
import numpy as np

ROM = 'ClassicKong-v0'
# EPISODES = 1280
# BATCH_SIZE = 32
EPISODES = 18
BATCH_SIZE = 3
RENDER = True

env = gym.make(ROM)
preprocessor = image_preprocess.ImagePreprocessors()
env.reset()
env.render(RENDER)


print('\n=====================================================================================')
print('Game ROM: {}'.format(ROM))
print('# of Episodes: {}'.format(EPISODES))
print('# of Batches: {}'.format(int(EPISODES / BATCH_SIZE)))
print('Batch Size: {}'.format(BATCH_SIZE))
print('=====================================================================================\n')


total_batches = 0
batch_scores = {}
current_episode = []
# an episode is a life
for e in range(EPISODES):
    state = env.reset()
    state = preprocessor.pre_process_image(state)
    done = False
    total_reward = 0

    while not done:
        if RENDER:
            env.render()
        action = 0
        next_state, reward, done, _ = env.step(env.action_space.sample())
        next_state = preprocessor.pre_process_image(next_state[-1])

        reward = reward if not done else -10

        state = next_state
        total_reward += reward
    print("Episode: {}/{}, Score: {}, e: {}".format(e + 1, EPISODES, total_reward, '~'))
    current_episode.append(total_reward)

    if (e % BATCH_SIZE) > 0 and ((e + 1) % BATCH_SIZE) == 0:
        print("Finished batch: {}/{}".format(total_batches + 1, int(EPISODES / BATCH_SIZE)))
        print("Mean: {} | Median: {}{}".format(np.mean(current_episode), np.median(current_episode), '\n'))
        batch_scores[total_batches] = sum(current_episode)
        total_batches += 1
        current_episode = []


print("\n\nFinished all batches\n")
for batch, episodes in batch_scores.items():
    print("| Batch#:{} | Mean:{} | Median:{} | Episodes:{} |".format(
        batch,
        np.mean(episodes), np.median(episodes),
        episodes))
print('\n')
