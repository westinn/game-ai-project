import gym
import gym_rle
import numpy as np


EPISODES = 1280
# BATCH_SIZE = 32
BATCH_SIZE = 2
RENDER = True

env = gym.make('ClassicKong-v0')
env.reset()
env.render(RENDER)


total_batches = 0
batch_scores = {}
# an episode is a life
for e in range(EPISODES):
    state = env.reset()
    done = False
    current_episode = []
    total_reward = 0

    while not done:
        print(current_episode)
        if RENDER:
            env.render()
        action = 0
        next_state, reward, done, _ = env.step(env.action_space.sample())
        reward = reward if not done else -10

        state = next_state
        total_reward += reward
        if done:
            print("episode: {}/{}, score: {}, e: {}".format(e+1, EPISODES, total_reward, '~'))
            #import ipdb; ipdb.set_trace();
            current_episode.append(total_reward)
            #import ipdb; ipdb.set_trace();
    
    if (e % BATCH_SIZE) > 0 and ((e+1) % BATCH_SIZE) == 0:
        print("Finished batch number: {}/{}".format(
            total_batches+1, (EPISODES/BATCH_SIZE)))
        print("Mean: {} | Median: {}{}".format(
            np.mean(current_episode), np.median(current_episode), 
            '\n'))
        batch_scores[total_batches] = current_episode
        total_batches += 1

print("\n\n\nFinished all batches\n")
for batch, episodes in batch_scores.items():
    print("| Batch#:{} | Mean:{} | Meadian:{} | Episodes:{} |".format(
        batch, 
        np.mean(episodes), np.median(episodes), 
        episodes))
