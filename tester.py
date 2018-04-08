import gym
import gym_rle
import numpy as np

ROM = 'ClassicKong-v0'
EPISODES = 1280
BATCH_SIZE = 2
RENDER = True

print('\n=====================================================================================')
print('Game ROM: {}'.format(ROM))
print('# of Episodes: {}'.format(EPISODES))
print('# of Batches: {}'.format(EPISODES/BATCH_SIZE))
print('Batch Size: {}'.format(BATCH_SIZE))
print('=====================================================================================')

print('\nTODO: Current logging records last episode score rather than proper mean/median.\n\n')

env = gym.make(ROM)
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
        if RENDER:
            env.render()
        action = 0
        next_state, reward, done, _ = env.step(env.action_space.sample())
        reward = reward if not done else -10

        state = next_state
        total_reward += reward
        if done:
            print("episode: {}/{}, score: {}, e: {}".format(e+1, EPISODES, total_reward, '~'))
            current_episode.append(total_reward)
    
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
