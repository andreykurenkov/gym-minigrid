import gym
import gym_minigrid
from matplotlib import pyplot as plt
import random

for i in range(10):

  env = gym.make('MiniGrid-FourRoomsMemory-v0')

  env.reset()

  for i in range(10000):
    action = env._rand_int(0,5)
    print(action)
    obs, reward, done, info = env.step(action)
    img = env.render('rgb_array')

    plt.imshow(img)
    plt.show()


