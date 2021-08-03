import gym
import gym_minigrid
from matplotlib import pyplot as plt

for i in range(10):

  env = gym.make('MiniGrid-FourRoomsMemory-v0')
  env.reset()


  for i in range(100):
    action = env.actions.right

    obs, reward, done, info = env.step(action)

    img = env.render('rgb_array')

    plt.imshow(img)
    plt.show()


