# DeepParticleRobot

[**Jeremy Shen**](https://github.com/jshen04), [**Erdong Xiao**](https://github.com/ErdongXiao), [**Yuchen Liu**](https://github.com/Rtlyc), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

## Abstract
Particle  robots  are  novel  biologically-inspired robotic systems where locomotion can be achieved collectively and robustly, but not independently. While its control iscurrently limited to a hand-crafted policy for basic locomotion tasks, such a multi-robot system can be potentially controlled via Deep Reinforcement Learning (DRL) for different tasks. However, the particle robot system presents a new set of challenges for DRL differing from existing swarm robotics systems: the low dimensionality of each robot and the increased necessity of coordination between robots. We present a 2D particle robot simulator using the OpenAI Gym interface and Pymunk as the physics engine, and introduce new tasks and challenges to research the underexplored applications of DRLin the particle robot system. Moreover, we use Stable-baselines3 to provide a set of benchmarks for the tasks. Current baseline DRL algorithms show signs of achieving the tasks but are unable to reach the performance of the hand-crafted policy. Further development of DRL algorithms is necessary in order to accomplish the proposed tasks.

## [Code (GitHub)](https://github.com/ai4ce/DeepParticleRobot/) & Dependencies
The environment scripts can be found in the [gym_dpr](https://github.com/ai4ce/DeepParticleRobot/tree/main/gym_dpr) folder. The environment is developed based on the [OpenAi Gym](https://gym.openai.com/). All baseline scripts are in [script](https://github.com/ai4ce/DeepParticleRobot/tree/main/script) folder. You need to install the [Pytorch](https://pytorch.org/) to run all baseline scripts. We use [Stable baselines3](https://github.com/DLR-RM/stable-baselines3) for A2C, PPO, and DQN algorithms. 
## How to use

Our environment is developed based on the [OpenAi Gym](https://gym.openai.com/). Here is a simple test episode with randomized actions.
```
from gym_dpr.envs.viz import Visualizer
env = gym.make('dpr_single-v0', numbBots=9, visualizer=Visualizer())

obs = env.reset()
while True:
    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        break
```
