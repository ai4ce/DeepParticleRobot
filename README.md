# DeepParticleRobot

[**Jeremy Shen**](https://github.com/jshen04), [**Erdong Xiao**](https://github.com/ErdongXiao), [**Yuchen Liu**](https://github.com/Rtlyc), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

## Abstract
Particle  robots  are  novel  biologically-inspiredrobotic systems where locomotion can be achieved collectivelyand robustly, but not independently. While its control iscurrently limited to a hand-crafted policy for basic locomotiontasks, such a multi-robot system can be potentially controlledvia Deep Reinforcement Learning (DRL) for different tasks.However, the particle robot system presents a new set ofchallenges for DRL differing from existing swarm roboticssystems: the low dimensionality of each robot and the increasednecessity of coordination between robots. We present a 2Dparticle robot simulator using the OpenAI Gym interface andPymunk as the physics engine, and introduce new tasks andchallenges to research the underexplored applications of DRLin the particle robot system. Moreover, we use Stable-baselines3to provide a set of benchmarks for the tasks. Current baselineDRL algorithms show signs of achieving the tasks but areunable to reach the performance of the hand-crafted policy.Further development of DRL algorithms is necessary in orderto accomplish the proposed tasks.

## [Code (GitHub)](https://github.com/ai4ce/DeepParticleRobot/) & Dependencies
The environment scripts can be found in [Env](https://github.com/ai4ce/DeepParticleRobot/tree/main/Env) folder. The environment is developed based on the [OpenAi Gym](https://gym.openai.com/). All baseline scripts are in [script](https://github.com/ai4ce/DeepParticleRobot/tree/main/script) folder. You need to install the [Pytorch](https://pytorch.org/) to run all baseline scripts. We use [Stable baselines3](https://github.com/DLR-RM/stable-baselines3) for A2C, PPO, and DQN algorithms. 
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
