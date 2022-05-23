# DeepParticleRobot

[**Jeremy Shen**](https://github.com/jshen04), [**Erdong Xiao**](https://github.com/ErdongXiao), [**Yuchen Liu**](https://github.com/Rtlyc), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

![Overview](https://raw.githubusercontent.com/ai4ce/DeepParticleRobot/main/docs/figs/environment_overview.jpg)

## Abstract
Particle  robots  are  novel  biologically-inspired robotic systems where locomotion can be achieved collectively and robustly, but not independently. While its control iscurrently limited to a hand-crafted policy for basic locomotion tasks, such a multi-robot system can be potentially controlled via Deep Reinforcement Learning (DRL) for different tasks. However, the particle robot system presents a new set of challenges for DRL differing from existing swarm robotics systems: the low dimensionality of each robot and the increased necessity of coordination between robots. We present a 2D particle robot simulator using the OpenAI Gym interface and Pymunk as the physics engine, and introduce new tasks and challenges to research the underexplored applications of DRLin the particle robot system. Moreover, we use Stable-baselines3 to provide a set of benchmarks for the tasks. Current baseline DRL algorithms show signs of achieving the tasks but are unable to reach the performance of the hand-crafted policy. Further development of DRL algorithms is necessary in order to accomplish the proposed tasks.

## [Code (GitHub)](https://github.com/ai4ce/DeepParticleRobot/) & Dependencies
The environment scripts can be found in the [gym_dpr](https://github.com/ai4ce/DeepParticleRobot/tree/main/gym_dpr) folder. The environment is developed based on the [OpenAi Gym](https://gym.openai.com/). All baseline scripts are in [script](https://github.com/ai4ce/DeepParticleRobot/tree/main/script) folder. You need to install the [Pytorch](https://pytorch.org/) to run all baseline scripts. We use [Stable baselines3](https://github.com/DLR-RM/stable-baselines3) for A2C, PPO, and DQN algorithms. 

## How to use
Our environment is developed with [OpenAi Gym](https://gym.openai.com/). Here is a sample simple navigation episode controlled by the handcrafted wave policy.
```
import math
import gym

from gym_dpr.envs.viz import Visualizer
from gym_dpr.envs.DPR_ParticleRobot import CircularBot
from gym_dpr.envs.DPR_SuperAgent import SuperCircularBot
from gym_dpr.envs.DPR_World import World

env = gym.make('dpr_single-v0',
               numBots=9, worldClass=World, botClass=CircularBot, superBotClass=SuperCircularBot,
               discreteActionSpace=False, continuousAction=False,
               goalFrame=True,
               rewardFunc="piecewise",
               randomSeed=0,
               fixedStart=False, fixedGoal=True,
               fixedStartCoords=None, fixedGoalCoords=(0, 0),
               polarStartCoords=False, polarGoalCoords=False,
               transformRectStart=(0, 0), transformRectGoal=(0, 0),
               xLower=-1000, xUpper=1000, yLower=-1000, yUpper=1000,
               radiusLower=450, radiusUpper=550, angleLower=0, angleUpper=2 * math.pi,
               numDead=0, deadIxs=None,
               gate=False, gateSize=150,
               manipulationTask=False, objectType="Ball", objectPos=None, initializeObjectTangent=True, objectDims=[100, 30],
               visualizer=Visualizer(), recordInfo=True)

obs = env.reset()
while True:
    totalSteps, actions = env.wavePolicy()     # hand crafted wave policy
    for i in range(totalSteps):
        for _ in range(10):
            action = actions[i]
            env.render()
            obs, reward, done, info = env.step(action)
    if done:
        break
env.close()
```

## Project Structure
```
   gym_dpr
   └── envs
       ├── DPR_ParticleRobot.py
       ├── DPR_SuperAgent.py
       ├── DPR_World.py
       ├── dpr_multi_env.py
       ├── dpr_single_env.py
       └── viz.py
```
- [DPR_ParticleRobot](https://github.com/ai4ce/DeepParticleRobot/blob/main/gym_dpr/envs/DPR_ParticleRobot.py): Defines the Particle Robot class
- [DPR_SuperAgent](https://github.com/ai4ce/DeepParticleRobot/blob/main/gym_dpr/envs/DPR_SuperAgent.py): Creates the structure for particle robot superagent (centralized control and fully observable)
- [DPR_World](https://github.com/ai4ce/DeepParticleRobot/blob/main/gym_dpr/envs/DPR_World.py): Creates the particle robot simulation space
- [dpr_multi_env](https://github.com/ai4ce/DeepParticleRobot/blob/main/gym_dpr/envs/dpr_multi_env.py): OpenAI Gym environment for decentralized particle robot control
- [dpr_single_env](https://github.com/ai4ce/DeepParticleRobot/blob/main/gym_dpr/envs/dpr_single_env.py): OpenAI Gym environment for centralized particle robot control
- [viz](https://github.com/ai4ce/DeepParticleRobot/blob/main/gym_dpr/envs/viz.py): Pygame visualizer
