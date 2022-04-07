# A Deep Reinforcement Learning Environment for Particle Robot Navigation and Object Manipulation

[**Jeremy Shen**](https://github.com/jshen04), [**Erdong Xiao**](https://github.com/ErdongXiao), [**Yuchen Liu**](https://github.com/Rtlyc), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

![Overview](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/overview.PNG?token=ANKETMQES4EKYCQYAE4K4WLAN4WB4)

|[Abstract](#abstract)|[Code](#code-github)|[Paper](#paper-arxiv)|[Results](#results)|[Acknowledgment](#acknowledgment)|

## Abstract
Particle robots are novel biologically-inspired robotic systems where locomotion can be achieved collectively and robustly, but not independently. While its control is currently limited to a hand-crafted policy for basic locomotion tasks, such a multi-robot system could be potentially controlled via Deep Reinforcement Learning (DRL) for different tasks more efficiently. However, the particle robot system presents a new set of challenges for DRL differing from existing swarm robotics systems: the low degrees of freedom of each robot and the increased necessity of coordination between robots. We present a 2D particle robot simulator using the OpenAI Gym interface and Pymunk as the physics engine, and introduce new tasks and challenges to research the underexplored applications of DRL in the particle robot system. Moreover, we use Stable-baselines3 to provide a set of benchmarks for the tasks. Current baseline DRL algorithms show signs of achieving the tasks but are yet unable to reach the performance of the hand-crafted policy. Further development of DRL algorithms is necessary in order to accomplish the proposed tasks.

## [Code (GitHub)](https://github.com/ai4ce/DeepParticleRobot)
```
The code is copyrighted by the authors. Permission to copy and use 
 this software for noncommercial use is hereby granted provided: (a)
 this notice is retained in all copies, (2) the publication describing
 the method (indicated below) is clearly cited, and (3) the
 distribution from which the code was obtained is clearly cited. For
 all other uses, please contact the authors.
 
 The software code is provided "as is" with ABSOLUTELY NO WARRANTY
 expressed or implied. Use at your own risk.
This code provides an implementation of the method described in the
following publication: 
Jeremy Shen, Erdong Xiao, Yuchen Liu, and Chen Feng,    
"A Deep Reinforcement Learning Environment for Particle Robot Navigation 
and Object Manipulation (arXiv)". 
``` 
## How to use

Our environment is developed based on the [OpenAi Gym](https://gym.openai.com/). You can simply follow the similar way to use our environment. Here we present an example for using 1D static task environment.
```
from DMP_Env_1D_static import deep_mobile_printing_1d1r ### you may need to find the path to this environment in [Env] folder 
env = deep_mobile_printing_1d1r(plan_choose=2) ### plan_choose could be 0: sin, 1: Gaussian, and 2: Step curve  
observation = env.reset()
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.clear()
for _ in range(1000):
  action = np.random.randint(env.action_dim) # your agent here (this takes random actions)
  observation, reward, done = env.step(action)
  env.render(ax)
  plt.pause(0.1)
  if done:
    break
plt.show()
```

## [Paper (arXiv)](https://arxiv.org/abs/2203.06464)
To cite our paper:
```
@misc{https://doi.org/10.48550/arxiv.2203.06464,
  doi = {10.48550/ARXIV.2203.06464},
  url = {https://arxiv.org/abs/2203.06464},
  author = {Shen, Jeremy and Xiao, Erdong and Liu, Yuchen and Feng, Chen},
  keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Deep Reinforcement Learning Environment for Particle Robot Navigation and Object Manipulation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

### Task environment setups  
![env](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/environment.PNG?token=ANKETMWZOL7HPJVJHNDX2B3AN4WDE)

## Comparison 
**Comparison between other robotic learning tasks with ours.**
![table](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/comparison_table.PNG?token=ANKETMXBFQCPNT5CBK5GVKDAN4YP4)

## Results
**Benchmark results for all baselines, including human baseline: average IoU(left) and minimum IoU(right). Human data of 3D environment is not collected, because it is time-consuming for human to play one game.**
![Baseline_curve](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/result_curve.PNG?token=ANKETMSFKVHG2SIV2JIVMLTAN4WEW)

**The best testing visualized results of baselines on all tasks.**
![Baseline_visualize](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/results_fig.PNG?token=ANKETMRREVGARACAVL44QJLAN4WFW)

## Acknowledgment
This research is supported by the NSF CPS program under CMMI-1932187.