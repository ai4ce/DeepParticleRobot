import math

import pandas as pd
import torch
import torch_geometric

from gym_dpr.envs.viz import Visualizer
import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot
from gym_dpr.envs.DPR_ParticleRobot import CircularBot
from gym_dpr.envs.DPR_SuperAgent import SuperCircularBot
from gym_dpr.envs.DPR_World import World


world = World(visualizer=None)

for n in range(8):
    world.removeAll()
    bot = SuperCircularBot(9, (250 * math.cos(n * math.pi / 8), 250 * math.sin(n * math.pi / 8)))
    world.addSuperAgent(bot)
    for _ in range(2500):
        totalSteps, actions = world.wavePolicy()
        for i in range(totalSteps):
            edge_index = torch.tensor(world.edge_index, dtype=torch.long)
            x = torch.tensor(world.node_features, dtype=torch.float)
            data = torch_geometric.data.Data(x=x, edge_index=edge_index)
            for j in range(10):
                action = actions[i]
                world.step(action)

                # world.frame_id = _ + (i * 10) + j
                # if world.visualizer is not None and world.visualizer.viz(_, world) == False:
                #     break





