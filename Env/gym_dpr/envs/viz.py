'''
viz_old.py in DeepParticleRobot
author  : cfeng
created : 6/18/20 9:08PM
'''

import os
import sys
import argparse

import pygame
import pygame.constants
import pymunk.pygame_util


class Visualizer(object):
    def __init__(self, width=600, height=500):
        pygame.init()
        self._screen = pygame.display.set_mode((width, height))
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)
        self._running = True
        pymunk.pygame_util.positive_y_is_up = True
        # self._screen = pygame.transform.flip(self._screen, True, False)
        # self.fig = plt.figure()

    def _process_events(self, timestep):
        for event in pygame.event.get():
            if event.type == pygame.constants.QUIT:
                self._running = False
            elif event.type == pygame.constants.KEYDOWN and event.key == pygame.constants.K_ESCAPE:
                self._running = False
            elif event.type == pygame.constants.KEYDOWN and event.key == pygame.constants.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")

    def viz(self, timestep, world):
        # print('step={}'.format(timestep))
        self._process_events(timestep)
        self._screen.fill(pygame.color.THECOLORS["white"])
        world.space.debug_draw(self._draw_options)
        pygame.display.flip()
        # Delay fixed time between frames
        self._clock.tick(50)
        pygame.display.set_caption("time={}, fps={:.0f}".format(timestep, self._clock.get_fps()))
        return self._running


def test(WorldClass, BotClass, policy=None):
    from DPR_SuperAgent import SuperCircularBot
    world = WorldClass(visualizer=Visualizer())
    world.addSuperAgent(SuperCircularBot(16, (400, 400), BotClass))
    world.run(10000)

def main(args):
    from DPR_World import World
    from DPR_ParticleRobot import CircularBot
    test(World, CircularBot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)