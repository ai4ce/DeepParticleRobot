import numpy as np
import pymunk
import pygame
import math

BOT_MASS = 10
BOT_RADIUS = 25
BOT_ELASTICITY = 0.1
BOT_FRICTION = 1
BOT_MAX_FORCE = 1000
MAGNETIC_FORCE = 750

PADDLE_LENGTH = BOT_RADIUS
PADDLE_WIDTH = BOT_RADIUS / 10
PADDLE_MASS = 1
PADDLE_SPEED = 5
PADDLE_ELASTICITY = 0
PADDLE_FRICTION = 1

class ParticleRobot(object):
    '''This is an abstract interface'''
    def __init__(self, botId, shape, paddles, joints, motors):
        self.botId = botId
        self.body = shape.body
        self.shape = shape
        self.paddles = paddles
        self.joints = joints
        self.motors = motors
        self.state = None

    def observe(self, world):
        raise NotImplementedError()

    def act(self, action):
        raise NotImplementedError()

    def react(self, others):
        raise NotImplementedError()

    def get_state(self):
        return self.state

class CircularBot(ParticleRobot):
    '''A particle robot that can change its radius to interact with the environment'''
    def __init__(self, pos, botId, numPaddles = 4 * 4,
                 continuousAction=False, dead=False):
        # Initializing main body of circular particle robot
        bot_body = pymunk.Body(mass=BOT_MASS, moment=pymunk.moment_for_circle(BOT_MASS, 0, BOT_RADIUS))
        bot_body.position = pos
        bot_shape = pymunk.Circle(bot_body, radius=BOT_RADIUS)
        bot_shape.collision_type = botId
        bot_shape.friction = BOT_FRICTION
        bot_shape.elasticity = BOT_ELASTICITY
        if dead:
            bot_shape.color = (255, 0, 0, 255)

        self.radius = BOT_RADIUS
        self.magnets = []

        # Initializing paddles of particle bot
        # 2 part paddles - paddle1 attached and pivots about central body,
        #                  paddle2 attached to paddle1 and remains parallel to surface of central body
        # Notation for joints - paddle + body1 + body2 + joint type
        paddles1 = []
        paddles2 = []

        paddle12Pins = []
        paddle12Motors = []
        paddleBot2RotaryJoints = []

        for i in range(numPaddles):
            paddle_x = BOT_RADIUS * math.cos(i * 2 * math.pi / numPaddles) + bot_body.position[0]
            paddle_y = BOT_RADIUS * math.sin(i * 2 * math.pi / numPaddles) + bot_body.position[1]

            paddle1body = pymunk.Body(mass=PADDLE_MASS,
                                      moment=pymunk.moment_for_box(PADDLE_MASS, (PADDLE_LENGTH, PADDLE_WIDTH)))
            paddle1body.position = (paddle_x, paddle_y)
            paddle1body.angle = i * 2 * math.pi / numPaddles - math.pi/2
            paddle1 = pymunk.Poly.create_box(paddle1body, (PADDLE_LENGTH, PADDLE_WIDTH))
            paddle1.collision_type = botId
            paddle1.elasticity = PADDLE_ELASTICITY
            paddle1.friction = PADDLE_FRICTION
            paddles1.append(paddle1)

            paddle2body = pymunk.Body(mass=PADDLE_MASS/2,
                                           moment=pymunk.moment_for_box(PADDLE_MASS/2, (PADDLE_LENGTH/2, PADDLE_WIDTH)))
            paddle2body.position = paddle1.body.local_to_world((PADDLE_LENGTH / 2, 0))
            paddle2body.angle = paddle1.body.angle + math.pi / 2
            paddle2 = pymunk.Poly.create_box(paddle2body, (PADDLE_LENGTH/2, PADDLE_WIDTH))
            paddle2.collision_type = botId
            paddle2.elasticity = PADDLE_ELASTICITY
            paddle2.friction = PADDLE_FRICTION
            paddles2.append(paddle2)

            # Connects paddle1 and paddle2
            pin12 = pymunk.PinJoint(paddle1.body, paddle2.body, (PADDLE_LENGTH / 2, 0), (0, 0))
            pin12.collide_bodies = False
            paddle12Pins.append(pin12)

            # Rotates paddle2 such that paddle2 remains parallel to surface of circular bot
            motor12 = pymunk.SimpleMotor(paddle1.body, paddle2.body, 0)
            motor12.max_force = BOT_MAX_FORCE
            paddle12Motors.append(motor12)

            # Forces paddle2 to be parallel to circular bot
            angle = i * 2 * math.pi / numPaddles + math.pi / 2
            rotary12 = pymunk.RotaryLimitJoint(bot_shape.body, paddle2.body, angle, angle)
            rotary12.collide_bodies = False
            paddleBot2RotaryJoints.append(rotary12)

        paddleBot1Pins = []
        paddleBot1Rotary = []
        paddleBot1Motors = []

        for num, paddle1 in enumerate(paddles1):
            x = BOT_RADIUS * math.cos(num * 2 * math.pi / numPaddles)
            y = BOT_RADIUS * math.sin(num * 2 * math.pi / numPaddles)

            # Connects paddle1 and central bot
            pinBot1 = pymunk.PinJoint(bot_shape.body, paddle1.body, (x, y), (0, 0))
            pinBot1.collide_bodies = False
            paddleBot1Pins.append(pinBot1)

            # Moves/rotates/pivots paddle1
            motorBot1 = pymunk.SimpleMotor(bot_shape.body, paddle1.body, 0)
            motorBot1.max_force = BOT_MAX_FORCE
            paddleBot1Motors.append(motorBot1)

            # Limits paddle1 to [0, 90] degree contraction/expansion
            angle = num * 2 * math.pi / numPaddles - math.pi / 2
            rotaryBot1 = pymunk.RotaryLimitJoint(bot_shape.body, paddle1.body, angle, angle + math.pi/2)
            rotaryBot1.collide_bodies = False
            paddleBot1Rotary.append(rotaryBot1)

        paddle12joints = [paddle12Pins, paddleBot2RotaryJoints]
        paddleBot1joints = [paddleBot1Pins, paddleBot1Rotary]
        motor12 = paddle12Motors
        motorBot1 = paddleBot1Motors

        paddles = {'paddle1': paddles1, 'paddle2': paddles2}
        motors = {'bot1': motorBot1, '12': motor12}
        joints = {'bot1': paddleBot1joints, '12': paddle12joints}

        self.angle = paddles['paddle1'][0].body.angle - bot_body.angle + math.pi / 2
        self.targetAngle = 0
        self.reachedTargetAngle = True
        self.continuousAction = continuousAction
        self.dead = dead
        super(CircularBot, self).__init__(botId=botId, shape=bot_shape, paddles=paddles, joints=joints, motors=motors)

    def update_angle(self):
        '''
        Updates current angle of the particle robot

        :return:
        '''
        self.angle = self.paddles['paddle1'][0].body.angle - self.body.angle + math.pi / 2

    def expand(self):
        '''
        The motors will move the paddles outwards

        :return:
        '''
        for motor1 in self.motors['bot1']:
            motor1.rate = -PADDLE_SPEED
        for motor2 in self.motors['12']:
            motor2.rate = PADDLE_SPEED

    def contract(self):
        '''
        The motors will contract the paddles

        :return:
        '''
        for motor1 in self.motors['bot1']:
            motor1.rate = PADDLE_SPEED
        for motor2 in self.motors['12']:
            motor2.rate = -PADDLE_SPEED

    def stop(self):
        '''
        The motors will freeze the paddles

        :return:
        '''
        for motor1 in self.motors['bot1']:
            motor1.rate = 0
        for motor2 in self.motors['12']:
            motor2.rate = 0

    def act(self, action):
        '''
        Triggers expand/contract/stop depending on current paddle angle vs desired action angle
        (Will not act if angle and desired angle are within 10 degrees)
        (Will not act if the robot is killed)

        :param action: 0 for fully contracted, 1 for fully expanded
        :return: -1 for contraction, 0 for no action, 1 for expansion
        '''

        self.update_angle()

        if self.dead:
            return 0

        if abs(self.targetAngle - self.angle) < (math.pi / 18):
            self.reachedTargetAngle = True

        # if paddles have not reached previous desired state, allow bot to finish previous desired state
        if not self.reachedTargetAngle:
            return 0

        # paddles have reached previous desired state past this point
        # update desired state
        if self.continuousAction:
            self.targetAngle = action
        else:
            self.targetAngle = action * math.pi / 2

        # if desired state is equal to current state, do nothing
        if abs(self.targetAngle - self.angle) < (math.pi / 18):
            self.reachedTargetAngle = True
            return 0

        # if desired state is bigger than current state, expand
        if self.targetAngle > self.angle:
            self.expand()
            self.reachedTargetAngle = False
            return 1

        # if desired state is smaller than current state, contract
        elif self.targetAngle < self.angle:
            self.contract()
            self.reachedTargetAngle = False
            return -1

    def observeSelf(self):
        '''
        :return: position and velocity of self
        '''
        return self.body.position[0], self.body.position[1], self.body.velocity[0], self.body.velocity[1]

    def createMagnet(self, otherBot):
        '''
        Creates a joint between two particle robots to simulate magnetic attraction

        :param otherBot: another particle robot
        :return: "magnetic" joint
        '''
        dists = np.array([np.linalg.norm(paddle.body.position - self.shape.body.position)
                          for paddle in otherBot.paddles['paddle2']])
        ix = np.argmin(dists)
        otherPaddle = otherBot.paddles['paddle2'][ix]

        dists = np.array([np.linalg.norm(paddle.body.position - otherBot.shape.body.position)
                          for paddle in self.paddles['paddle2']])
        ix = np.argmin(dists)
        thisPaddle = self.paddles['paddle2'][ix]

        if len(thisPaddle.body.constraints) == 3 and len(otherPaddle.body.constraints) == 3:
            joint = pymunk.PinJoint(thisPaddle.body, otherPaddle.body)
            joint.max_force = MAGNETIC_FORCE
            return joint

    def createAllMagnets(self, otherBots):
        '''
        Local creation of magnets. Gets all particle robots near itself (< 2 * bot diameter) and creates a magnetic joint

        :param otherBots: List of all bots in the world
        :return: a list of magnets
        '''
        magnets = []
        dists = np.array([np.linalg.norm(otherBot.body.position - self.body.position) for otherBot in otherBots])
        ixs = np.where(np.logical_and(dists > 0, dists < BOT_RADIUS * 4))[0]

        for i in ixs:
            magnet = self.createMagnet(otherBots[i])
            if magnet != None:
                magnets.append(magnet)
        self.magnets = magnets
        return magnets
