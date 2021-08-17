#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from collections import deque
from gym.spaces import Box
from gym import Wrapper
import numpy as np
import random

def create_shape(shape,color):
    if shape=='square':
        return ColoredSquare(color)
    elif shape=='circle':
        return ColoredCircle(color)
    elif shape=='triangle':
        return ColoredTriangle(color)
    elif shape=='upside_down_triangle':
        return ColoredUpsideDownTriangle(color)

class FourRoomsMemoryEnv(MiniGridEnv):

    def __init__(self, agent_pos=None, goal_pos=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self._current_ep = 0
        self._randomization_freq = 25000000000
        self._num_room_objs = 3
        super().__init__(grid_size=30, max_steps=250)
        self.observation_space = self.observation_space.spaces['image']

    def _gen_grid(self, width, height):
        #self.seed(0)
        #random.seed(0)
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 3
        room_h = height // 3

        #self.grid.horz_wall(0, room_h)
        #self.grid.horz_wall(0, 2*room_h)
        #self.grid.vert_wall(room_w, 0)
        #self.grid.vert_wall(room_w*2, 0)

        #pos = (room_w, self._rand_int(room_h+3, 2*room_h-3))
        #self.grid.set(*pos, None)
        #pos = (2*room_w, self._rand_int(room_h+3, 2*room_h-3))
        #self.grid.set(*pos, None)
        #pos = (self._rand_int(room_w+3, 2*room_w-3), room_h)
        #self.grid.set(*pos, None)
        #pos = (self._rand_int(room_w+3, 2*room_w-3), 2*room_h)
        #self.grid.set(*pos, None)

        shape_colors = ['red','green','blue','purple']
        shape_types = ['square','circle','triangle','upside_down_triangle']

        if self._current_ep % self._randomization_freq == 0:
            self.goal_shape = 'triangle'#random.choice(shape_types)
            self.goal_color = 'purple'#random.choice(shape_colors)
            #self.shape_rooms = list(shape_types)
            self.shape_rooms = shape_types = ['square','circle','triangle','upside_down_triangle']
            #random.shuffle(self.shape_rooms)
        self.shape_rooms = list(shape_types)
        random.shuffle(self.shape_rooms)
        self.goal_shape = random.choice(shape_types)
        self.goal_color = random.choice(shape_colors)

        hint_placements = [(width//2, height//2 - 3),
                     (width//2 + 3, height//2),
                     (width//2, height//2 + 3),
                     (width//2 - 3, height//2)]
        self.hint_placement = random.choice(hint_placements)
        self.hint_obj = create_shape(self.goal_shape,self.goal_color)
        self.grid.set(*self.hint_placement, self.hint_obj)

        obj_placement = [(width//2, height//2 - 3),
                     (width//2 + 3, height//2),
                     (width//2, height//2 + 3),
                     (width//2 - 3, height//2)]

        #for i in range(4):
        #    shape = self.shape_rooms[i]
        #    obj = create_shape(shape,'grey')
        #    self.grid.set(*obj_placement[i], obj)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._agent_default_dir#self._rand_int(0, 4)  # assuming random start direction
        else:
            #pos = (self._rand_int(room_w+1, 2*room_w-1), self._rand_int(room_h+1, 2*room_h-1))
            #self._agent_default_pos = pos
            self.agent_pos = (width//2,height//2)
            self.agent_dir = self._rand_int(0, 4)
            #self._agent_default_dir = self.agent_dir

        room_tops = [(room_w+2,2),
                     (room_w*2+2,room_h+2),
                     (room_w+2,room_h*2+2),
                     (2,room_h+2)]

        for i in range(4):
            shape = self.shape_rooms[i]
            for j in range(self._num_room_objs):
                if shape==self.goal_shape and j==0:
                    color = self.goal_color
                else:
                    color = random.choice(shape_colors)
                    while color == self.goal_color:
                        color = random.choice(shape_colors)

                obj = create_shape(shape,color)
                pos = self.place_obj(obj,room_tops[i],(room_w-4,room_h-4))
                if shape==self.goal_shape and j==0:
                    self.goal_pos = pos

        self.mission = 'win'

    def step(self, action):
        self.step_count += 1

        done = False
        reward = 0

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
            if fwd_cell != None and \
                'colored' in fwd_cell.type and \
                fwd_cell.shape == self.goal_shape and \
                fwd_cell.color == self.goal_color and \
                fwd_cell!=self.hint_obj:
                reward = 1#self._reward()
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        reward+= self._reward()
        obs = self.gen_obs()
        obs = obs['image']

        return obs, reward, done, {}

    def _reward(self):
        agent_pos = np.array(self.agent_pos)
        goal_pos = np.array(self.goal_pos)
        return -np.linalg.norm(goal_pos - agent_pos)/100.0

    def reset(self):
        obs = super().reset()
        self._current_ep+=1
        obs = obs['image']
        return obs


register(
    id='MiniGrid-FourRoomsMemory-v0',
    entry_point='gym_minigrid.envs:FourRoomsMemoryEnv'
)

def rgb_env():
    env = FourRoomsMemoryEnv()
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    return env

register(
    id='MiniGrid-FourRoomsMemoryRGB-v0',
    entry_point='gym_minigrid.envs:rgb_env'
)


class FrameStack(Wrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low, num_stack, axis=-1)
        high = np.repeat(self.observation_space.high, num_stack, axis=-1)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.concatenate(list(self.frames),-1)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation()

def frame_stack_env():
    env = FourRoomsMemoryEnv()
    env = FrameStack(env, 4)
    return env

register(
    id='MiniGrid-FourRoomsMemoryStacked-v0',
    entry_point='gym_minigrid.envs:frame_stack_env'
)
