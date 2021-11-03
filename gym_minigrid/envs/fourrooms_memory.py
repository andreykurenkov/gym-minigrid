#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from gym_minigrid.wrappers import FrameStack
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

    def __init__(self,
                 agent_pos=(7,7),
                 goal_pos=None,
                 random_seed=True,
                 random_goal=True,
                 random_rooms=False,
                 random_agent_pos=False,
                 room_walls=False,
                 room_shape_hints=False,
                 bits_actions=True,
                 num_bits = 8):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self._current_ep = 0
        self._randomization_freq = 25000000000
        self._num_room_objs = 1
        self._random_seed = random_seed
        self._random_goal = random_goal
        self._random_rooms = random_rooms
        self._random_agent_pos = random_agent_pos
        self._room_walls = room_walls
        self._room_shape_hints = room_shape_hints
        self._bits_actions = bits_actions
        self._num_bits = num_bits
        self.shape_colors = ['red','green','blue','purple']
        self.shape_types = ['square','circle','triangle','upside_down_triangle']
        super().__init__(grid_size=15, max_steps=100)
        if bits_actions:
            self.memory_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(1,),
                dtype='uint8'
            )
            self.observation_space.spaces['memory'] = self.memory_observation_space
            self.bits = [False]*num_bits
            self.action_space = spaces.Discrete(len(self.actions)+4)
            self.bit_memory = 0
        else:
            self.observation_space = self.observation_space.spaces['image']

    def _gen_grid(self, width, height, reset = True):
        if not self._random_seed:
            self.seed(0)
            random.seed(0)

        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 3
        room_h = height // 3

        if self._room_walls:
            self.grid.horz_wall(0, room_h)
            self.grid.horz_wall(0, 2*room_h)
            self.grid.vert_wall(room_w, 0)
            self.grid.vert_wall(room_w*2, 0)

            pos = (room_w, self._rand_int(room_h+3, 2*room_h-3))
            self.grid.set(*pos, None)
            pos = (2*room_w, self._rand_int(room_h+3, 2*room_h-3))
            self.grid.set(*pos, None)
            pos = (self._rand_int(room_w+3, 2*room_w-3), room_h)
            self.grid.set(*pos, None)
            pos = (self._rand_int(room_w+3, 2*room_w-3), 2*room_h)
            self.grid.set(*pos, None)

        shape_colors = ['red','green','blue','purple']
        shape_types = ['square','circle','triangle','upside_down_triangle']

        if self._current_ep % self._randomization_freq == 0:
            if self._random_goal:
                self.goal_shape = random.choice(shape_types)
                self.goal_color = random.choice(shape_colors)
            else:
                self.goal_shape = 'triangle'
                self.goal_color = 'purple'

            if self._random_rooms:
                self.shape_rooms = list(shape_types)
                random.shuffle(self.shape_rooms)
            else:
                self.shape_rooms = shape_types = ['square','circle','triangle','upside_down_triangle']

        if reset:
            self.shape_rooms = list(shape_types)
            random.shuffle(self.shape_rooms)
            self.goal_shape = random.choice(shape_types)
            self.goal_color = random.choice(shape_colors)

        hint_placements = [(width//2, 1),#height//2 - 3),
                     (width//2 + 3, height//2),
                     (width//2, height//2 + 3),
                     (width//2 - 3, height//2)]
        if self._random_goal:
            self.hint_placement = hint_placements[0]
            self.hint_obj = create_shape(self.goal_shape,self.goal_color)
            self.grid.set(*self.hint_placement, self.hint_obj)

        obj_placement = [(width//2, height//2 - 3),
                     (width//2 + 3, height//2),
                     (width//2, height//2 + 3),
                     (width//2 - 3, height//2)]

        if self._room_shape_hints:
            for i in range(4):
                shape = self.shape_rooms[i]
                obj = create_shape(shape,'grey')
                self.grid.set(*obj_placement[i], obj)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = 3
        else:
            if self._random_agent_pos:
                pos = (self._rand_int(room_w+1, 2*room_w-1), self._rand_int(room_h+1, 2*room_h-1))
                self._agent_default_pos = pos
            else:
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
               'colored' in fwd_cell.type:
                if fwd_cell.shape == self.goal_shape and \
                fwd_cell.color == self.goal_color and \
                fwd_cell!=self.hint_obj:
                    reward = 5#self._reward()
                    #self._gen_grid(30, 30, False)
                    done = True
                else:
                    reward = -5

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

        elif action >= 7:
            if not self.bits[action-7]:
                self.bit_memory+= 2**(action-7)
            else:
                self.bit_memory-= 2**(action-7)
            self.bits[action-7] = not self.bits[action-7]

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        reward+= self._reward()
        if action > 2 and action < 7:
            reward-= 0.1
        '''
        elif action >= 7 and action < 11:
            shape_index = self.shape_types.index(self.goal_shape)
            if action-7 == shape_index:
                if self.bits[shape_index]:
                    reward+=0.5
                else:
                    reward-=0.5
            else:
                reward-=0.05

        elif action >=11:
            color_index = self.shape_colors.index(self.goal_color)
            if action-11 == color_index:
                if self.bits[color_index+4]:
                    reward+=0.5
                else:
                    reward-=0.5
            else:
                reward-=0.05
        '''
        obs = self.gen_obs()
        obs['memory'] = [int(x) for x in self.bits]
        #obs = obs['image']

        return obs, reward, done, {}

    def _reward(self):
        agent_pos = np.array(self.agent_pos)
        goal_pos = np.array(self.goal_pos)
        reward = -np.linalg.norm(goal_pos - agent_pos)/100.0
        return reward

    def reset(self):
        obs = super().reset()
        if self._bits_actions:
            self.bits = [False]*self._num_bits
            self.bit_memory =  0
            obs['memory'] = [int(x) for x in self.bits]
        else:
            obs = obs['image']
        self._current_ep+=1
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

def frame_stack_env():
    env = FourRoomsMemoryEnv()
    env = FrameStack(env, 4)
    return env

register(
    id='MiniGrid-FourRoomsMemoryStacked-v0',
    entry_point='gym_minigrid.envs:frame_stack_env'
)
