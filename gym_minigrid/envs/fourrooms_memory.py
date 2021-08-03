#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
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
        self._randomization_freq = 10000
        self._num_room_objs = 3
        super().__init__(grid_size=30, max_steps=100)

    def _gen_grid(self, width, height):
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
        
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            pos = (self._rand_int(room_w+1, 2*room_w-1), self._rand_int(room_h+1, 2*room_h-1))
            self.agent_pos = pos
            self.agent_dir = self._rand_int(0, 4)

        shape_colors = ['red','green','blue','purple']
        shape_types = ['square','circle','triangle','upside_down_triangle']
        
        if self._current_ep % self._randomization_freq == 0:
            self.goal_shape = random.choice(shape_types)
            self.goal_color = random.choice(shape_colors)
            self.shape_rooms = list(shape_types)
            random.shuffle(self.shape_rooms)
            
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
                self.place_obj(obj,room_tops[i],(room_w-4,room_h-4))
        
        self.mission = 'win'
        
    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

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
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
                
            if fwd_cell != None and \
                fwd_cell.shape == self.goal_shape and \
                fwd_cell.color == self.goal_color:
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

        obs = self.gen_obs()

        return obs, reward, done, {}

    def reset(self):
        super().reset()
        self._current_ep+=1
    

register(
    id='MiniGrid-FourRoomsMemory-v0',
    entry_point='gym_minigrid.envs:FourRoomsMemoryEnv'
)
