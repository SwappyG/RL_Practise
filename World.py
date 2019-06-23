# -*- coding: future_fstrings -*-

import sys
import numpy as np

class World(object):

    def __init__(self, state_space, action_space):
        self.S = state_space
        self.A = action_space

    def GetReward(self, S):
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}")

    def GetAllStates(self):
        return self.S

    def GetAllActions(self):
        return self.A

    def GetNextState(self, S, A):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')


class Coord(object):
    def __init__(self, x=0, y=0, **lims):
        self.x = x
        self.y = y
        self.x_min = lims.get('x_min', None)
        self.x_max = lims.get('x_max', None)
        self.y_min = lims.get('y_min', None)
        self.y_max = lims.get('y_max', None)

    def _Clip(self, val, min, max):
        return np.maximum( np.minimum(val, max) , min)

    def Up(self, val=1):
        self.y = self._Clip(self.y+val, self.y_min, self.y_max)

    def Down(self, val=1):
        self.y = self._Clip(self.y-val, self.y_min, self.y_max)

    def Right(self, val=1):
        self.y = self._Clip(self.x+val, self.x_min, self.x_max)

    def Left(self, val=1):
        self.y = self._Clip(self.x-val, self.x_min, self.x_max)

    def Get(self):
        return (self.x, self.y)

class DynamicNDWorld(World):

    NORMAL_STATE = 0
    START_STATE = 1
    GOAL_STATE = 2
    HAZARD_STATE = 3

    NORMAL_REWARD = -1
    OUT_OF_BOUND_REWARD = -50
    HAZARD_REWARD = -50
    GOAL_REWARD = -1

    def __init__(self, max_coords, move_dict, **kwargs):
        World.__init__(self, max_coords, move_dict)

        state_space = np.ones(shape=max_coords, dtype=np.int32) * DynamicNDWorld.NORMAL_STATE
        self._zero_state = np.array( [0 for _ in max_coords] )
        self._no_move = np.array( [0 for _ in max_coords] )
        self._final_state = np.array( [coord-1 for coord in max_coords] )
        self._num_dims = len(max_coords)

        self.start_state  =  kwargs.get( 'start_pos', self._zero_state )
        self.goal_state   =  kwargs.get( 'goal_pos', self._final_state )
        self.hazard_func  =  kwargs.get( 'hazard_func', lambda *_: False )
        self.dynamics_func    =  kwargs.get( 'dynamics_func', lambda *_: self._no_move )
        self.noise_func    =  kwargs.get( 'noise_func', lambda *_: self._no_move )

        try:
            state_space[self.start_state] = DynamicNDWorld.START_STATE
            state_space[self.goal_state] = DynamicNDWorld.GOAL_STATE
        except IndexError as err:
            print f"start, goal and hazard pos must all be within state_space, got exception: {err}"


    def _Clip(self, val, min, max):
        return np.maximum( np.minimum(val, max) , min)

    def _IsOutOfBound(self, S):
        return np.any(S > self._final_state) or np.any(S < self._zero_state)

    def GetNextState(self, S, A):

        # Grab the move for the given action from the move dict
        move = self.A[A]
        S = S + move

        if self._IsOutOfBound(S):
            self.out_of_bounds = True
            return self.start_state

        S = S + self.dynamics_func(S) + self.noise_func(S)
        if self._IsOutOfBound(S):
            self.out_of_bounds = True
            return self.start_state

        if self.hazard_func(S):
            self.hit_hazard = True
            return self.start_state

        self.hit_hazard = False
        self.out_of_bounds = False
        return S

    def GetReward(self, S):
        if self.out_of_bounds:
            return DynamicNDWorld.OUT_OF_BOUND_REWARD
        elif self.hit_hazard:
            return DynamicNDWorld.HAZARD_REWARD
        elif S == self.goal_state:
            return DynamicNDWorld.GOAL_REWARD
        else:
            return DynamicNDWorld.NORMAL_REWARD




if __name__=="__main__":

    # Run tests for World and DynamicNDWorld classes

    ss = [1,2,4,7,8,9]
    actions = ['A', 'Z']
    world = World(ss, actions)

    try:
        if (world.GetAllStates() == ss): print "Got All states correctly"
        if (world.GetAllActions() == actions): print "Got All actions correctly"
        try:
            world.GetNextState(0, 0)
        except NotImplementedError:
            print "GetNextState threw NotImplementedError, as expected"
        try:
            world.GetReward(0)
        except NotImplementedError:
            print "GetReward threw NotImplementedError, as expected"

        print f"Class {world.__class__.__name__} passed all tests"
    except Exception as e:
        print f"Class {world.__class__.__name__} encountered an exception: {e}"

    move = {'U':(0,1), 'D':(0,-1), 'R':(1,0), 'L':(-1,0)}
    coords = np.array((7,3))
    st = np.array((0,0))
    gl = np.array((5,2))

    def dynamics_func(S):
        a = 0
        if S[0]%2 and S[1]%2:
            a = 1
        return np.array([a,0])

    def noise_func(S):
        if np.random.rand() > .5:
            mv = np.random.normal(loc=0, scale=0.1,size=(2))
            print f"hit noise: {mv}"
            return mv
        else:
            return np.array((0,0))

    ww = DynamicNDWorld(max_coords=(7,3), move_dict=move, start_pos=st, goal_pos=gl)

    s = np.array((0,0))
    while True:
        print f"s: {s}, enter move or enter 'Q'/'q' to quit:"
        mv = raw_input()
        if mv == 'Q' or mv == 'q':
            break
        else:
            try:
                s = ww.GetNextState(s, mv)
            except KeyError as e:
                print f"Invalid move command, got error: {e}\nTry again"

    # print ww.GetNextState(np.array((0,0)), 'D')
    # print ww.GetNextState(np.array((1,1)), 'L')
    # print ww.GetNextState(np.array((1,1)), 'U')
    # print ww.GetNextState(np.array((1,2)), 'L')
    # print ww.GetNextState(np.array((1,2)), 'R')
