# -*- coding: future_fstrings -*-

import sys
import numpy as np

"""
Holds a WorldSpace instance defining the valid states and actions
Returns reward given state, and next state given state and action
This is a template class that must be implemented by a child class
"""
class World(object):

    def __init__(self, world_space):
        self.world_space = world_space

    def GetReward(self, S):
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}")

    def GetNextState(self, S, A):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

    def IsTerminal(self, S):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

"""
Implements the World class.
Defines an Ndim World (discrete) with a start state and goal state
Optional dynamics, hazard and noise functions can also be defined
State transitions occur by adding an action (from world_space) to state
Additional motion is achieved via dynamics, hazard and noise functions
"""
class DynamicNDWorld(World):

    NORMAL_REWARD = -1
    OUT_OF_BOUND_REWARD = -50
    HAZARD_REWARD = -50
    GOAL_REWARD = -1

    def __init__(self, world_space, **kwargs):
        World.__init__(self, world_space)

        # creates a [0,0,..,0] array, len of dims in state
        self._no_move = np.array( [0 for _ in self.world_space.GetSDims()] )

        # Stores the start and goal states if specified, otherwise use zero/last
        self.start_state  =  kwargs.get( 'start_pos', self.world_space.ZeroState() )
        self.goal_state   =  kwargs.get( 'goal_pos', self.world_space.LastState() )

        # Stores func handles if specified, otherwise store default lambdas
        self.hazard_func  =  kwargs.get( 'hazard_func', lambda *_: False )
        self.dynamics_func =  kwargs.get( 'dynamics_func', lambda *_: self._no_move )
        self.noise_func    =  kwargs.get( 'noise_func', lambda *_: self._no_move )

        # Make sure specified star and goal is valid
        if not self.world_space.IsValidState(self.start_state) or
            not self.world_space.IsValidState(self.goal_state):
            print f"start, start and goal must all be within world_space.\nS: {self.world_space.GetSDims()}\nGot the following...\nstart: {self.start_state}\ngoal: {self.goal_state}"

    # Returns the next state
    # param: S - indices for each dim as np.array
    # param: A - index of action
    def GetNextState(self, S, A):

        # Grab the move for the given action from the move dict
        move = self.world_space.ActionVal(index=A)
        S = S + move

        # If we moved out of bounds, set flag, return to start_state
        if not self.world_space.IsValidState(S):
            self.out_of_bounds = True
            return self.start_state

        # Apply dynamics and noise from new state
        S = S + self.dynamics_func(S) + self.noise_func(S)

        # If we moved out of bounds, set flag, return to start_state
        if not self.world_space.IsValidState(S):
            self.out_of_bounds = True
            return self.start_state

        # Set flag if in hazard and move to start
        if self.hazard_func(S):
            self.hit_hazard = True
            return self.start_state

        # clear flags if we hit nothing, return the state
        self.hit_hazard = False
        self.out_of_bounds = False
        return S

    # Returns reward corresponding to latest (S,A) pair
    def GetReward(self, S):
        # return appropriate reward based on flags set during GetNextState
        if self.out_of_bounds:
            return DynamicNDWorld.OUT_OF_BOUND_REWARD
        elif self.hit_hazard:
            return DynamicNDWorld.HAZARD_REWARD
        elif S == self.goal_state:
            return DynamicNDWorld.GOAL_REWARD
        else:
            return DynamicNDWorld.NORMAL_REWARD

    def IsTerminal(self, S):
        return S == self.goal_state


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
