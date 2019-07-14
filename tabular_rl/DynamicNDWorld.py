# -*- coding: future_fstrings -*-

from World import World
from WorldSpace import WorldSpace
import numpy as np

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
        self.start_state  =  kwargs.get( 'start_state', self.world_space.ZeroState() )
        self.goal_state   =  kwargs.get( 'goal_state', self.world_space.LastState() )

        # Stores func handles if specified, otherwise store default lambdas
        self.hazard_func  =  kwargs.get( 'hazard_func', lambda *_: False )
        self.dynamics_func =  kwargs.get( 'dynamics_func', lambda *_: self._no_move )
        self.noise_func    =  kwargs.get( 'noise_func', lambda *_: self._no_move )

        self.out_of_bounds = False
        self.hit_hazard = False

        # Make sure specified star and goal is valid
        if not self.world_space.IsValidState(self.start_state) or \
            not self.world_space.IsValidState(self.goal_state):
            raise ValueError(f"start, start and goal must all be within world_space.\nS: {self.world_space.GetSDims()}\nGot the following...\nstart: {self.start_state}\ngoal: {self.goal_state}")

    # Returns the next state
    # param: S - indices for each dim as np.array
    # param: A - index of action
    def GetNextState(self, S, A):

        S = np.array(S)
        self.hit_hazard = False
        self.out_of_bounds = False

        if not self.world_space.IsValidState(S):
            raise ValueError("Starting state is not in state_space")

        # Grab the move for the given action from the move dict
        try:
            move = self.world_space.ActionVal(index=A)
        except KeyError:
            raise ValueError(f"Got KeyError from WorldSpace object, Action [{A}] must be less than size of action_map in world_space")

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

        return S

    # Returns reward corresponding to latest (S,A) pair
    def GetReward(self, S):
        # return appropriate reward based on flags set during GetNextState
        if self.out_of_bounds:
            return DynamicNDWorld.OUT_OF_BOUND_REWARD
        elif self.hit_hazard:
            return DynamicNDWorld.HAZARD_REWARD
        elif np.all(S == self.goal_state):
            return DynamicNDWorld.GOAL_REWARD
        else:
            return DynamicNDWorld.NORMAL_REWARD

    def IsTerminal(self, S):
        return S == self.goal_state

"""
If this file is run as main, it performs unit tests on this class
"""
if __name__=="__main__":

    # Run tests for World and DynamicNDWorld classes

    import unittest
    from collections import OrderedDict

    class TestDynamicNDWorldInit(unittest.TestCase):

        ARR = lambda x: np.array(x)
        ALL = lambda x: np.all(x)
        ACT = lambda x: self.ws.ActionVal(index=x)

        def setUp(self):

            # Insert elements in order to keep track
            self.a_map = OrderedDict()
            self.a_map['U'] = (1,1)
            self.a_map['D'] = (-1,-1)
            self.a_map['R'] = (1,0)
            self.a_map['L'] = (1,-2)

            # Make WorldSpace instance using ss and a_map
            self.ss = np.array((9,7))
            self.ws = WorldSpace(self.ss, self.a_map)

            # instantiate our (plain) DynamicNDWorld with some args
            self.w_kw = {'start_state':(3,4), 'goal_state':(8,3)}
            self.world = DynamicNDWorld(self.ws, **self.w_kw)

            self.ARR = lambda x: np.array(x)
            self.ALL = lambda x: np.all(x)
            self.ACT = lambda x: self.ws.ActionVal(index=x)

        def tearDown(self):
            del self.ws
            del self.world

        def hazard_func(self, S):
            if S[0] % 2:
                return True
            else:
                return False

        def dynamics_func(self, S):
            if S[0] % 2:
                return (1,0)
            else:
                return (0,0)

        def test_Initialization(self):
            ARR, ALL, ACT = self.ARR, self.ALL, self.ACT

            bad_kw = {'start_state':(-1, 2), 'goal_state':(8,3)}
            with self.assertRaises(ValueError):
                DynamicNDWorld(self.ws, **bad_kw)

            bad_kw = {'start_state':(1, 2), 'goal_state':(1,10)}
            with self.assertRaises(ValueError):
                DynamicNDWorld(self.ws, **bad_kw)

            large_ws = WorldSpace((934,36,52,343), {"U":(0,0,0,1)})
            scrap_world = DynamicNDWorld(large_ws)

            try:
                scrap_world = DynamicNDWorld(large_ws, **self.w_kw)
            except ValueError:
                pass
            except:
                self.fail()

        def test_NextState_Transitions(self):
            ARR, ALL, ACT = self.ARR, self.ALL, self.ACT

            self.assertTrue( ALL( self.world.GetNextState( ARR((3,4)), 0) == ARR((3,4)) + ACT(0) ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((2,2)), 0) == ARR((2,2)) + ACT(0) ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((8,6)), 0) == self.world.start_state ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((2,3)), 1) == ARR((2,3)) + ACT(1) ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((0,0)), 1) == self.world.start_state ) )
            self.assertFalse( ALL( self.world.GetNextState( ARR((7,5)), 0) == self.world.start_state ) )

        def test_NextState_InvalidStates(self):
            ARR, ALL, ACT = self.ARR, self.ALL, self.ACT

            with self.assertRaises(ValueError):
                ALL( self.world.GetNextState( ARR((-4,5)), 0) )
            with self.assertRaises(ValueError):
                ALL( self.world.GetNextState( ARR((1,-5)), 0) )
            with self.assertRaises(ValueError):
                ALL( self.world.GetNextState( ARR((9,9)), 0) )

        def test_NextState_InvalidActions(self):
            ARR, ALL, ACT = self.ARR, self.ALL, self.ACT

            with self.assertRaises(ValueError):
                ALL( self.world.GetNextState( ARR((3,4)), 5) )
            with self.assertRaises(ValueError):
                ALL( self.world.GetNextState( ARR((3,4)), -2) )

        def test_HazardFunc(self):
            ARR, ALL, ACT = self.ARR, self.ALL, self.ACT

            self.world.hazard_func = self.hazard_func
            self.assertTrue( ALL( self.world.GetNextState( ARR((1,1)), 0) == ARR((1,1)) + ACT(0) ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((1,2)), 0) == ARR((1,2)) + ACT(0) ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((2,1)), 0) == self.world.start_state ) )

        def test_NextState_RewardCheck(self):
            ARR, ALL, ACT = self.ARR, self.ALL, self.ACT

            self.world.hazard_func = self.hazard_func

            self.world.GetNextState( ARR((1,1)), 0)
            self.assertFalse( self.world.out_of_bounds )
            self.assertFalse( self.world.hit_hazard )

            self.world.GetNextState( ARR((1,2)), 0)
            self.assertFalse( self.world.out_of_bounds )
            self.assertFalse( self.world.hit_hazard )

            self.world.GetNextState( ARR((8,6)), 0)
            self.assertTrue( self.world.out_of_bounds )
            self.assertFalse( self.world.hit_hazard )

            self.world.GetNextState( ARR((0,0)), 1)
            self.assertTrue( self.world.out_of_bounds )
            self.assertFalse( self.world.hit_hazard )

            self.world.GetNextState( ARR((2,1)), 0)
            self.assertFalse( self.world.out_of_bounds )
            self.assertTrue( self.world.hit_hazard )

            self.world.GetNextState( ARR((4,4)), 1)
            self.assertFalse( self.world.out_of_bounds )
            self.assertTrue( self.world.hit_hazard )

        def test_GetReward(self):
            ARR, ALL, ACT = self.ARR, self.ALL, self.ACT

            S = self.world.GetNextState( ARR((3,4)), 0)
            self.assertTrue( self.world.GetReward(S) == DynamicNDWorld.NORMAL_REWARD )

            S = self.world.GetNextState( ARR((2,3)), 0)
            self.assertTrue( self.world.GetReward(S) == DynamicNDWorld.NORMAL_REWARD )

            S = self.world.GetNextState( ARR((8,6)), 0)
            self.assertTrue( self.world.GetReward(S) == DynamicNDWorld.OUT_OF_BOUND_REWARD )

            S = self.world.GetNextState( ARR((7,2)), 0)
            self.assertTrue( self.world.GetReward(S) == DynamicNDWorld.GOAL_REWARD )

        def test_DynamicFunc(self):
            ARR, ALL, ACT = self.ARR, self.ALL, self.ACT

            self.world.dynamics_func = self.dynamics_func

            self.assertTrue( ALL( self.world.GetNextState( ARR((1,1)), 0) == ARR((1,1)) + ACT(0) ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((1,4)), 0) == ARR((1,4)) + ACT(0) ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((2,1)), 0) == ARR((2,1)) + ACT(0) + ARR((1,0)) ) )
            self.assertTrue( ALL( self.world.GetNextState( ARR((4,2)), 1) == ARR((4,2)) + ACT(1) + ARR((1,0)) ) )

    unittest.main()
