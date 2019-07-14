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

if __name__=="__main__":

    import unittest
    from collections import OrderedDict
    from WorldSpace import WorldSpace

    class TestWorld(unittest.TestCase):

        def setUp(self):

            self.ss = (6,7,9)

            # Insert elements in order to keep track
            self.a_map = OrderedDict()
            self.a_map['U'] = (1,1)
            self.a_map['D'] = (-1,-1)
            self.a_map['R'] = (1,0)
            self.a_map['L'] = (1,-2)

            # Create a World and WorldSpace instance
            self.ws = WorldSpace(self.ss, self.a_map)
            self.world = World(self.ws)

        # All functions in World should raise NotImplementedError
        def test_NotImplementedGetReward(self):
            with self.assertRaises(NotImplementedError):
                self.world.GetReward((2,3,4))
            with self.assertRaises(NotImplementedError):
                self.world.GetReward((45,346,435))

        def test_NotImplementedNextState(self):
            with self.assertRaises(NotImplementedError):
                self.world.GetNextState((2,3,4), 2)
            with self.assertRaises(NotImplementedError):
                self.world.GetNextState((1,3,0), 9)

        def test_NotImplementedIsTerminal(self):
            with self.assertRaises(NotImplementedError):
                self.world.IsTerminal((2,3,4))
            with self.assertRaises(NotImplementedError):
                self.world.IsTerminal((1,3,0))

    unittest.main()
