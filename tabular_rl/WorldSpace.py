# -*- coding: future_fstrings -*-
import numpy as np

"""
Holds the state space and action space for a "world"
State space is an np array of size n, with each number i repr max value of dim i
Action map is a dict, where each value is an additive modification to a state
The keys are just
"""
class WorldSpace(object):
    def __init__(self, state_space, action_map):
        self._S = np.array(state_space)
        self._action_map = action_map
        self._action_keys = { index: action for index, action in enumerate(self._action_map.keys()) }
        self._A = len(self._action_map.keys())

        # state_space = np.ones(shape=max_coords, dtype=np.int32) * DynamicNDWorld.NORMAL_STATE
        self._zero_state = np.array( [0 for _ in self._S] )
        self._final_state = np.array( [coord-1 for coord in self._S] )
        self._num_dims = len(self._S)

        for k, v in self._action_map.items():
            if not len(v) == self._num_dims:
                raise ValueError(f"Each action in action_map must have same dims as state_space. Action {v} does not.")

    def IsValidState(self, S):
        return np.all(np.array(S) < self._S) and np.all(np.array(S) >= self._zero_state)

    def IsValidAction(self, A):
        if not isinstance(A, (int,long)): raise TypeError("Action must be an int or long")
        return A < self._A and A >= 0

    def IsValidStateAction(self, S, A):
        return self.IsValidAction(A) and self.IsValidState(S)

    def GetSDims(self):
        return self._S

    def GetNumA(self):
        return self._A

    def ActionKey(self, index):
        return self._action_keys[index]

    def ActionVal(self, index=None, key=None):
        if not index == None:
            return self._action_map[ self.ActionKey(index) ]
        elif not key == None:
            return self._action_map[ key ]
        else:
            raise TypeError("Both key and index arguments cannot be None, at least one must be populated (index takes precedence)")

    def ActionVals(self):
        return self._action_map

    def ActionsIndices(self):
        return self._action_keys

    def ZeroState(self):
        return self._zero_state

    def LastState(self):
        return self._final_state

    def StateDims(self):
        return self._num_dims

if __name__=="__main__":

    import unittest
    from collections import OrderedDict

    class TestWorldSpace(unittest.TestCase):

        def setUp(self):
            self.ss = (9,7)
            self.a_map = OrderedDict()
            self.a_map['U'] = (1,1)
            self.a_map['D'] = (-1,-1)
            self.a_map['R'] = (1,0)
            self.a_map['L'] = (1,-2)
            self.ws = WorldSpace(self.ss, self.a_map)

        def test_Init(self):
            try:
                scrap_ws = WorldSpace((3,4,5), self.a_map)
            except ValueError:
                pass
            except:
                self.fail("Didn't get ValueError when expected")

        # Check that we get the right values back from our get functions
        def test_get(self):
            self.assertItemsEqual(self.ss, self.ws.GetSDims(), 'got incorrect s dims')
            self.assertEqual(len(self.a_map.keys()), self.ws.GetNumA(), 'got incorrect a nums')
            self.assertItemsEqual(self.a_map, self.ws.ActionVals(), 'got incorrect a action_map')

        # Test IsValidState to make sure it works with lists/tuples/nparrays
        def test_IsValidStateTrue(self):
            self.assertTrue(self.ws.IsValidState((3,4)), 'IsValidState tuple failed')
            self.assertTrue(self.ws.IsValidState((0,0)), 'IsValidState tuple failed')
            self.assertTrue(self.ws.IsValidState((2,1)), 'IsValidState tuple failed')
            self.assertTrue(self.ws.IsValidState([7,1]), 'IsValidState list failed')
            self.assertTrue(self.ws.IsValidState(np.array((3,4))), 'IsValidState nparray failed')

        # Check some false states, as well as dim_mismatch raising ValueError
        def test_IsValidStateFalse(self):
            self.assertFalse(self.ws.IsValidState((0,-1)), 'IsValidState failed')
            self.assertFalse(self.ws.IsValidState((-1,0)), 'IsValidState failed')
            self.assertFalse(self.ws.IsValidState((9,1,)), 'IsValidState failed')
            with self.assertRaises(ValueError):
                self.ws.IsValidState((4,7,5))

        # Check IsValidAction, make sure its a singleton and TypeError is raised otherwise
        def test_IsValidAction(self):
            self.assertFalse(self.ws.IsValidAction(6), 'IsValidAction failed')
            self.assertFalse(self.ws.IsValidAction(-1), 'IsValidAction failed')
            self.assertTrue(self.ws.IsValidAction(3), 'IsValidAction failed')
            with self.assertRaises(TypeError):
                self.ws.IsValidAction((0,0))

        # Check IsValidStateAction
        def test_IsValidStateAction(self):
            self.assertTrue(self.ws.IsValidStateAction([3,4],1), 'IsValidStateAction failed')
            self.assertFalse(self.ws.IsValidStateAction([3,4],5), 'IsValidAction failed')
            with self.assertRaises(TypeError):
                self.ws.IsValidStateAction([3,4], (0,))

        # Make sure we get the right key back from ActionKey, and KeyError raised if A is invalid
        def test_ActionKey(self):
            self.assertEqual(self.a_map.keys()[0], self.ws.ActionKey(0), 'got incorrect action key')
            self.assertEqual(self.a_map.keys()[2], self.ws.ActionKey(2), 'got incorrect action key')
            self.assertRaises(KeyError, self.ws.ActionKey, 4)

        # Make sure we get the right key back from ActionKey
        # Check that KeyError raised if A is invalid
        # Check that TypeError is raised if A is not a singleton
        def test_ActionVal(self):
            self.assertEqual(self.a_map.values()[0], self.ws.ActionVal(index=0), 'got incorrect action val from index')
            self.assertEqual(self.a_map.values()[2], self.ws.ActionVal(index=2), 'got incorrect action val from index')
            self.assertEqual(self.a_map['D'], self.ws.ActionVal(key='D'), 'got incorrect action val from index')
            with self.assertRaises(KeyError):
                self.ws.ActionVal(key='X')
            with self.assertRaises(TypeError):
                self.ws.ActionVal()

        # Check that the zero and final states are set correctly
        def test_ZeroFinal(self):
            self.assertTrue(np.all((0,0) == self.ws.ZeroState()), 'got incorrect zero state')
            self.assertTrue(np.all((8,6) == self.ws.LastState()), 'got incorrect zero state')

        # Check that we get the right value back from StateDims
        def test_StateDims(self):
            self.assertEqual(2, self.ws.StateDims(), 'got incorrect state dims')

    unittest.main()
