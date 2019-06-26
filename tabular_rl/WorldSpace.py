import numpy as np

"""
Holds the state space and action space for a "world"
State space is an np array of size n, with each number i repr max value of dim i
Action map is a dict, where each value is an additive modification to a state
The keys are just
"""
class WorldSpace(object):
    def __init__(self, state_space, action_map):
        self._S = state_space
        self._action_map = action_map
        self._action_keys = { index: action for index, action in enumerate(self._action_map.keys()) }
        self._A = len(self._action_map.keys())

        # state_space = np.ones(shape=max_coords, dtype=np.int32) * DynamicNDWorld.NORMAL_STATE
        self._zero_state = np.array( [0 for _ in self._S] )
        self._final_state = np.array( [coord-1 for coord in self._S] )
        self._num_dims = len(self._S)

    def IsValidState(self, S):
        return np.all(S < self._final_state) and np.all(S >= self._zero_state)

    def IsValidAction(self, A):
        return A < self.A and A >= 0

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
            raise ValueError("Both key and index arguments cannot be None, at least one must be populated (index takes precedence)")
            
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
