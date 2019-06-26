# -*- coding: future_fstrings -*-

import sys
import numpy as np

class Policy(object):

	STATE_VALUES = 0
	ACTION_STATE_VALUES = 1

	def __init__(self, world, **kwargs):

		self.world_space = 	world_space
		self.gamma = 		kwargs.get("discount_factor", 1)
		self.epsilon = 		kwargs.get("exploration_factor", 0.95)
		self.is_static = 	kwargs.get("static", True)
		self.type = 		kwargs.get("value_type", Policy.STATE_VALUES)
		self.init_var = 	kwargs.get("init_variance", 0.01)
		self.known_vals =   kwargs.get("known_state_vals", [])

		self._s_dim = self.world_space.GetSDims()
		self._num_a = self.world_space.GetNumA()

		if self.type == Policy.STATE_VALUES:
			self.vals = np.random.normal(loc=0, scale=self.init_var, shape=self._s_dim)
		elif self.type == Policy.ACTION_STATE_VALUES:
			self.vals = np.random.normal(loc=0, scale=self.init_var, np.append(s_dim, self._num_a) )
		else:
			raise ValueError("kwarg value_type is invalid")



	def GetAction(self, S):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def GetProbabilityOfAction(self, A):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def GetTargetEstimate(self, packet, n_step=None):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	# def IsValidState(self, S):
	# 	return self.world.IsValidState(S)
	#
	# def IsValidAction(self, A):
	# 	return self.world.IsValidAction(A)
	#
	# def IsValidStateAction(self, S, A):
	# 	return self.world.IsValidStateAction(S, A)

class SarsaPolicy(Policy):

	def __init__(self, world, **kwargs):
		params = {}
		params["discount_factor"] = kwargs.get("discount_factor", 1)
		params["exploration_factor"] = kwargs.get("exploration_factor", 0.95)
		params["static"] = kwargs.get("static", False)
		params["value_type"] = Policy.ACTION_STATE_VALUES
		Policy.__init__(self, world, params)

	# Returns a selection of equally valid actions
	def _GetActions(self, S, eps=1):

		# occasionally select a random action
		if (np.random.rand() < eps):
			selection = range(self._num_a)

		# get a set of actions tied for the max value for this state
		else:
			vals = self.vals[S, :]
			selection = [index for index, val in enumerate(vals) if v == np.max(vals)]

		return selection

	# Picks a random action from a selection
	def GetAction(self, S):
		# Grab a selection of equal valued actions
		selection = self._GetActions(S, eps=self.epsilon)

		# Pick randomly from our selection and return
		return np.random.choice(selection)

	# Returns the estimated value of (S[0],A[0]) pair based on exp in packet
	def GetTargetEstimate(self, packet):
		S_list, A_list, R_list, n = packet.Get()
		try:
			next_val = self.vals[ S_list[1], A_list[1] ]
		except IndexError as e:
			print(f"Bad ExpPacket sent to GetTargetEstimate. S_list and A_list must be at least length 2, their values must be valid a valid S,A pair. \nGot S: {S_list},\nA: {A_list}")
			raise IndexError
		return R[0] + self.gamma*next_val

	def GetProbabilityOfAction(self, S, A):
		# Get the probability of selecting A by exploring, P(A|S,exploit) = eps/num_a
		prob_exploration = (1-self.epsilon)/self._num_a

		# Get the probability of selecting A by exploiting, P(A,not_best|S,explore) = 0
		selection = self._GetActions(S,eps=1)
		if A in selection:
			prob_exploitation = self.epsilon / len(selection)
			return prob_exploration + prob_exploitation
		else:
			return prob_exploration

	# Updates value of state in policy as given
	def UpdateState(self, S, A, val):
		if self.IsValidStateAction(S, A):
			self.vals[S, A] = val
			return True
		else:
			print f"Invalid (S, A) pair, got S: {S} and A: {A}"
			return False
