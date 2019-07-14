# -*- coding: future_fstrings -*-
import sys
import numpy as np
from Policy import Policy, TabularPolicy
import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class SarsaPolicy(TabularPolicy):

	def __init__(self, world, **kwargs):
		params = {}
		params["discount_factor"] = kwargs.pop("discount_factor", 1)
		params["exploration_factor"] = kwargs.pop("exploration_factor", 0.95)
		params["is_static"] = kwargs.pop("is_static", False)
		params["init_variance"] = kwargs.pop("init_variance", 0.01)
		params["value_type"] = Policy.ACTION_STATE_VALUES

		if len(kwargs) > 0:
			raise ValueError("Got unknown args in kwargs passed to SarsaPolicy")

		TabularPolicy.__init__(self, world, **params)
		DEBUG("Initialized SarsaPolicy instance")

	# Returns a selection of equally valid actions
	def _GetActions(self, S, eps=1):

		# occasionally select a random action
		if (np.random.rand() > eps):
			selection = range(self._num_a)

		# get a set of actions tied for the max value for this state
		else:
			vals = self.vals[S]
			selection = [index for index, val in enumerate(vals) if val == np.max(vals)]

		return selection

	def GetStateVal(self, S, A):
		return super(type(self), self).GetStateVal( np.append(S,A) )

	# Picks a random action from a selection
	def GetAction(self, S):
		# Grab a selection of equal valued actions and return a random one
		selection = self._GetActions(S, eps=self.epsilon)
		return np.random.choice(selection)

	# Returns the estimated value of (S[0],A[0]) pair based on exp in packet
	def GetTargetEstimate(self, packet):
		S_list, A_list, R_list, n = packet.Get()
		try:
			next_val = self.GetStateVal(S_list[1], A_list[1])
		except IndexError as e:
			ERROR(f"Bad ExpPacket sent to GetTargetEstimate. \nS_list and A_list must be at least length 2\nEach S,A pair must be valid. Got:\nS: {S_list}\nA: {A_list}")
			raise IndexError
		return R_list[0] + self.gamma*next_val

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

	# Wrapper around TabularPolicy.UpdateState, taking S, A as seperate args
	def UpdateState(self, S, A, val):
		if self.IsValidStateAction(S, A):
			super(type(self), self).UpdateStateVal( np.append(S,A), val )
			return True
		else:
			WARN(f"Invalid (S, A) pair, got S: {S} and A: {A}")
			return False


if __name__=="__main__":

	import unittest
	from collections import OrderedDict
	from WorldSpace import WorldSpace
	from ExpPacket import ExpPacket

	class TestSarsaPolicy(unittest.TestCase):

		def setUp(self):
			self.ss = (7,9)

			self.a_map = OrderedDict()
			self.a_map['U'] = (1,1)
			self.a_map['D'] = (-1,-1)
			self.a_map['R'] = (1,0)
			self.a_map['L'] = (1,-2)

			self.p_kw = {}
			self.p_kw['discount_factor'] = 1
			self.p_kw['exploration_factor'] = 0.95
			self.p_kw['is_static'] = False
			self.p_kw['init_variance'] = 0.01

			self.ws = WorldSpace(self.ss, self.a_map)
			self.policy = SarsaPolicy(self.ws, **self.p_kw)

		def test_UpdateState(self):
			self.assertTrue(self.policy.UpdateState( (0,0), 0, 42)) # Check that we can update
			self.assertTrue(self.policy.vals[(0,0,0)] == 42) # Check that it updates correctly

			self.assertFalse(self.policy.UpdateState( (0,-1), 0, 32)) # Check that it rejects bad (S,A)
			self.assertFalse(self.policy.UpdateState( (0,1), 8, 22)) # Check that it rejects bad (S,A)

		def test_GetActionExploitation(self):
			self.policy.epsilon = 1 # guarentee exploitation

			self.policy.UpdateState( (0,0), 0, 100 ) # Set the value of [(0,0) , 0] to something high
			[self.assertTrue(self.policy.GetAction((0,0)) == 0) for _ in range(500)] # Repeat 500x to make sure the action returned is always 0 (100% exploitation)

			self.policy.UpdateState( (0,0), 2, 100 ) # Set the value of [(0,0) , 2] to match [(0,0) , 0]
			actions = np.array([self.policy.GetAction((0,0)) for _ in range(100)])

			# Make sure that not every action is either 0 or 2
			self.assertFalse( np.all(actions == 0) )
			self.assertFalse( np.all(actions == 2) )

			# Make sure at least one action is 0, and at least one is 2
			self.assertTrue( np.any(actions == 0) )
			self.assertTrue( np.any(actions == 2) )

			# Make sure no actions are 1 or 3
			self.assertFalse( np.any(actions == 1) )
			self.assertFalse( np.any(actions == 3) )


		def test_GetActionExploration(self):

			self.policy.epsilon = 0.9 # Add some exploration
			self.policy.UpdateState( (0,0), 1, 1000 ) # Set the value of [(0,0) , 1] to something higher
			actions = np.array([self.policy.GetAction((0,0)) for _ in range(5000)])

			# Make sure there's at least one action from each of the other choices
			self.assertFalse( np.all(actions == 1) )
			self.assertTrue( np.any(actions == 0) )
			self.assertTrue( np.any(actions == 2) )
			self.assertTrue( np.any(actions == 3) )

		def test_GetTargetEstimate(self):

			s_list = [(0,0), (1,2)]
			a_list = [0, 1]
			r_list = [-3, -5]
			n_step = 1
			self.packet = ExpPacket(s_list, a_list, r_list, n_step)
			G = self.policy.GetTargetEstimate(self.packet)
			self.assertTrue(G == -3 + 1 * self.policy.GetStateVal((1,2), 1))

			s_list = [(0,34), (1,343)]
			a_list = [0, 1]
			r_list = [-3, -5]
			n_step = 1
			self.packet = ExpPacket(s_list, a_list, r_list, n_step)
			with self.assertRaises(IndexError):
				self.policy.GetTargetEstimate(self.packet)


		def test_GetProbabilityOfAction(self):
			eps = 0.85372
			self.policy.epsilon = eps # Add some exploration
			self.policy.UpdateState( (0,0), 1, 1000 ) # Set the value of [(0,0) , 1] to something high

			# Fuzzy equals check since computer digitization errors
			self.assertAlmostEqual( self.policy.GetProbabilityOfAction((0,0), 1), eps + (1-eps)/4 )
			self.assertAlmostEqual( self.policy.GetProbabilityOfAction((0,0), 0), (1-eps)/4)
			self.assertAlmostEqual( self.policy.GetProbabilityOfAction((0,0), 2), (1-eps)/4)
			self.assertAlmostEqual( self.policy.GetProbabilityOfAction((0,0), 3), (1-eps)/4)

			self.policy.UpdateState( (0,0), 2, 1000 ) # Set the value of [(0,0) , 2] to something equally high
			self.assertAlmostEqual( self.policy.GetProbabilityOfAction((0,0), 1), eps/2 + (1-eps)/4 )
			self.assertAlmostEqual( self.policy.GetProbabilityOfAction((0,0), 2), eps/2 + (1-eps)/4 )
			self.assertAlmostEqual( self.policy.GetProbabilityOfAction((0,0), 0), (1-eps)/4)
			self.assertAlmostEqual( self.policy.GetProbabilityOfAction((0,0), 3), (1-eps)/4)

	unittest.main()
