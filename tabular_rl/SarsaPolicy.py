# -*- coding: future_fstrings -*-
import sys
import numpy as np
from Policy import Policy, TabularPolicy
from ExpPacket import ExpPacket

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
		params["learn_rate"] = kwargs.pop("learn_rate", 0.001)
		params["value_type"] = Policy.ACTION_STATE_VALUES

		self._req_S = 2
		self._req_A = 2
		self._req_R = 1

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

	# Picks a random action from a selection
	def GetAction(self, S):
		# Grab a selection of equal valued actions and return a random one
		selection = self._GetActions(S, eps=self.epsilon)
		return np.random.choice(selection)

	# Returns the estimated value of (S[0],A[0]) pair based on exp in packet
	def GetTargetEstimate(self, packet):
		S_list, A_list, R_list = packet.Get()
		try:
			next_val = self.GetStateVal(S_list[1], A_list[1])
		except IndexError as e:
			ERROR(f"Bad ExpPacket sent to GetTargetEstimate. \nS_list and A_list must be at least length 2\nEach S,A pair must be valid. Got:\nS: {S_list}\nA: {A_list}")
			raise IndexError
		return R_list[0] + self.gamma*next_val

	# Returns the prob of performing given A in state S
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

	# Returns the value of the given [S,A] pair
	def GetStateVal(self, S, A):
		return super(type(self), self).GetStateVal( np.append(S,A) )

	# Wrapper around TabularPolicy.UpdateState, taking S, A as seperate args
	def UpdateState(self, S, A, val):
		if self.IsValidStateAction(S, A):
			super(type(self), self).UpdateStateVal( np.append(S,A), val )
			return True
		else:
			WARN(f"Invalid (S, A) pair, got S: {S} and A: {A}")
			return False

	# Returns the min req for [S,A,R] in packets sent to ImprovePolicy
	def PacketSizeReq(self):
		return (self._req_S, self._req_A, self._req_R)

	# Check whether packet has sufficient [S,A,R] for learning
	def IsValidPacket(self, packet):
		if not isinstance(packet, ExpPacket):
			return False

		return packet.IsReqDepth(self._req_R, self._req_A, self._req_R)

	def ImprovePolicy(self, packet):
		if not IsValidPacket(packet):
			return False

		S_list, A_list, R_list, n = exp_packet.Get() # Grab elements out of our exp_packet

		V = self.GetStateVal(S_list[0], A_list[0]) # grab the current value of the state
		G = self.GetTargetEstimate(exp_packet) # Calculate the new target based on the exp_packet
		new_val = (1-self.alpha) * V + self.alpha * G # Increment towards the new target based on learning rate

		self.UpdateState(S_list[0], A_list[0], new_val) # Update the value of the state and return True

		return True


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
			self.p_kw['learn_rate'] = 0.001

			self.ws = WorldSpace(self.ss, self.a_map)
			self.policy = SarsaPolicy(self.ws, **self.p_kw)

		def test_UpdateStateGetState(self):
			self.assertTrue(self.policy.UpdateState( (0,0), 0, 42)) # Check that we can update
			self.assertAlmostEqual(self.policy.vals[(0,0,0)], 42) # Check that it updates correctly
			self.assertAlmostEqual(self.policy.GetStateValue( (0,0), 0 ), 42) # Make sure we can get value back properly
			self.assertNotEqual(self.policy.GetStateValue( (0,0), 1 ), 42) # Make sure neighbouring states didn't update
			self.assertNotEqual(self.policy.GetStateValue( (0,1), 0 ), 42) # Make sure neighbouring states didn't update
			self.assertNotEqual(self.policy.GetStateValue( (1,0), 0 ), 42) # Make sure neighbouring states didn't update
			self.assertNotEqual(self.policy.GetStateValue( (1,1), 0 ), 42) # Make sure neighbouring states didn't update
			self.assertNotEqual(self.policy.GetStateValue( (1,1), 1 ), 42) # Make sure neighbouring states didn't update

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
			self.packet = ExpPacket(s_list, a_list, r_list)
			G = self.policy.GetTargetEstimate(self.packet)
			self.assertTrue(G == -3 + 1 * self.policy.GetStateVal((1,2), 1))

			s_list = [(0,34), (1,343)]
			a_list = [0, 1]
			r_list = [-3, -5]
			self.packet = ExpPacket(s_list, a_list, r_list)
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

		def test_Packets(self):
			self.assertTrue( (self._req_S, self._req_A, self._req_R) == self.policy.PacketSizeReq() )

			pkt = ExpPacket([],[],[])
			self.assertFalse(self.policy.IsValidPacket(pkt)) # depth of 0, not good enough
			pkt.Push( (0,0), 0, -1 )
			self.assertFalse(self.policy.IsValidPacket(pkt)) # Depth of 1, not good enough
			pkt.Push( (1,1), 2, -1 )
			self.assertTrue(self.policy.IsValidPacket(pkt)) # Has depth of 2, should be good now

			self.assertFalse(self.policy.IsValidPacket([])) # Make sure it returns False instead of throwing TypeError

		def test_ImprovePolicy(self):
			pkt = ExpPacket([],[],[])

			old_val = self.policy.GetStateValue((0,0), 0)
			val_1_1_2 = self.policy.GetStateValue((1,1), 2)
			alpha = self.policy.alpha
			gamma = self.policy.gamma

			self.assertFalse(self.policy.ImprovePolicy(pkt)) # Empty packet, not enough depth to improve policy
			self.assertFalse(self.policy.ImprovePolicy([])) # Make sure it returns False instead of throwing TypeError
			pkt.Push( (0,0), 0, -1 )
			self.assertFalse(self.policy.ImprovePolicy(pkt)) # Depth of 1, not good enough
			pkt.Push( (1,1), 2, -1 )
			self.assertTrue(self.policy.ImprovePolicy(pkt)) # Depth of 1, not good enough

			new_val = self.policy.GetStateValue((0,0), 0)
			self.assertNotEqual(old_val, new_val) # Make sure (0,0), 0 got updated
			self.assertAlmostEqual(new_val, (1-alpha)*old_val + alpha*(-1 + gamma*val_1_1_2) ) # make sure it follows the SARSA update



	unittest.main()
