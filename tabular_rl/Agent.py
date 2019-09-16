# -*- coding: future_fstrings -*-
import sys
import numpy as np
from Policy import Policy
from TabularPolicy import TabularPolicy
from ExpPacket import ExpPacket
from SarsaPolicy import SarsaPolicy

import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

class Agent(object):

	"""
	This is a base class for an RL Agent.

	It encapsulate an ID and a policy
	and provides functions to get an action, improve it's policy and update
	its current state

	Attributes:
		policy (Policy): defines the way this agent will behave. Must be a derived class of type Policy
		is_training (bool): flag to track if the current agent is static or training
		ID (int): unique number to identify this agent in a multi-agent Game
		curr_state (list): current state of this agent (must be valid based on WorldSpace inside Policy)
	"""

	def __init__(self, policy, start_state, **kwargs):

		"""
		Constructor for Agent base class

		Parameters:
			policy (Policy): defines the way this agent will behave. Must be a derived class of type Policy
			start_state (list): starting state of this agent (must be valid based on WorldSpace inside Policy)
			kwargs (dict):
				ID (int): unique number to identify this agent in a multi-agent Game
				is_training (bool): flag to track if the current agent is static or training

		Returns:
			Agent: A newly constructed Agent
		"""
		self._policy = policy

		if self._policy.IsValidState(start_state):
			self.curr_state = start_state
		else:
			ERROR(f"start_state [{start_state}] was invalid")
			raise ValueError

		if not isinstance(policy, Policy):
			ERROR(f"policy must be of type {type(Policy)}, got [{type(policy)}]")
			raise TypeError
	
		self._id = kwargs.get("ID", np.random.randint(2**31-1))
		self._is_training = kwargs.get("is_training", True)

	def IsTrainable(self):
		return self._is_training

	def SetTrainable(self, do_train):
		if do_train:
			self._is_training = True
		else:
			self._is_training = False

	def GetCurrState(self):
		"""Returns the current state of the agent"""
		return self.curr_state

	def GetID(self):
		return self._id

	def GetAction(self, S):
		"""
		Returns an action (list) for given state based on internal policy.

		Parameters:
		- S: [list] State to take action for (must be valid based on WorldSpace inside Policy)

		Returns: 
		- [int] int representing the action to take
		"""

		return self._policy.GetAction(S)

	def ImprovePolicy(self, exp_packet):
		"""
		Calls ImprovePolicy on the Agent's policy.

		Returns: (Bool) True if policy was improved, false otherwise
		"""

		# Only improve policy if training flag is set
		if self._is_training:
			return self._policy.ImprovePolicy(exp_packet)

		DEBUG("is_training is set to FALSE")
		return False

	def UpdateCurrentState(self, new_state):
		if self._policy.IsValidState(new_state):
			self.curr_state = new_state
			return True

		return False

	def PacketSizeReq(self):
		return self._policy.PacketSizeReq()

if __name__=="__main__":

	import unittest
	from collections import OrderedDict
	from WorldSpace import WorldSpace
	from ExpPacket import ExpPacket

	class TestAgent(unittest.TestCase):

		def setUp(self):
			self.ss = (7,9)

			self.a_map = OrderedDict()
			self.a_map['U'] = (1,1)
			self.a_map['D'] = (-1,-1)
			self.a_map['R'] = (1,0)
			self.a_map['L'] = (1,-2)

			self.p_kw = {}
			self.p_kw['discount_factor'] = 1
			self.p_kw['exploration_factor'] = 1
			self.p_kw['is_static'] = False
			self.p_kw['init_variance'] = 0.01

			self.ws = WorldSpace(self.ss, self.a_map)
			self._policy = SarsaPolicy(self.ws, **self.p_kw)

			self.agent = Agent(self._policy, (0,0))

		def test_ImprovePolicy(self):

			# make sure that policy can't be improved with empty packet
			self.assertFalse(self.agent.ImprovePolicy( ExpPacket() ))

			# improve the policy and check that the corresponding action is now selected
			pkt = ExpPacket([(0,0),(1,1)],[0, 0],[100])
			self.agent.ImprovePolicy( pkt )
			self.assertTrue(self.agent.GetAction((0,0)) == 0)

			# improve again, with diff action, and check that it is now the best
			pkt = ExpPacket([(0,0),(1,0)],[2, 2],[1000])
			self.agent.ImprovePolicy( pkt )
			self.assertTrue(self.agent.GetAction((0,0)) == 2)

			# Improve again, with negative reward for same action, so its no longer best
			pkt = ExpPacket([(0,0),(1,0)],[2, 2],[-10000])
			self.agent.ImprovePolicy(pkt)
			self.assertFalse(self.agent.GetAction((0,0)) == 2)

		def test_GetAction(self):
			self.assertTrue(self.agent.GetAction((0,0)) == self._policy.GetAction((0,0)))
			self.assertTrue(self.agent.GetAction((3,5)) == self._policy.GetAction((3,5)))

		def test_UpdateCurrentState(self):
			curr_state = self.agent.GetCurrState()
			self.agent.UpdateCurrentState((4,5))
			if self._policy.IsValidState((4,5)):
				self.assertTrue( np.all( self.agent.GetCurrState() == (4,5) ) )
			else:
				self.assertTrue( np.all( self.agent.GetCurrState() == curr_state ) )

			curr_state = self.agent.GetCurrState()
			self.agent.UpdateCurrentState(4)
			if self._policy.IsValidState(4):
				self.assertTrue( np.all( self.agent.GetCurrState() == 4 ) )
			else:
				self.assertTrue( np.all( self.agent.GetCurrState() == curr_state ) )

			curr_state = self.agent.GetCurrState()
			self.agent.UpdateCurrentState((4,6,3,3,6,7))
			if self._policy.IsValidState((4,6,3,3,6,7)):
				self.assertTrue( np.all( self.agent.GetCurrState() == (4,6,3,3,6,7) ) )
			else:
				self.assertTrue( np.all( self.agent.GetCurrState() == curr_state ) )


	unittest.main()
