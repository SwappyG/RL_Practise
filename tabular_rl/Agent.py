# -*- coding: future_fstrings -*-
import sys
import numpy as np
from Policy import Policy, TabularPolicy
from ExpPacket import ExpPacket
from SarsaPolicy import SarsaPolicy

import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

class Agent(object):

	def __init__(self, policy, start_state, **kwargs):
		self.policy = policy
		self.ID = kwargs.get("ID", np.random.randint(2**31-1))
		self.is_training = kwargs.get("is_training", True)
		self.curr_state = start_state

	# Use our policy to get the action
	def GetAction(self, S):
		return self.policy.GetAction(S)

	def ImprovePolicy(self, exp_packet):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def UpdateCurrentState(self, new_state):
		self.curr_state = new_state

class TabularAgent(Agent):

	def __init__(self, policy, start_state, **kwargs):

		if not isinstance(policy, TabularPolicy):
			raise TypeError("policy argument must be of type TabularPolicy")

		Agent.__init__(self, policy, start_state, **kwargs)

	def ImprovePolicy(self, exp_packet):

		# Only improve policy if training flag is set
		if self.training:

			# Grab elements out of our exp_packet
			S_list, A_list, R_list, n = exp_packet.Get()

			# grab the current value of the state

			V = self.policy.GetStateVal(S_list[0], A_list[0])

			# Calculate the new target based on the exp_packet
			G = self.policy.GetTargetEstimate(exp_packet)

			# Increment towards the new target based on learning rate
			new_val = (1-self.alpha) * V + self.alpha * G

			# Update the value of the state and return True
			self.policy.UpdateState(S_list[0], A_list[0], new_val)
			return True

		# Training flag not set, return false
		else:
			return False

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
			self.policy = SarsaPolicy(self.ws, **self.p_kw)

			self.agent = Agent(self.policy, (0,0))

		def test_NotImplentedImprovePolicy(self):
			with self.assertRaises(NotImplementedError):
				self.agent.ImprovePolicy( ExpPacket([],[],[],1) )

		def test_GetAction(self):
			self.assertTrue(self.agent.GetAction((0,0)) == self.policy.GetAction((0,0)))
			self.assertTrue(self.agent.GetAction((3,5)) == self.policy.GetAction((3,5)))

		def test_UpdateCurrentState(self):
			self.agent.UpdateCurrentState((4,5))
			self.assertTrue( np.all( self.agent.curr_state == (4,5) ) )

			self.agent.UpdateCurrentState(4)
			self.assertTrue( np.all( self.agent.curr_state == 4 ) )

			self.agent.UpdateCurrentState((-4,-7))
			self.assertTrue( np.all( self.agent.curr_state == (-4,-7) ) )

			self.agent.UpdateCurrentState((4,5,6,8,3))
			self.assertTrue( np.all( self.agent.curr_state == (4,5,6,8,3) ) )

	unittest.main()
