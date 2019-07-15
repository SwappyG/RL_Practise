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

		if self.policy.IsValidState(start_state):
			self.curr_state = start_state
		else:
			ERROR(f"start_state [{start_state}] was invalid")
			raise ValueError

		if not isinstance(policy, Policy):
			ERROR(f"policy must be of type {type(Policy)}, got [{type(policy)}]")
			raise TypeError

		self.ID = kwargs.get("ID", np.random.randint(2**31-1))
		self.is_training = kwargs.get("is_training", True)

		self.policy = policy
		# reqs = self.policy.PacketSizeReq()
		# self._req_S = reqs[0]
		# self._req_A = reqs[1]
		# self._req_R = reqs[2]

	# Use our policy to get the action
	def GetAction(self, S):
		return self.policy.GetAction(S)

	def ImprovePolicy(self, exp_packet):

		# Only improve policy if training flag is set
		if self.training:
			return self.policy.ImprovePolicy(pkt)
			# # If exp is deep enough, grab latest slice and call ImprovePolicy
			# if exp_packet.IsReqDepth(self._req_R, self._req_A, self._req_R):
			# 	pkt = exp_packet.GetLatestAsPacket(self._req_S, self._req_A, self._req_R)

		return False

	def UpdateCurrentState(self, new_state):
		if self.policy.IsValidState(new_state):
			self.curr_state = new_state
			return True

		return False


class TabularAgent(Agent):

	def __init__(self, policy, start_state, **kwargs):

		if not isinstance(policy, TabularPolicy):
			raise TypeError("policy argument must be of type TabularPolicy")

		Agent.__init__(self, policy, start_state, **kwargs)

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
