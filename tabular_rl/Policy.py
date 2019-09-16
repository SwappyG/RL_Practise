# -*- coding: future_fstrings -*-

import sys
import numpy as np

import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

class Policy(object):

	STATE_VALUES = 0
	ACTION_STATE_VALUES = 1

	def __init__(self, world_space, **kwargs):

		self.world_space = 	world_space
		self.gamma = 		kwargs.pop("discount_factor")
		self.alpha = 		kwargs.pop("learn_rate")
		self.epsilon = 		kwargs.pop("exploration_factor")
		self.is_static = 	kwargs.pop("is_static")
		self.type = 		kwargs.pop("value_type", Policy.STATE_VALUES)
		self.known_vals =   kwargs.pop("known_state_vals", [])

		if len(kwargs) > 0:
			raise KeyError(f"Received Unexpected keys in kwargs, {kwargs}")

		self._s_dim = self.world_space.GetSDims()
		self._num_a = self.world_space.GetNumA()

	def GetAction(self, S):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def GetProbabilityOfAction(self, A):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def GetTargetEstimate(self, packet, n_step=None):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def GetStateVal(self, S):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def ImprovePolicy(self, packet):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def PacketSizeReq(self):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def IsValidPacket(self, packet):
		raise NotImplementedError(f'{sys._getframe().f_code.co_name} must be implemented by derived class of class: {self.__class__.__name__}')

	def IsValidState(self, S):
		return self.world_space.IsValidState(S)

	def IsValidAction(self, A):
		return self.world_space.IsValidAction(A)

	def IsValidStateAction(self, S ,A):
		return self.IsValidState(S) and self.IsValidAction(A)


if __name__=="__main__":

	import unittest
	from collections import OrderedDict
	from WorldSpace import WorldSpace
	from ExpPacket import ExpPacket

	class TestPolicy(unittest.TestCase):

		def setUp(self):
			self.ss = (7,9)

			self.a_map = OrderedDict()
			self.a_map['U'] = (1,1)
			self.a_map['D'] = (-1,-1)
			self.a_map['R'] = (1,0)
			self.a_map['L'] = (1,-2)

			self.ws = WorldSpace(self.ss, self.a_map)

			self.p_kw = {}

			self.p_kw['discount_factor'] = 1
			self.p_kw['exploration_factor'] = 0.95
			self.p_kw['is_static'] = False
			self.p_kw['learn_rate'] = 0.001

			self.policy = Policy(self.ws, **self.p_kw)

		# Test if IsValidState, IsValidAction and IsValidStateAction are good
		def test_Init(self):
			self.assertTrue(self.policy.IsValidState((0,0)))
			self.assertTrue(self.policy.IsValidState((0,1)))
			self.assertTrue(self.policy.IsValidState((1,0)))
			self.assertTrue(self.policy.IsValidState((1.4356534,4.3)))
			self.assertFalse(self.policy.IsValidState((-1.4356534,4.3)))
			self.assertFalse(self.policy.IsValidState((7,9)))
			self.assertFalse(self.policy.IsValidState((0, -1)))

			self.assertTrue(self.policy.IsValidAction(0))
			self.assertFalse(self.policy.IsValidAction(5))
			self.assertFalse(self.policy.IsValidAction(-1))
			with self.assertRaises(TypeError):
				self.policy.IsValidAction(1.2)

			s = self.policy.IsValidState((0,0))
			a = self.policy.IsValidAction(0)
			s_a = self.policy.IsValidStateAction((0,0), 0)
			self.assertEqual( s_a, s and a )

			s = self.policy.IsValidState((1,5))
			a = self.policy.IsValidAction(7)
			s_a = self.policy.IsValidStateAction((1,5), 7)
			self.assertEqual( s_a, s and a )

			s = self.policy.IsValidState((-1,5))
			a = self.policy.IsValidAction(7)
			s_a = self.policy.IsValidStateAction((-1,5), 2 )
			self.assertEqual( s_a, s and a )

		# Raise NotImplementedError for rest of functions
		def test_NotImplementedGetAction(self):
			with self.assertRaises(NotImplementedError):
				self.policy.GetAction((0,0))
			with self.assertRaises(NotImplementedError):
				self.policy.GetAction((-1, 3, 67, 7.4))

		def test_NotImplementedGetProbabilityOfAction(self):
			with self.assertRaises(NotImplementedError):
				self.policy.GetProbabilityOfAction(55)
			with self.assertRaises(NotImplementedError):
				self.policy.GetProbabilityOfAction(0.5)

		def test_NotImplementedGetTargetEstimate(self):
			with self.assertRaises(NotImplementedError):
				self.policy.GetTargetEstimate( ExpPacket([], [], []) )
			with self.assertRaises(NotImplementedError):
				self.policy.GetTargetEstimate( ExpPacket([(0,0),(3,1),(8,1)], [0,2,1], [-1,-1,0]) )

		def test_NotImplementedImprovePolicy(self):
			with self.assertRaises(NotImplementedError):
				self.policy.ImprovePolicy( ExpPacket([], [], []) )

		def test_NotImplementedIsValidPacket(self):
			with self.assertRaises(NotImplementedError):
				self.policy.IsValidPacket( ExpPacket([], [], []) )

		def test_NotImplementedPacketSizeReq(self):
			with self.assertRaises(NotImplementedError):
				self.policy.PacketSizeReq()

		def test_MismatchedKwargs(self):
			self.p_kw = {}

			self.p_kw['discount_factor'] = 1
			self.p_kw['exploration_factor'] = 0.95
			self.p_kw['is_static'] = False

			# Check for keyerror with too few kwargs
			with self.assertRaises(KeyError):
				self.policy = Policy(self.ws, **self.p_kw)

			self.p_kw['learn_rate'] = 0.001
			self.p_kw['random_kwarg'] = 42

			# Check for keyerror with too many kwargs
			with self.assertRaises(KeyError):
				self.policy = Policy(self.ws, **self.p_kw)

	unittest.main()
