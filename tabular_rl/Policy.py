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
		self.gamma = 		kwargs.get("discount_factor")
		self.alpha = 		kwargs.pop("learn_rate")
		self.epsilon = 		kwargs.get("exploration_factor")
		self.is_static = 	kwargs.get("is_static")
		self.type = 		kwargs.get("value_type", Policy.STATE_VALUES)
		self.known_vals =   kwargs.get("known_state_vals", [])

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



class TabularPolicy(Policy):

	def __init__(self, world_space, **kwargs):
		self.init_var = kwargs.pop("init_variance")
		Policy.__init__(self, world_space, **kwargs)

		if self.type == Policy.STATE_VALUES:
			self.vals = np.random.normal(loc=0, scale=self.init_var, size=self._s_dim)
		elif self.type == Policy.ACTION_STATE_VALUES:
			self.vals = np.random.normal(loc=0, scale=self.init_var, size=np.append(self._s_dim, self._num_a) )
		else:
			raise ValueError("kwarg value_type is invalid")

	# Returns true if state is valid (enforces int/long for each dim in S)
	def IsValidState(self, S):

		# Check if every element in S is a whole number
		try:
			if isinstance(S, (int,long)) or \
				np.all( [isinstance(s, (int,long)) for s in S ] ):
				# Let the world_space determine validity if we have all ints/longs
				return self.world_space.IsValidState(S)

		except TypeError:
			WARN(f"Dimensions or type of S {S} are incorrect, got TypeError")

		return False

	# Returns the value of the specified indices
	def GetStateVal(self, indices):
		return self.vals[tuple(indices)]

	# Updates the value at specified indices with val given
	def UpdateStateVal(self, indices, val):
		self.vals[tuple(indices)] = val


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

		# Test that the TabularPolicy enforces discrete states and creates a value matrix for the state_space
		def test_TabularPolicy(self):
			self.p_kw['value_type'] = Policy.STATE_VALUES
			self.p_kw['init_variance'] = 0.01
			self.tab_pol = TabularPolicy(self.ws, **self.p_kw)
			self.assertTrue( np.all(self.tab_pol.vals.shape == self.ss) )

			self.assertTrue( self.tab_pol.IsValidState((0,0)) )
			self.assertTrue( self.tab_pol.IsValidState((1,3)) )
			self.assertFalse( self.tab_pol.IsValidState((0.1,0)) ) # non integer states are invalid for TabularPolicy
			self.assertFalse( self.tab_pol.IsValidState((3,2.3)) ) # non integer states are invalid for TabularPolicy
			self.assertFalse( self.tab_pol.IsValidState(2.3) ) # Make sure there's no index error
			self.assertFalse( self.tab_pol.IsValidState('a') ) # Make sure there's no index error or type error

			self.assertTrue( self.tab_pol.IsValidStateAction((1,3), 0) )
			self.assertFalse( self.tab_pol.IsValidStateAction((1,3.3), 0) ) # non integer states are invalid for TabularPolicy
			self.assertTrue( self.policy.IsValidStateAction((1,3.3), 0) ) # non integer states are valid for Policy

			self.p_kw['value_type'] = Policy.ACTION_STATE_VALUES
			self.tab_pol = TabularPolicy(self.ws, **self.p_kw)
			self.assertTrue( np.all(self.tab_pol.vals.shape == np.append(self.ss, len(self.a_map)) ) )


	unittest.main()
