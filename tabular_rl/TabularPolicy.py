# -*- coding: future_fstrings -*-

import sys
import numpy as np

from Policy import Policy

import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

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
			if isinstance(S, (int,long)):
				return self.world_space.IsValidState(S)

			if np.all( [isinstance(s, (int,long)) for s in S ] ):
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

	class TestTabularPolicy(unittest.TestCase):

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

			self.p_kw['value_type'] = Policy.STATE_VALUES
			self.p_kw['init_variance'] = 0.01
			self.tab_pol = TabularPolicy(self.ws, **self.p_kw)

		# Test that the TabularPolicy enforces discrete states and creates a value matrix for the state_space
		def test_TabularPolicy(self):

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
