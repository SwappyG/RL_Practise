# -*- coding: future_fstrings -*-
import sys
import numpy as np

import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

from TabularRLUtils import PadList

class ExpPacket(object):

	MAX_SIZE = 2**15-1

	"""Init with list of S, A and R"""
	def __init__(self, S_list=[], A_list=[], R_list=[]):
		self._S_list = list(S_list)
		self._A_list = list(A_list)
		self._R_list = list(R_list)

	def __str__(self):
		
		# Pad S_list or A_list to make them the same length
		S = []
		A = []
		R = []
		if self.LenS() > self.LenA():
			A = PadList(self._A_list, self.LenS(), None)
			S = self._S_list
		elif self.LenS() < self.LenA():
			S = PadList(self._S_list, self.LenA(), None )
			A = self._A_list
		else: 
			S = self._S_list
			A = self._A_list

		# Pad R_list (or S_list and A_list) to make all same length
		if self.LenS() > self.LenR():
			R = PadList(self._R_list, self.LenS(), None)
		elif self.LenS() < self.LenS():
			S = PadList(self._S_list, self.LenR(), None)
			A = PadList(self._A_list, self.LenR(), None)
			R = self._R_list
		else: 
			R = self._R_list

		# Create the string to return by zipping S,A,R and adding line breaks
		ret_str = ""
		for triplet in zip(S,A,R):
			ret_str = ret_str + f"{triplet}\n"
		
		return ret_str

	def Get(self):
		"""Returns list of S, A and R"""
		return (self._S_list, self._A_list, self._R_list)

	def GetLatest(self, nS, nA, nR):
		"""
		Returns:
		[list] - len nS, nA and nR of latest S, A and R, [None] if not sufficient depth
		Returns None if nS, nA or nR exceed size of S, A or R, respectively
		"""
		if nS > self.LenS() or nA  > self.LenA() or nR > self.LenR():
			raise IndexError(f"nS {nS}, nA {nA} and nR {nR} must be <= LenS {self.LenS()}, LenA {self.LenA()}, LenR {self.LenR()}, respectively")

		ret_val = (	self._S_list[ -nS : ],
					self._A_list[ -nA : ],
					self._R_list[ -nR : ]	)
		return ret_val

	def GetLatestAsPacket(self, nS, nA, nR):
		"""
		Same as GetLatest, but returns a new ExpPacket with sublists instead
		
		Returns:
		[ExpPacket] if a valid one can be generated, [None] otherwise
		"""

		try:
			latest = self.GetLatest(nS,nA,nR)
		except IndexError:
			return None

		sl, al, rl = latest
		return ExpPacket(sl, al, rl)
	
	def Push(self, S, A, R):
		"""
		Pushes a set of S, A and R onto their respective lists
		
		Parameters:
			S (list/int): State (caller must ensure S is valid for their use case)
			A (list/int): Action (caller must ensure A is valid for their use case)
			R (list/int): Reward (caller must ensure R is valid for their use case)
		
		Returns:
			True if success, False if lists are already full
		"""

		if self.LenS() >= ExpPacket.MAX_SIZE or \
			self.LenA() >= ExpPacket.MAX_SIZE or \
			self.LenR() >= ExpPacket.MAX_SIZE:
			return False

		self._S_list.append(self._ToTuple(S))
		self._A_list.append(self._ToTuple(A))
		self._R_list.append(self._ToTuple(R))

		return True

	def IsReqDepth(self, s_size=0, a_size=0, r_size=0):
		"""Returns true if this packet has the specified depth of data"""
		return self.LenS() >= s_size and \
				self.LenA() >= a_size and \
				self.LenR() >= r_size

	def LenA(self):
		"""Returns the current length of the list A"""
		return len(self._A_list)

	def LenS(self):
		"""Returns the current length of the list S"""
		return len(self._S_list)

	def LenR(self):
		"""Returns the current length of the list R"""
		return len(self._R_list)

	def _ToTuple(self, val):
		# Try to convert lists, ndarrays, etc to tuple
		try:
			return tuple(val)

		# If we get type error, just return the val without conversion
		except TypeError:
			return val

if __name__=="__main__":

	import unittest
	from collections import OrderedDict
	from WorldSpace import WorldSpace
	from ExpPacket import ExpPacket

	class TestAgent(unittest.TestCase):

		def setUp(self):
			self.sl = [2,3,4]
			self.al = [0,0,1]
			self.rl = [-1,-1]
			self.pckt_f = ExpPacket( self.sl, self.al, self.rl ) # filled packet
			self.pckt_e = ExpPacket()	# Empty packet
			self.pckt_max = ExpPacket( np.zeros(ExpPacket.MAX_SIZE),
										np.zeros(ExpPacket.MAX_SIZE),
										np.zeros(ExpPacket.MAX_SIZE) ) # Max packet

		def test_InitAndLen(self):

			# Test that constructor with args works and Len is correct
			self.sl = [2,3,4]
			self.al = [0,0,1]
			self.rl = [-1,-1]
			_ = ExpPacket( self.sl, self.al, self.rl )
			self.assertTrue(self.pckt_f.LenS() == 3)
			self.assertTrue(self.pckt_f.LenA() == 3)
			self.assertTrue(self.pckt_f.LenR() == 2)

			# Test that constructor with no args works and Len is correct
			self.assertTrue(self.pckt_e.LenS() == 0)
			self.assertTrue(self.pckt_e.LenA() == 0)
			self.assertTrue(self.pckt_e.LenR() == 0)

			# Test that constructor accepts diff types of lists
			ExpPacket( [2,3,4], (0,0,1), np.array((-1,-1)) )

		def test_Push(self):

			# Test that we can push diff types of ints, floats and lists
			self.assertTrue(self.pckt_f.Push(2,6,1))
			self.assertTrue(self.pckt_f.Push(1.1,0.35,-2.32))
			self.assertTrue(self.pckt_f.Push( (1,3) , [0, 5.3] , np.array([-2,-2.21,-1]) ))

			# Make sure Len is still correct
			self.assertTrue(self.pckt_f.LenS() == 6)
			self.assertTrue(self.pckt_f.LenA() == 6)
			self.assertTrue(self.pckt_f.LenR() == 5)

			# Check that we get false when trying to push into a max packet
			self.assertFalse(self.pckt_max.Push(2,6,1))

		def test_Get(self):

			# Make sure we can get the list in packet correctly
			sl, al, rl = self.pckt_f.Get()
			self.assertTrue( np.all( sl == self.sl ) )
			self.assertTrue( np.all( al == self.al ) )
			self.assertTrue( np.all( rl == self.rl ) )

			# Push some new exp into packet
			self.pckt_f.Push(2,6,1)
			self.sl.append(2)
			self.al.append(6)
			self.rl.append(1)

			# Make sure we can get the list in packet correctly after pushing
			sl, al, rl = self.pckt_f.Get()
			self.assertTrue( np.all( sl == self.sl ) )
			self.assertTrue( np.all( al == self.al ) )
			self.assertTrue( np.all( rl == self.rl ) )

			# Make sure we can grab the latest packets properly
			sl, al, rl = self.pckt_f.GetLatest(2,2,2)
			self.assertTrue( np.all( sl == self.sl[-2:] ) )
			self.assertTrue( np.all( al == self.al[-2:] ) )
			self.assertTrue( np.all( rl == self.rl[-2:] ) )

			# Grab the curr length of the lists
			slen = self.pckt_f.LenS()
			alen = self.pckt_f.LenA()
			rlen = self.pckt_f.LenR()

			# Make sure GetLatest works with Len_() funcs
			sl, al, rl = self.pckt_f.GetLatest(slen, alen, rlen)

			# Make sure we get an IndexError if we try to access Latest > Num_Avail
			with self.assertRaises(IndexError):
				sl, al, rl = self.pckt_f.GetLatest(slen+1, alen, rlen)

			with self.assertRaises(IndexError):
				self.pckt_f.GetLatest(slen, alen+1, rlen)

			with self.assertRaises(IndexError):
				self.pckt_f.GetLatest(slen, alen, rlen+1)

		def test_IsReqDepth(self):

			# Make sure IsReqDepth reports true when indices <= max for each list
			self.assertTrue( self.pckt_f.IsReqDepth(1,1,1) )
			self.assertTrue( self.pckt_f.IsReqDepth(2,2,2) )
			self.assertTrue( self.pckt_f.IsReqDepth(3,3,2) )

			# Make sure IsReqDepth reports false if ANY index is > max for that list
			self.assertFalse( self.pckt_f.IsReqDepth(1,1,4) )
			self.assertFalse( self.pckt_f.IsReqDepth(1,6,1) )
			self.assertFalse( self.pckt_f.IsReqDepth(8,1,1) )

		def test_GetLatestAsPacket(self):

			# Make sure we get back an ExpPacket
			self.assertTrue( isinstance( self.pckt_f.GetLatestAsPacket(1,1,1), self.pckt_f.__class__))

			# Get the sub packet of slice of latest exp
			sub_packet = self.pckt_f.GetLatestAsPacket(1,2,1)
			self.assertTrue( sub_packet.LenS() == 1 )
			self.assertTrue( sub_packet.LenA() == 2 )
			self.assertTrue( sub_packet.LenR() == 1 )

			# Get the lists from the subpack and compare to sub_lists from original packet
			sl, al, rl = sub_packet.Get()
			sl_orig, al_orig, rl_orig = self.pckt_f.GetLatest(1,2,1)
			self.assertTrue( sl == sl_orig )
			self.assertTrue( al == al_orig )
			self.assertTrue( rl == rl_orig )

		def test_Print(self):
			try:
				print(self.pckt_f)
			except Exception as e:
				self.fail(f"Encountered error {e}")

	unittest.main()
