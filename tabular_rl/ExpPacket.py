# -*- coding: future_fstrings -*-
import sys
import numpy as np

import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

class ExpPacket(object):

	MAX_SIZE = 2**15-1

	# Init with list of S, A and R
	def __init__(self, S_list=[], A_list=[], R_list=[]):
		self._S_list = np.array(S_list)
		self._A_list = np.array(A_list)
		self._R_list = np.array(R_list)

	# Returns list of S, A and R
	def Get(self):
		return (self._S_list, self._A_list, self._R_list)

	# Returns lists of len nS, nA and nR of latest S, A and R
	# Returns None if nS, nA or nR exceed size of S, A or R, respectively
	def GetLatest(self, nS, nA, nR):
		try:
			ret_val = (	S_list[ -nS : ],
						A_list[ -nA : ],
						R_list[ -nR : ]	)
			return ret_val
		except IndexError:
			return None

	# Same as GetLatest, but returns a new ExpPacket with sublists instead
	# Returns None if nS, nA or nR are invalid
	def GetLatestAsPacket(self, nS, nA, nR):
		latest = self.GetLatest(nS,nA,nR)
		if latest == None:
			return None

		sl, al, rl = latest
		return ExpPacket(sl, al, rl)

	# Pushes a set of S, A and R onto their respective lists
	# Returns True if success, False if lists are already full
	def Push(self, S, A, R):
		if self.LenS() >= ExpPacket.MAX_SIZE or \
			self.LenA() >= ExpPacket.MAX_SIZE or \
			self.LenR() >= ExpPacket.MAX_SIZE:
			return False

		np.append(self._S_list, S)
		np.append(self._A_list, A)
		np.append(self._R_list, R)

		return True

	# Returns true if this packet has the specified depth of data
	def IsReqDepth(self, s_size=0, a_size=0, r_size=0):
		return self.LenS() >= s_size and \
				self.LenA() >= a_size and \
				self.LenR() >= r_size

	# Returns the current length of the lists
	def LenA(self):
		return len(self._A_list)

	def LenS(self):
		return len(self._S_list)

	def LenR(self):
		return len(self._R_list)


if __name__=="__main__":

	import unittest
	from collections import OrderedDict
	from WorldSpace import WorldSpace
	from ExpPacket import ExpPacket

	class TestAgent(unittest.TestCase):

		def setUp(self):
			self.pckt_f = ExpPacket( [2,3,4], [0,0,1], [-1,-1] ) # filled packet
			self.pckt_e = ExpPacket()	# Empty packet
			self.pckt_max = ExpPacket( np.zeros(ExpPacket.MAX_SIZE),
										np.zeros(ExpPacket.MAX_SIZE),
										np.zeros(ExpPacket.MAX_SIZE) ) # Max packet

		def test_InitAndLen(self):

			# Test that constructor with args works and Len is correct
			self.sl = [2,3,4]
			self.al = [0,0,1]
			self.rl = [-1,-1]
			full_pckt = ExpPacket( self.sl, self.al, self.rl )
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
			print self.pckt_f.LenS()
			self.assertTrue(self.pckt_f.LenS() == 6)
			self.assertTrue(self.pckt_f.LenA() == 6)
			self.assertTrue(self.pckt_f.LenR() == 5)

			# Check that we get false when trying to push into a max packet
			self.assertFalse(self.pckt_max.Push(2,6,1))

		def test_Get(self):
			sl, al, rl = self.pckt_f.Get()
			self.assertTrue( np.all( sl == self.sl ) )
			self.assertTrue( np.all( al == self.al ) )
			self.assertTrue( np.all( rl == self.rl ) )

			self.pckt_f.Push(2,6,1)
			self.sl.push_back(2)
			self.al.push_back(6)
			self.rl.push_back(1)

			self.assertTrue( np.all( sl == self.sl ) )
			self.assertTrue( np.all( al == self.al ) )
			self.assertTrue( np.all( rl == self.rl ) )

		def test_IsReqDepth(self):
			self.assertTrue( self.pckt_f.IsReqDepth(1,1,1) )
			self.assertTrue( self.pckt_f.IsReqDepth(2,2,2) )
			self.assertTrue( self.pckt_f.IsReqDepth(3,3,2) )
			self.assertFalse( self.pckt_f.IsReqDepth(1,1,4) )
			self.assertFalse( self.pckt_f.IsReqDepth(1,6,1) )
			self.assertFalse( self.pckt_f.IsReqDepth(8,1,1) )

		# def test_GetLatest(self):

	unittest.main()
