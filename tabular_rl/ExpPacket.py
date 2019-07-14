class ExpPacket(object):
	def __init__(self, S_list, A_list, R_list, n_step):
		self._S_list = S_list
		self._A_list = A_list
		self._R_list = R_list
		self._n_step = n_step
		self._len_A = len(A_list)
		self._len_S = len(S_list)
		self._len_R = len(R_list)

	def __init__(self):
		self._S_list = []
		self._A_list = []
		self._R_list = []
		self._n_step = 1

	def Get(self):
		return self._S_list, self._A_list, self._R_list, self._n_step

	def Push(self, S, A, R):
		self._S_list.push_back(S)
		self._A_list.push_back(A)
		self._R_list.push_back(R)
		self._len_S += 1
		self._len_A += 1
		self._len_R += 1

	def Trainable(self):
		return  self._len_S > self._n_step + 1 and \
				self._len_A > self._n_step + 1 and \
				self._len_R > self._n_step + 1

	def LenA(self):
		return self._len_A

	def LenS(self):
		return self._len_S

	def LenR(self):
		return self._len_R
