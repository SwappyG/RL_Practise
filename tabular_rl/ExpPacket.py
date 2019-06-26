class ExpPacket(object):
	def __init__(self, S_list, A_list, R_list, n_step):
		self._S_list = S_list
		self._A_list = A_list
		self._R_list = R_list
		self._n_step = n_step
	def Get(self):
		return self._S_list, self._A_list, self._R_list, self._n_step
