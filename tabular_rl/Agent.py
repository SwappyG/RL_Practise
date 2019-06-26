class Agent(object):

	def __init__(self, policy, start_state, **kwargs):
		self.policy = policy
		self.ID = kwargs.get("ID", np.random.randint(2**32-1))
		self.training = kwargs.get("training", True)
        self.curr_state = start_state

	def GetAction(self, S):
		return policy.GetAction(S)

	def ImprovePolicy(self, exp_packet):
		if self.training:
			S_list, A_list, R_list, n = exp_packet.Get()
			V = self.policy.vals[S_list[0], A_list[0]]
			G = self.policy.GetTargetEstimate(exp_packet)
            new_val = (1-self.alpha) * V + self.alpha * G
			self.policy.UpdateState(S_list[0], A_list[0], new_val)
			return True
        else:
			return False

    def UpdateCurrentState(self, new_state):
        self.curr_state = new_state
