import numpy as np
import math
import RFeatures as rf
from kinematicModels import staticGroupSimple
from Discretisation import *

class Model(object):
	def __init__(self,disc_model,reward_weights,kin_model= staticGroupSimple,actions = {"linear" : [0,0.05,0.25],"angular" : np.arange(-0.5,0.5,0.1)}):
		self.disc = disc_model
		self.actions = actions
		self.kinematics = kin_model
		self.w = reward_weights
		self.transition_f = self.buildTransitionFunction(self.disc)
		self.feature_f = self.buildFeatureFunction(self.disc)
		self.reward_f = self.buildRewardFunction()
	def buildTransitionFunction(self,model):
		tot_states =sum(model.states_per_angle)
		print tot_states
		tot_actions = len(self.actions["linear"])*len(self.actions["angular"])
		transition_f = np.zeros([tot_actions,tot_states,tot_states])
		for i in range(tot_states):
			for j in range(tot_actions):
				bins = model.stateToBins(i)
				for k in range(100):
					quantity = model.binsToQuantity(bins,sample = True)
			 		angular_idx = int(math.floor(j/len(self.actions["linear"])))
			 		linear_idx = int(j%len(self.actions["linear"]))
					action = [self.actions["linear"][linear_idx],self.actions["angular"][angular_idx]]
					next_quantity = self.kinematics(quantity,action)
					next_bins = model.quantityToBins(next_quantity)
					next_state = model.binsToState(next_bins)
					transition_f[j,i,next_state] += 1
		row_sums = np.sum(transition_f,axis=2)
		#TODO Make nans failsafe
		transition_f = transition_f/row_sums[:,np.newaxis]
		return transition_f
			#print i,bins
	def buildFeatureFunction(self,model):
		tot_states = sum(model.states_per_angle)
		for i in range(tot_states):
			quantity = model.binsToQuantity(model.stateToBins(i))
			features = rf.staticGroupSimple(quantity[1],quantity[0]) #Still hacky. Need to find a solution to his
			if i == 0: feature_f = np.zeros([tot_states,len(features)])
			feature_f[i,:] = features
		return feature_f
	def buildRewardFunction(self):
		reward_f = np.zeros(self.feature_f.shape[0])
		for n,i in enumerate(self.feature_f):
			reward_f[n] = np.sum(self.w * i)
		return reward_f


if __name__ == "__main__":

	d = DiscModel()
	actions = {
		"linear" : [0,0.05,0.25],
		"angular" : np.arange(-0.5,0.5,0.1)
	}
	m = Model(d,staticGroupSimple,actions,[-1,-1])
	rew = m.buildRewardFunction()
	#print m.transition_f[8,1:20,1:20]
	st = [1,4,0.0,0.0]
	action = [0.5,0.44]
	out = m.kinematics(st,action)
	print out
