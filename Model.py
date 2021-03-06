import numpy as np
import math
from discretisationmodel import *
import time
import scipy.spatial as spat
from dataload import extract_info
from kinematics import staticGroupSimple,staticGroupSimple2
from learn_transition import learn_tran,loss_augmentation,learn_correction,predict_next

class Transition(object):
	def __init__(self):
		self.forward = []
		self.backward = []
		self.tot_states = 0
		self.tot_actions = 0

class Model(object):
	def __init__(self,discretisation,reward_weights):
		self.disc = discretisation#discretisation model
		#self.kinematics = kin_model
		self.w = reward_weights
		self.buildFeatureFunction()
		self.buildTransitionFunction(10, learn=False)

		self.buildRewardFunction()
	def buildTransitionFunction(self,bin_samples,learn =True):
		def transition_smart():
			self.transition = Transition()
			tot_states=self.transition.tot_states = transition_f.shape[1]
			tot_actions=self.transition.tot_actions =transition_f.shape[0]
			self.transition.backward = [{} for j in xrange(tot_states*tot_actions)]
			self.transition.forward = [{} for j in xrange(tot_states*tot_actions)]
			idx = np.transpose(np.array(np.nonzero(transition_f)))
			for i in idx:
				self.transition.backward[i[0] + i[1]*tot_actions][str(i[2])] = transition_f[i[0],i[1],i[2]]
				self.transition.forward[i[0] + i[2]*tot_actions][str(i[1])] = transition_f[i[0],i[1],i[2]]
			print self.transition.backward
		#print "Got Here estemated"	
		transition_f = np.zeros([self.disc.tot_actions,self.disc.tot_states,self.disc.tot_states])
		transtion_test = np.zeros([self.disc.tot_actions,self.disc.tot_states,self.disc.tot_states])
		estimators = learn_correction(10,10)
		for i in xrange(self.disc.tot_states):
			for j in xrange(self.disc.tot_actions):
				bins = self.disc.stateToBins(i)	
				for k in xrange(bin_samples):
					assert bin_samples != 0
					if bin_samples==1:
						samp = False
					else:
						samp = True
					quantity = self.disc.binsToQuantity(bins,sample = samp)
					action = self.disc.indexToAction(j)
					correction = predict_next(quantity,action,estimators)
					correction[0] = np.arcsin(correction[0])*2
					next_quantity = staticGroupSimple2(quantity,action) - correction
					#next_quantity2 = staticGroupSimple2(quantity,action)
					#if np.sum(next_quantity[:1]-next_quantity2[:1])>1:
					#	print "Quantity",quantity
					#	print "ACTION",action
					#	print next_quantity[:2],next_quantity2[:2]
					next_state = self.disc.quantityToState(next_quantity)
					#next2 = self.disc.quantityToState(next_quantity2)
					#print "state",ic
					#print "actions",j
					transition_f[j,i,next_state] += 1
		if learn ==True:
			trans = learn_tran()
			transition_f =np.add( transition_f ,2* trans)
		row_sums = np.sum(transition_f,axis=2)
		transition_f = transition_f/row_sums[:,:,np.newaxis]
		self.transition_f = transition_f
		transition_smart()
	def buildFeatureFunction(self):
		for i in xrange(self.disc.tot_states):
			state_quantity = self.disc.stateToQuantity(i)
			for j in xrange(self.disc.tot_actions):
				action_quantity = self.disc.indexToAction(j)
				features = self.disc.quantityToFeature(state_quantity,action_quantity) #Still hacky. Need to find a solution to his
				if i == 0 and j == 0: feature_f = np.zeros([self.disc.tot_actions,self.disc.tot_states,len(features)])
				feature_f[j,i,:] = features
				#print features
	 	self.feature_f = feature_f
	def buildRewardFunction(self):
		self.reward_f= np.dot(self.feature_f,self.w)
		#l = loss_augmentation(0.7)
		#self.reward_f_la= self.reward_f * l
	def actionSimilarity(self):
		all_actions = []
		#maxdiff = sum([max(self.actions["linear"]-min(actions["linear"])),max(actions["angular"]-min(actions["angular"]))])
		for i in range(self.disc.tot_actions):
			action = self.disc.indexToAction(i)
			all_actions.append(action)
		dist = spat.distance.squareform(spat.distance.pdist(all_actions,"euclidean"))
		maxx = np.amax(dist)
		s = 0.1
		t = 0.001
		k = np.exp(-dist**2/s**2)
		k[k<t]=0
		self.action_similarity = k

#TODO MAKE NON LINEAR MODEL INHERIT FROM THE OTHER ONE. OR MAKE A MORE BASIC VERSION OF THE MODEL FROM WHICH LINEAR AND NON LINEAR INHERIT
class Model_non_linear(object):
	def __init__(self,discretisation):
		self.disc = discretisation#discretisation model
		#self.kinematics = kin_model
		self.buildTransitionFunction(1,learn=True)
		self.buildFeatureFunction()
		self.buildRewardFunction()
		self.actionSimilarity()
	def buildTransitionFunction(self,bin_samples,learn =True):
		def transition_smart():
			self.transition = {}
			tot_states=self.transition["tot_states"] = transition_f.shape[1]
			tot_actions=self.transition["tot_actions"] =transition_f.shape[0]
			self.transition["backward"] = [{} for j in xrange(tot_states*tot_actions)]
			self.transition["forward"] = [{} for j in xrange(tot_states*tot_actions)]
			idx = np.transpose(np.array(np.nonzero(transition_f)))
			for i in idx:
				self.transition["backward"][i[0] + i[1]*tot_actions][str(i[2])] = transition_f[i[0],i[1],i[2]]
				self.transition["forward"][i[0] + i[2]*tot_actions][str(i[1])] = transition_f[i[0],i[1],i[2]]

		transition_f = np.zeros([self.disc.tot_actions,self.disc.tot_states,self.disc.tot_states])
		for i in xrange(self.disc.tot_states):
			for j in xrange(self.disc.tot_actions):
				bins = self.disc.stateToBins(i)
				for k in xrange(bin_samples):
					assert bin_samples != 0
					if bin_samples==1:
						samp = False
					else:
						samp = True
					quantity = self.disc.binsToQuantity(bins,sample = samp)

					action = self.disc.indexToAction(j)
					next_quantity = staticGroupSimple(quantity,action)
					next_state = self.disc.quantityToState(next_quantity)
					transition_f[j,i,next_state] += 1
		if learn ==True:
			trans = learn_tran()
			transition_f =np.add( transition_f , trans)
		row_sums = np.sum(transition_f,axis=2)
		transition_f = transition_f/row_sums[:,:,np.newaxis]
		self.transition_f = transition_f
		transition_smart()
	def buildFeatureFunction(self):
		counter = 0
		for i in xrange(self.disc.tot_states):
			state_quantity = self.disc.stateToQuantity(i)
			for j in xrange(self.disc.tot_actions):
				action_quantity = self.disc.indexToAction(j)
				features = self.disc.quantityToFeature(state_quantity,action_quantity) #Still hacky. Need to find a solution to this
				if i == 0 and j == 0: feature_f = np.zeros([self.disc.tot_actions,self.disc.tot_states,len(features)])
				feature_f[j,i,:] = features
				print features
				if features[-1]==1:
					counter+=1
					print counter
		print counter
	 	self.feature_f = feature_f
	def buildRewardFunction(self):
		self.reward_f = -1*np.ones([self.disc.tot_actions,self.disc.tot_states])
	def actionSimilarity(self):
		all_actions = []
		#maxdiff = sum([max(self.actions["linear"]-min(actions["linear"])),max(actions["angular"]-min(actions["angular"]))])
		for i in range(self.disc.tot_actions):
			action = self.disc.indexToAction(i)
			all_actions.append(action)
		dist = spat.distance.squareform(spat.distance.pdist(all_actions,"euclidean"))
		maxx = np.amax(dist)
		s = 0.1
		t = 0.001
		k = np.exp(-dist**2/s**2)
		k[k<t]=0
		self.action_similarity = k




if __name__ == "__main__":
	w =[-1.,-0.6,-1.8,-1.5,-1.3,-1.2,-1.,-1.,-1.,-1.,-0.7,-1.,-1.,-1,-1,-1.,-1.,-1,-1,-1.,-1.,-0.6,-1,-1]
	d = DiscModel()
	m = Model(d,w)
	#for i in m.feature_f:
		#print i
	#print d.states_per_angle
	#print m.transition_backward
	#print m.transition_forward
	#print rew
	#print m.transition_f[8,1:30,1:30]
	#st = [1,4,0.0,0.0]
	#action = [0.5,0.44]
	#out = m.kinematics(st,action)
	#print out
