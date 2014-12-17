from discretisationmodel import *
from forwardBackward import *
from dataload import *
import numpy as np
from itertools import compress
from plott import *
from Model import Model_non_linear
from Model import Model
from RFeatures import continouous
import math
from fg_gradient import adaboost_reg

#initialise models

disc_model = DiscModel()
w =[-1.,-0.2,-4.,-1.,-0.3,-1.,-1.,-1.,-0.3,-2.,-1.,-0.5,-1.,-3.]
model = Model(disc_model,w)
trans = model.transition
steps = 5
# Initial reward function

#model.reward_f = np.random.uniform(-2,-0.1,model.reward_f.shape)
r_initial = model.reward_f
examples,distribution = extract_info(disc_model,steps,dist = True)
policy_ref,lg=caus_ent_backward(model.transition,model.reward_f,examples[1]["end_state"],steps)
start_states=[example["start_state"] for example in examples]
state_freq_ref,state_action_frequencies_ref= forward_sa(policy_ref,model.transition,start_states,steps)
iterations = 300


#initialise reward model
feat = {"function":continouous,"inputs":None}
disc_model = DiscModel(feature = feat)
model = Model_non_linear(disc_model)
model.transition = trans
model.reward_f = np.zeros(model.reward_f.shape)
model.reward_f += r_initial
model.reward_f[1,:]-=0.5
#model.reward_f = r_initial
actions,states,features = model.feature_f.shape
for itera in xrange(iterations):
	policy_test,lg=caus_ent_backward(model.transition,model.reward_f,examples[1]["end_state"],steps)
	state_freq_test,state_action_frequencies_test= forward_sa(policy_test,model.transition,start_states,steps)
	reward_diff = np.sum(np.sum(np.absolute(model.reward_f - r_initial)))
	policy_diff = np.sum(np.sum(np.absolute(policy_test-policy_ref)))
	print "Difference in Reward --->", reward_diff
	print "Difference in Policy --->", policy_diff
	X= np.reshape(model.feature_f,(disc_model.tot_states*disc_model.tot_actions,4))
	Y = (state_action_frequencies_ref - state_action_frequencies_test).reshape((disc_model.tot_states*disc_model.tot_actions))
	#X = X[np.nonzero(Y)[0]]
	#Y = Y[np.nonzero(Y)[0]]
	
	print np.nonzero(Y)[0].shape
	print Y.shape
	print "starting adaboost"
	ht = adaboost_reg(X,Y,100,10)
	print "finished adaboost"
	print "reducing gradient"
	model.reward_f = (model.reward_f.reshape(states*actions)*np.exp(-1.*ht.predict(model.feature_f.reshape(states*actions,features)))).reshape(actions,states)	
	print "finished gradient"

def sign(x):
	if x!=0:
		return math.copysign(1,x)
	else:
		return 0

#Y = np.array(map(int,map(sign,(avg_counts - state_action_frequencies).reshape((disc_model.tot_states*disc_model.tot_actions)))))









	