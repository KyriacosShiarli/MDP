from discretisationmodel import *
from forwardBackward import *
from dataload import *
import numpy as np
from itertools import compress
from plott import *
from Model import Model_non_linear
from RFeatures import continouous
import math
from fg_gradient import adaboost_reg


feat = {"function":continouous,"inputs":None}

disc_model = DiscModel(feature = feat)
model = Model_non_linear(disc_model)
examples,distribution = extract_info(disc_model,10,dist = True)
policy,lg=caus_ent_backward(model.transition,model.reward_f,examples[1]["end_state"],10)
start_states=[example["start_state"] for example in examples]
state_freq,state_action_frequencies= forward_sa(policy,model.transition,start_states,10)
avg_counts = np.zeros([disc_model.tot_actions,disc_model.tot_states])

for example in examples:
	avg_counts+=example["counts"]
avg_counts/=len(examples)

X= np.reshape(model.feature_f,(disc_model.tot_states*disc_model.tot_actions,4))
Y = np.zeros([disc_model.tot_states*disc_model.tot_actions])

def sign(x):
	if x!=0:
		return math.copysign(1,x)
	else:
		return 0

#Y = np.array(map(int,map(sign,(avg_counts - state_action_frequencies).reshape((disc_model.tot_states*disc_model.tot_actions)))))
Y = (avg_counts - state_action_frequencies).reshape((disc_model.tot_states*disc_model.tot_actions))
X = X[np.nonzero(Y)[0]]
Y = Y[np.nonzero(Y)[0]]

ht = adaboost_reg(X,Y,200,4)

for i in range(disc_model.tot_states):
	for j in range(disc_model.tot_actions):
		print "Before--->", model.reward_f[j,i]
		model.reward_f[j,i] = model.reward_f[j,i]*np.exp(-0.001*ht.predict([model.feature_f[j,i]]))	
		print "After---->",model.reward_f[j,i] 
print np.amax(model.reward_f)





	