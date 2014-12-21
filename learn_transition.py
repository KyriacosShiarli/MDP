from discretisationmodel import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataload import loadFile,loadFile2,loadFile3
from fg_gradient import adaboost_reg

def learn_tran():
	#examples = loadFile2()
	#examples2 = loadFile()
	examples3 = loadFile3("data/UPO/Experiments Folder/2014-11-17 11.08.31 AM/")
	examples4 = loadFile3("data/UPO/Experiments Folder/2014-11-28 01.22.03 PM/") 
	tot_examples  = examples3[:12] + examples4[:12]
	#tot_examples = examples + examples2
	disc = DiscModel()
	unorm_transition_f = np.zeros([disc.tot_actions,disc.tot_states,disc.tot_states])
	for example in tot_examples:
		for i in range(len(example.states)-1):
			state = disc.quantityToState(example.states[i])
			action = disc.actionToIndex(example.actions[i+1])
			next = disc.quantityToState(example.states[i+1])
			unorm_transition_f[action,state,next]+=1
	return unorm_transition_f

def learn_tran_regression(num_learners,tree_depth):
	#learns a transition function from data using adaboost.
	#------------------------------------------------------
	#Load all data
	examples = loadFile2()
	examples2 = loadFile()
	tot_examples = examples + examples2
	#Folds get generated here
	train_examples = tot_examples
	#test_examples = tot_examples[-2::]
	dimensions = examples[0].states.shape[1]
	#Build X which is the same for all regressors
	estimators = []	

	X =np.concatenate([np.hstack((example.states[:-1,:],example.actions[:-1,:])) for example in train_examples],axis = 0)
	y =np.concatenate([example.states[1:,:] for example in train_examples],axis = 0)


	estimators = [adaboost_reg(X,y[:,i],num_learners,tree_depth) for i in range(dimensions)]
	
	return estimators

def predict_next(state,action,estimators):
	sa = np.hstack((state,action))
	return np.concatenate([estimator.predict([sa]) for estimator in estimators])


def loss_augmentation(amount):
	examples = loadFile()
	disc = DiscModel()
	loss_aug = np.ones([disc.tot_states])*amount
	for example in examples:
		for step in range(len(example.states)):
			state = disc.quantityToState(example.states[step])
			action = disc.actionToIndex(example.actions[step])
			loss_aug[state] = 1
	return loss_aug

'''
examples_bad = loadFile2()
examples_good = loadFile()
disc = DiscModel()
feature_sums_good =np.zeros([13,len(examples_good)])
feature_sums_bad =np.zeros([13,len(examples_bad)])
for n,example in enumerate(examples_good):
	for i in range(100):
		feature_sums_good[:,n]+=disc.quantityToFeature(example["states"][i])

for n,example in enumerate(examples_bad):
	for i in range(100):
		feature_sums_bad[:,n]+=disc.quantityToFeature(example["states"][i])


fig = plt.figure()
ax = fig.add_subplot(111)

x = range(1,14)
for i in feature_sums_bad.T:
	h1 = ax.scatter(x,i,color = "red",alpha = 0.2)
for i in feature_sums_good.T:
	h2 = ax.scatter(x,i,color = "green",alpha = 0.2)
red_patch = mpatches.Patch(color='red', label='Bad',alpha = 0.5)
greed_patch = mpatches.Patch(color='green', label='Good',alpha = 0.5)
plt.legend([h1,h2],["bad","good"],bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
plt.show()
'''
if __name__ == "__main__":
	est = learn_tran_regression(20,10)
	state = np.array([1,2,0.1,0.4])
	action = np.array([0.3,0.4])
	out = predict_next(state,action,est)
	print out










