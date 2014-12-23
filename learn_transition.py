from discretisationmodel import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataload import loadFile,loadFile2,loadFile3
from fg_gradient import adaboost_reg
from kinematics import staticGroupSimple2
from functions import *

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


def get_dataset(examples):
	next_kin = []
	next_data = []
	for example in examples:
		for n, step in enumerate(example.states[:-1]):
			next_kin .append(staticGroupSimple2(step,example.actions[n+1],0.033))
			next_data.append(example.states[n+1])
			#if np.sin((np.absolute(np.array(next_kin)[-1,0]-np.array(next_data)[-1,0]))/2) >0.1:
			#	print np.array(next_kin)[-1,:]
			#	print np.array(next_data)[-1,:]
			#	print step,example.actions[n+1]
			#	print example.states[n+2]
			#	print '-------------------------------------------'
	next_kin = np.array(next_kin)
	next_data = np.array(next_data)
	diff = next_kin-next_data
	diff[:,0] = np.sin(diff[:,0]/2)
	#simple filter
	for i in range(len(diff[:,0])):
		if np.absolute(diff[i,0]) >0.2:
			diff[i,0] = 0
	X =np.concatenate([np.hstack((example.states[:-1,:],example.actions[1:,:])) for example in examples],axis = 0)
	return X,diff	

def learn_correction(num_learners,tree_depth):
	#learns a transition function from data using adaboost.
	#------------------------------------------------------
	#Load all data
	examples = loadFile3("data/UPO/Experiments Folder/2014-11-17 11.08.31 AM/")
	examples2 = loadFile3("data/UPO/Experiments Folder/2014-11-28 01.22.03 PM/") 
	tot_examples = examples[0:12] + examples2[0:12]
	
	#Folds get generated here
	train_examples = tot_examples[0::]
	test_examples = tot_examples[0:5]
	X_train,diff_train = get_dataset(train_examples)
	X_test,diff_test = get_dataset(test_examples)
	#plt.show()
	#test_examples = tot_examples[-2::]
	dimensions = examples[0].states.shape[1]
	print 'Dimentions',dimensions
	#Build X which is the same for all regressors
	#y =np.concatenate([example.states[1:,:] for example in train_examples],axis = 0)
	estimators = [adaboost_reg(X_train,diff_train[:,i],num_learners,tree_depth) for i in range(dimensions)]
	fit = []
	#for example in train_examples:
	#	for n, step in enumerate(example.states[:-1]):
	#		fit.append(predict_next(step,exax]mple.actions[n+1],estimators))
	fit = np.array([estimator.predict(X_test) for estimator in estimators])
	print fit.shape
	#fit[:,0] =np.arcsin(fit[:,0])*2
	x = range(diff_test.shape[0])
	plt.plot(x,diff_test[:,0]) 
	plt.plot(x,fit[0,:],color='red',alpha = 0.6)
	plt.show()
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
	#est = learn_tran_regression(20,10)
	#state = np.array([1,2,0.1,0.4])
	#action = np.array([0.3,0.4])
	#out = predict_next(state,action,est)
	#print out
	est = learn_correction(300,10)










