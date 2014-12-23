from dataload import *
from kinematics import *
from discretisationmodel import *
import matplotlib.pyplot as plt
from forwardBackward import *
from Model import Model
from learn_transition import learn_tran,loss_augmentation,learn_correction,predict_next

def plotNextState():
	m = DiscModel()
	next_data=np.zeros(2)
	next_kinematics1 = np.zeros(2)
	next_kinematics2 = np.zeros(2)
	examples = extract_info(m,"Full")
	print len(examples)
	for example in examples:
		for j in range(len(example.states)):		
			next_data = np.vstack([next_data,example.states[j]])
			next_kinematics1 = np.vstack([next_kinematics1,staticGroupSimple2(example.states[j-1],example.actions[j-1],0.1)])
			
			next_kinematics2 = np.vstack([next_kinematics2,staticGroupSimple2(example.states[j-1],example.actions[j])])
	print next_data.shape
	x = range(next_data.shape[0])
	next_data = np.array(next_data)
	print len(next_data[:,0])
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].scatter(x,next_data[:,1],alpha = 0.7,label = "data")
	plt.scatter(x,next_kinematics1[:,1],color = "red",alpha = 0.3)
	axarr[0].scatter(x,next_kinematics2[:,1],color = "green",alpha = 0.2,label = "kinematic")
	axarr[0].legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)

	axarr[1].scatter(x,next_data[:,0],alpha = 0.7,)
	plt.scatter(x,next_kinematics1[:,1],color = "red",alpha = 0.3)
	axarr[1].scatter(x,next_kinematics2[:,0],color = "green",alpha = 0.2)

	axarr[1].set_ylabel("Angle/rad")
	axarr[0].set_ylabel("Distance")
	axarr[1].set_xlabel("Example")
	plt.show()

def plotDrift(idx,startpoint,estimators):
	m = DiscModel()
	examples = extract_info(m,"Full","good")
	
	example = examples[idx]
 	print example.states.shape
	states_kin = np.array(example.states[startpoint])
	states_kin2 =np.array(example.states[startpoint])
	for n,i in enumerate(example.actions[startpoint:-1]):
		if n == 0:
			nex = predict_next(states_kin,i,estimators)
			nex[0] = np.arcsin(nex[0])*2
			states_kin = np.vstack([states_kin,staticGroupSimple2(states_kin,i)-nex])

			states_kin2 = np.vstack([states_kin2,staticGroupSimple2(states_kin2,i)])
		elif np.absolute(example.states[startpoint+n][0] -example.states[startpoint + n-1][0])>0.1:
			states_kin = np.vstack([states_kin,example.states[startpoint+n]])	 
			states_kin2 = np.vstack([states_kin2,example.states[startpoint+n]])	 
		else:
			nex = predict_next(states_kin[n,:],example.actions[n+1],estimators)
			nex[0] = np.arcsin(nex[0])*2
			states_kin = np.vstack([states_kin,staticGroupSimple2(states_kin[n,:],example.actions[n+1])-nex])
			states_kin2 = np.vstack([states_kin2,staticGroupSimple2(states_kin2[n,:],example.actions[n-1])])
	ex = np.array(example.states[startpoint::])
	x = range(len(ex))
	f, axarr = plt.subplots(2,sharex = True)
	axarr[0].scatter(x,ex[:,1],color = "blue",label = "data")
	axarr[0].scatter(x,states_kin[:,1],color = "green",alpha = 0.4, label = "kinematic")
	axarr[0].scatter(x,states_kin2[:,1],color = "red",alpha = 0.2, label = "kinematic2")
	axarr[0].legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)

	axarr[1].scatter(x,ex[:,0],color = "blue")
	axarr[1].scatter(x,states_kin[:,0],color = "green", alpha = 0.4)
	axarr[1].scatter(x,states_kin2[:,0],color = "red", alpha = 0.2)

	axarr[1].set_ylabel("Angle/rad")
	axarr[0].set_ylabel("Distance")
	axarr[1].set_xlabel("Example")
	#plt.show()

def plot_reg_drift(idx,startpoint):
	m = DiscModel()
	examples = extract_info(m,"Full","good")
	example = examples[idx]
 	estimators = learn_tran_regression(300,4)
	states_kin = np.array(example.states[startpoint])
	for n,i in enumerate(example.actions[startpoint:-1]):
		if n == 0:
			states_kin = np.vstack([states_kin,staticGroupSimple2(states_kin,i)])
		elif np.absolute(example.states[startpoint+n][0] -example.states[startpoint + n-1][0])>0.1:
			states_kin = np.vstack([states_kin,example.states[startpoint+n]])	 
		else:
			print example.actions[n]
			states_kin = np.vstack([states_kin,predict_next(states_kin[n,:],example.actions[n],estimators)])
	ex = np.array(example.states[startpoint::])
	x = range(len(ex))

	f, axarr = plt.subplots(2,sharex = True)
	axarr[0].scatter(x,ex[:,1],color = "blue",label = "data")
	axarr[0].scatter(x,states_kin[:,1],color = "green",alpha = 0.4, label = "kinematic")
	axarr[0].legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)

	axarr[1].scatter(x,ex[:,0],color = "blue")
	axarr[1].scatter(x,states_kin[:,0],color = "green", alpha = 0.4)

	axarr[1].set_ylabel("Angle/rad")
	axarr[0].set_ylabel("Distance")
	axarr[1].set_xlabel("Example")
	plt.show()


#trajectoryCompare()

#w_fold1 =([-1.06272686, -1.26774088, -1.0001488,  -1.01855057, -0.37169806, -0.74641914,-2.10024122,
#-1.25042713, -1.15339101, -0.28474766, -0.34990737, -2.13778004,
 #-4.07049444])
#trajectoryCompare(w_fold1)
if __name__ == '__main__':
	estimators = learn_correction(300,7)
	for i in range(5):
		plotDrift(i,0,estimators)
#plot_reg_drift(1,1)
plt.show()