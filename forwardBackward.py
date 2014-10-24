import sys
import numpy as np
import math
import time
import random as rd

def backward(transition_f,reward_f,goal,conv=0.1,z_states = None):
    num_actions = transition_f.shape[0];num_states = transition_f.shape[1] 
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = np.ones(num_states)*1.0e-20
    timing = {}
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Unormalised measure calculations done in log space to prevent nans
    print "Backward"
    delta = 10
    action_probs = np.ndarray([num_actions,num_states])
    while delta > conv or math.isnan(delta) == True:
      prev = np.zeros((num_actions,num_states))
      prev += action_probs
      z_states[goal]+=1
      for i in range(num_states):
        for j in range(num_actions):
          m = np.amax(z_states)
          z_actions[j,i] =m+np.log(np.sum(transition_f[j,i,:]*np.exp(z_states-m))) + np.log(math.exp(reward_f[i]))
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
    #Action Probability Computation - - - - - - - - - - - - - - - -
      action_probs= np.exp(z_actions-z_states)
      delta = sum(sum(np.absolute(prev-action_probs)))
      print delta
    print "Finished Backward. Converged at: %s"%delta
    return action_probs,z_states
    #for i in range(num_states):
      #print "Action Probs for state %s,%s,\n" %(i,action_probs[:,i])
    #print "Unormalised state --------------> %s,\n"%z_states
    #Forward - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def forward(action_probs,transition_f,start,time_steps):
    print "STARTING FORWARD ------------------->"
    num_states = transition_f.shape[1] 

    dt_states = 0.0625 * np.zeros([num_states,time_steps])
    dt_states[start,0] =1 
    t = []
    for i in range(time_steps-1):
      tic = time.clock()
      for j in range(num_states):
          dt_states[j,i+1] = np.sum(dt_states[:,i] * np.sum(action_probs * transition_f[:,:,j],axis=0))  
    state_freq = np.sum(dt_states,axis = 1)
    print "END Forward Backward Calculation------------------------------------------------------------"
    return state_freq
