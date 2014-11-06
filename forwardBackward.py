import sys
import numpy as np
import math
import time
import random as rd

def backward(transition_f,reward_f,goal,conv=0.1,z_states = None):
    num_actions = transition_f.shape[0];num_states = transition_f.shape[1] 
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = -np.ones(num_states)*1.0e5
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Unormalised measure calculations done in log space to prevent nans
    print "Backward"
    delta = 10
    action_probs = np.zeros([num_actions,num_states])
    while delta > conv or math.isnan(delta) == True:
      prev = np.zeros((num_actions,num_states))
      prev += action_probs
      z_states[goal]=0
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
def forward(action_probs,transition_f,start,time_steps):
    print "STARTING FORWARD ------------------->"
    num_states = transition_f.shape[1] 
    dt_states = 0.0625 * np.zeros([num_states,time_steps])
    dt_states[start,0] =1 
    for i in range(time_steps-1):
      for j in range(num_states):
          dt_states[j,i+1] = np.sum(dt_states[:,i] * np.sum(action_probs * transition_f[:,:,j],axis=0))  
    state_freq = np.sum(dt_states,axis = 1)
    print "END Forward Backward Calculation------------------------------------------------------------"
    return state_freq

def smart_backward(transition_backward,transition_f,reward_f,goal,conv=0.1,z_states = None):
    num_actions = transition_f.shape[0];num_states = transition_f.shape[1] 
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = -np.ones(num_states)*1.0e5
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Unormalised measure calculations done in log space to prevent nans
    print "Backward"
    delta = 10
    action_probs = np.zeros([num_actions,num_states])
    while delta > conv or math.isnan(delta) == True:
      prev = np.zeros((num_actions,num_states))
      prev += action_probs
      
      for i in range(num_states):
        for j in range(num_actions):
          m = np.amax(z_states)
          keys = map(int,transition_backward[j + i*num_actions].keys())
          val = transition_backward[j + i*num_actions].values()
          z_actions[j,i] =m+np.log(np.sum(np.array(val)*np.exp(z_states[keys]-m))) + reward_f[i]
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      #Action Probability Computation - - - - - - - - - - - - - - - -
      action_probs= np.exp(z_actions-z_states)
      delta = sum(sum(np.absolute(prev-action_probs)))
      print delta
    print "Finished Backward. Converged at: %s"%delta
    return action_probs,z_states

def smart_forward(action_probs,transition_forward,transition_f,start,time_steps):
    print "STARTING FORWARD ------------------->"
    num_states = transition_f.shape[1] 
    num_actions = transition_f.shape[0]
    dt_states = 0.0625 * np.zeros([num_states,time_steps])
    dt_states[start,0] =1 
    for i in range(time_steps-1):
      for j in range(num_states):
        for k in range(num_actions):
          if transition_forward[k+j*num_actions].keys() !=[]:
            keys = map(int,transition_forward[k+j*num_actions].keys())
            values = transition_forward[k+j*num_actions].values()
            dt_states[j,i+1] += np.sum(dt_states[keys,i] *action_probs[k,keys] * np.array(values)) 
    state_freq = np.sum(dt_states,axis = 1)
    print "END Forward Backward Calculation------------------------------------------------------------"
    return state_freq

def caus_ent_backward(transition_backward,transition_f,reward_f,goal,steps,conv=0.1,z_states = None):
    num_actions = transition_f.shape[0];num_states = transition_f.shape[1] 
    action_probs = np.zeros([num_actions,num_states])
    gamma = 0.9
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = -np.ones(num_states)*1.0e-5  
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    print "Caus Ent Backward"
    t=0
    delta = 12
    z_states[goal] =reward_f[goal]
    while True:
      prev = np.zeros((num_actions,num_states))
      prev += action_probs
      prev2 = np.zeros((num_actions,num_states))
      prev2 += z_actions
      
      for i in range(num_states):
        for j in range(num_actions):
          keys = map(int,transition_backward[j + i*num_actions].keys())
          val = transition_backward[j + i*num_actions].values()
          z_states[goal] =reward_f[goal]
          z_actions[j,i] =gamma*np.sum(np.array(val)*z_states[keys]) +reward_f[i]
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      #Action Probability Computation - - - - - - - - - - - - - - - -
      action_probs= np.exp(z_actions-z_states)
      delta = sum(sum(np.absolute(prev-action_probs)))
      delta2 = sum(sum(np.absolute(prev2-z_actions)))
      #print "DElta cause",delta,delta2
      if delta2 < conv:
        break
    return action_probs,z_states

def caus_ent_forward(all_probs,transition_forward,transition_f,start,time_steps):
    print "STARTING FORWARD ------------------->"
    num_states = transition_f.shape[1] 
    num_actions = transition_f.shape[0]
    dt_states = 0.0625 * np.zeros([num_states,time_steps])
    dt_states[start,0] =1 
    for i in range(time_steps-1):
      for j in range(num_states):
        for k in range(num_actions):
          if transition_forward[k+j*num_actions].keys() !=[]:
            keys = map(int,transition_forward[k+j*num_actions].keys())
            values = transition_forward[k+j*num_actions].values()
            dt_states[j,i+1] += np.sum(dt_states[keys,i] *all_probs[i,k,keys] * np.array(values)) 
    state_freq = np.sum(dt_states,axis = 1)
    print "END Forward Backward Calculation------------------------------------------------------------"
    return state_freq
def q_rollout(transition_backward,transition_f,reward_f,duration):
  num_actions = transition_f.shape[0];num_states = transition_f.shape[1] 
  q_roll = np.zeros([num_actions,num_states]) 
  for i in xrange(num_actions):
    q_roll[i,:]
    for j in xrange(num_states):
      for k in xrange(duration):
          keys = map(int,transition_backward[j + i*num_actions].keys())
          val = transition_backward[j + i*num_actions].values()
          q_roll[i,j] = np.sum(np.array(val)*q_roll[i,keys]) +reward_f[i]
  return q_roll
