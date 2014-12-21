import sys
import numpy as np
import math
import random as rd
def smart_backward(transition,reward_f,goal,conv=0.1,z_states = None):
    num_actions = transition.tot_actions;num_states = transition.tot_states
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = -np.ones(num_states)*1.0e5
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Unormalised measure calculations done in log space to prevent nans
    print "Backward"
    delta = 10
    policy = np.zeros([num_actions,num_states])
    while delta > conv or math.isnan(delta) == True:
      prev = np.zeros((num_actions,num_states))
      prev += policy
      for i in range(num_states):
        for j in range(num_actions):
          m = np.amax(z_states)
          keys = map(int,transition.backward[j + i*num_actions].keys())
          val = transition.backward[j + i*num_actions].values()
          z_actions[j,i] =m+np.log(np.sum(np.array(val)*np.exp(z_states[keys]-m))) + reward_f[i]
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      #Action Probability Computation - - - - - - - - - - - - - - - -
      policy= np.exp(z_actions-z_states)
      delta = sum(sum(np.absolute(prev-policy)))
      print delta
    print "Finished Backward. Converged at: %s"%delta
    return policy,z_states

def forward(policy,transition,start,time_steps):
    print "STARTING FORWARD ------------------->"
    num_actions = transition.tot_actions;num_states = transition.tot_states
    dt_states = 0.0625 * np.zeros([num_states,time_steps])
    dt_states_actions = np.zeros([num_actions,num_states,time_steps])
    for i in start: dt_states[i,0]+=1 
    dt_states[:,0] /=len(start)
    for i in range(time_steps):
      for j in range(num_states):
        for k in range(num_actions):
          if transition.forward[k+j*num_actions].keys() !=[] and i != time_steps-1:
            keys = map(int,transition.forward[k+j*num_actions].keys())
            values = transition.forward[k+j*num_actions].values()
            dt_states[j,i+1] += np.sum(dt_states[keys,i] *policy[k,keys] * np.array(values)) 
          dt_states_actions[k,j,i] = dt_states[j,i]*policy[k,j]
    state_action_freq = np.sum(dt_states_actions,axis=2)        
    state_freq = np.sum(dt_states,axis = 1)
    print "END Forward Backward Calculation------------------------------------------------------------"
    return state_freq,state_action_freq

def forward_alt(policy,transition,start,time_steps):
    print "STARTING FORWARD ------------------->"
    num_actions = transition.tot_actions;num_states = transition.tot_states
    dt_states = np.zeros([num_states,time_steps])
    dt_states[:,0] = 1./num_states*np.ones(num_states)
    #for i in start: dt_states[i,0]+=1 
    #dt_states[start,0] = dt_states[start,0]/np.sum(dt_states[start,0]) 
    for i in range(time_steps-1):
      for j in range(num_states):
        for k in range(num_actions):
          if transition.forward[k+j*num_actions].keys() !=[]:
            keys = map(int,transition.forward[k+j*num_actions].keys())
            values = transition.forward[k+j*num_actions].values()
            dt_states[j,i+1] += np.sum(dt_states[keys,i] *policy[k,keys] * np.array(values)) 
    state_freq = np.sum(dt_states,axis = 1)
    print "END Forward Backward Calculation------------------------------------------------------------"
    return state_freq,dt_states
def forward_sa(policy,transition,start,time_steps):
    print "STARTING FORWARD ------------------->"
    num_actions = transition.tot_actions;num_states = transition.tot_states
    dt_states = np.zeros([num_states,time_steps])
    dt_states_actions = np.zeros([num_actions,num_states,time_steps])
    #for i in start: dt_states[i,0]+=1 
    #dt_states[:,0] /=len(start)
    dt_states[:,0] = 1./num_states*np.ones(num_states)
    for i in range(time_steps-1):
      for j in range(num_states):
        for k in range(num_actions):
          if transition.forward[k+j*num_actions].keys() !=[]:
            keys = map(int,transition.forward[k+j*num_actions].keys())
            values = transition.forward[k+j*num_actions].values()
            dt_states[j,i+1] += np.sum(dt_states[keys,i] *policy[k,keys] * np.array(values)) 
          dt_states_actions[k,j,i] = dt_states[j,i]*policy[k,j]
    
    state_freq = np.sum(dt_states,axis = 1)
    state_action_freq = np.sum(dt_states_actions,axis=2)
    print "END Forward Backward Calculation------------------------------------------------------------"
    return state_freq,state_action_freq


def caus_ent_backward(transition,reward_f,goal,steps,conv=0.1,z_states = None):
    num_actions = transition.tot_actions;num_states = transition.tot_states
    if reward_f.shape[0] ==num_actions:
      state_action = True
    else: state_action =False
    gamma = 1
    z_actions = np.zeros([num_actions,num_states])
    if z_states==None:
      z_states = np.zeros(num_states)
      #z_states[goal] = 0
    #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
    print "Caus Ent Backward"
    delta = 0
    #reward_f[:,goal]/=1.1
    while True:
      #prev = np.zeros(z_states.shape)
      #prev += z_states
      for i in range(num_states):
        for j in range(num_actions):
          keys = map(int,transition.backward[j + i*num_actions].keys())
          val = transition.backward[j + i*num_actions].values()
          if state_action == True:
            z_actions[j,i] =(gamma*np.sum(np.array(val)*z_states[keys]) +reward_f[j,i])  
          else:
            z_actions[j,i] =(gamma*np.sum(np.array(val)*z_states[keys]) +reward_f[i])  
      m = np.amax(z_actions)
      z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
      #Action Probability Computation - - - - - - - - - - - - - - - -
      #delta = sum(sum(np.absolute(prev-z_states)))
      delta +=1 
      #print delta
      #print "DElta cause",delta,delta2
      if delta ==steps:
        z_actions = z_actions
        m = np.amax(z_actions)
        z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
        policy= np.exp(z_actions-z_states)
        break
    return policy,np.log(policy)

def q_rollout(transition_backward,transition_f,reward_f,duration):
  num_actions = transition.tot_actions;num_states = transition.tot_states
  q_roll = np.zeros([num_actions,num_states]) 
  for i in xrange(num_actions):
    q_roll[i,:]
    for j in xrange(num_states):
      for k in xrange(duration):
          keys = map(int,transition.backward[j + i*num_actions].keys())
          val = transition.backward[j + i*num_actions].values()
          q_roll[i,j] = np.sum(np.array(val)*q_roll[i,keys]) +reward_f[i]
  return q_roll

if __name__=="main":
  x = 5

