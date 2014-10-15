import sys
import math
import numpy as np
import random as rd
import time

class grid_environment():
  def __init__(self,layout,obstacle=6,goal=15):
    self.num_states = layout[0]*layout[1]
    self.layout = layout
    self.obstacle = obstacle
    self.goal = goal
    self.actions =['north','south','east','west','stay']
    self.build_reward_function()
    self.build_transition_function()
  def build_reward_function(self,weights= np.array([-0.2,-0.3])):
    #weights = weights/np.linalg.norm(weights)
    self.reward_f = np.zeros(self.layout[0]*self.layout[1])
    o_coord = state_to_coordinate(self.obstacle,self.layout)
    g_coord = state_to_coordinate(self.goal,self.layout)
    for i in range(self.layout[1]*self.layout[0]):
      a_coord = state_to_coordinate(i,self.layout)
      features = calc_features(a_coord,o_coord,g_coord,math.sqrt(self.layout[0]**2+self.layout[1]**2))
      self.reward_f[i] = sum(weights*features)    
    print "Reward Function built"
  def query_transition_function(self,state,action):
    #given a state and action returns a distribution over next states
    coord = state_to_coordinate(state,self.layout)
    out = np.zeros(self.num_states)
    new_coord = [0,0]
    if action == "north":
      if coord[1]==0:
        new_coord = coord
      else:
        new_coord = np.array([coord[0],coord[1]-1])
    elif action == "south":
      if coord[1]==self.layout[1]-1:
        new_coord = coord
      else:
        new_coord = np.array([coord[0],coord[1]+1])
    elif action ==  "east":
      if coord[0]==self.layout[0]-1:
        new_coord = coord
      else:
        new_coord = np.array([coord[0]+1,coord[1]])
    elif action ==  "west":
      if coord[1]==0:
        new_coord = coord
      else:
        new_coord = np.array([coord[0]-1,coord[1]])
    elif action ==  "stay":
      new_coord = coord
    new_state = coordinate_to_state(new_coord,self.layout)
    out[new_state] = 1
    if action == "all":
      actions = ["north","south","east","west","stay"]
      out = np.zeros([len(actions),self.num_states])
      for (i, act) in  enumerate(actions):
        out[i,:] = self.query_transition_function(state,act)
    return out
  def build_transition_function(self):
    transition_f = np.zeros([len(self.actions),self.num_states,self.num_states])
    for i in range(self.num_states):
      transition_f[:,i,:] = self.query_transition_function(i,"all")
    self.transition_f = transition_f
    print "Transition Function Built"

def state_to_coordinate(state,layout):
  assert state < layout[0]*layout[1]
  coordinate = np.array([math.fmod(state,layout[0]),math.floor(state/layout[0])])
  return coordinate

def coordinate_to_state(coord,layout):
  state = coord[1]*layout[0] + coord[0]
  return state

def calc_features(agent_coords,obstacle_coords,goal_coords,max_dist):
  feature1 = (max_dist - np.linalg.norm(agent_coords-obstacle_coords))/(max_dist/5)
  feature2 = (np.linalg.norm(agent_coords-goal_coords))/(max_dist/5)
  return np.array([feature1,feature2])

def forward_backard(transition_f,reward_f,start,goal,time_steps):
  num_actions = transition_f.shape[0];num_states = transition_f.shape[1] 
  z_actions = np.zeros([num_actions,num_states])
  z_states = np.ones(num_states)*1.0e-20
  timing = {}
  #Backward - - - - - - - - - - - - - - - - - - - - - - - - - -
  print "Backward"
  tic = time.clock()
  delta = 1
  action_probs = np.ndarray([num_actions,num_states])
  while delta > 0.001:
    prev = np.zeros((num_actions,num_states))
    prev += action_probs
    z_states[goal]+=1
    for i in range(num_states):
      for j in range(num_actions):
        m = np.amax(z_states)
        z_actions[j,i] =m+np.log(np.sum(transition_f[j,i,:]*np.exp(z_states-m))) + np.log(math.exp(reward_f[i]))
    m = np.amax(z_actions)
    z_states = m + np.log(np.sum(np.exp(z_actions-m),axis = 0))
    toc = time.clock()
   
  #Action Probability Computation - - - - - - - - - - - - - - - -
    action_probs= np.exp(z_actions-z_states)
    delta = sum(sum(np.absolute(prev-action_probs)))
  timing["backward"] = toc-tic
  #for i in range(num_states):
    #print "Action Probs for state %s,%s,\n" %(i,action_probs[:,i])
  #print "Unormalised state --------------> %s,\n"%z_states
  #Forward - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  print "STARTING FORWARD ------------------->"
  dt_states = 0.0625 * np.zeros([num_states,time_steps+1])
  dt_states[start,0] =1 
  t = []
  for i in range(time_steps):
    tic = time.clock()
    for j in range(num_states):
        dt_states[j,i+1] = np.sum(dt_states[:,i] * np.sum(action_probs * transition_f[:,:,j],axis=0))  
    toc = time.clock()
    t.append(toc - tic)
  timing["forward p timestep"] = sum(t)/len(t)
  state_freq = np.sum(dt_states,axis = 1)
  #print "State Frequencies %s,\n"%state_freq
  print "TIming", timing  
  return state_freq,timing

if __name__ == "__main__":
  obstacle = 50
  goal = 99
  layout = [60,60]
  env = grid_environment(layout,obstacle,goal)
  w_init = [-5,-10]
  w_init = w_init/np.linalg.norm(w_init)
  env.build_reward_function(w_init)
  freq = forward_backard(env.transition_f,env.reward_f,0,goal,3)


  o_coord = state_to_coordinate(obstacle,layout)
  g_coord = state_to_coordinate(goal,layout)

  w = [10,-0.7]
  grad_prev = 0
  for k in range(200):
    print "INITIAL",w_init
    w = w/np.linalg.norm(w)
    print "Iterarion ----------->", k
    env.build_reward_function(w)
    freq2,t = forward_backard(env.transition_f,env.reward_f,0,goal,3)
    cumulative1 = np.zeros(2)
    cumulative2 = np.zeros(2)
    for i,j in enumerate(freq):
     coord = state_to_coordinate(i,layout)
     feat = calc_features(coord,o_coord,g_coord,math.sqrt(layout[0]**2+layout[1]**2))
     cumulative1+= j * feat
     cumulative2+= freq2[i] * feat 

    print cumulative1,cumulative2
    print cumulative1-cumulative2
    w = w + 0.04*(cumulative1-cumulative2) + 0.01*grad_prev
    w = w/np.linalg.norm(w)
    grad_prev = cumulative1-cumulative2
    print "W = ",w


