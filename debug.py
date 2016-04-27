# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:55:10 2016

@author: liys
"""
import pickle, sqlite3
from queue import PriorityQueue
import matplotlib.pyplot as plt
from OffRoadPlanning import load_forward_path_primitives, load_reverse_path_primitives, test_load_path_primitives, \
test_load_motion_primitives, State

# test_load_path_primitives()
# test_load_motion_primitives()
# primitives1 = load_forward_path_primitives()
# primitives2 = load_reverse_path_primitives()
motion_primitives = {}
with open('motion_primitives2.pickle', 'rb') as f:
    motion_primitives.update(pickle.load(f))

conn = sqlite3.connect('InitialGuessTable.db')
cursor = conn.cursor()


start = State(index=(50,50,-5,-1), time=10., length=50., cost=0.)
pq = PriorityQueue()
pq.put(start)
node_dict = {start.index:start}
edge_dict = {}
times = 0

while times < 100 and not pq.empty():
    times += 1
    state = pq.get()
    print(state.index)
    State.ControlSet(state, motion_primitives, pq, node_dict, edge_dict)

print(len(edge_dict))
for traj in edge_dict.values():
    plt.plot(traj[:,2], traj[:,3])
for state in node_dict.values():
    plt.plot(state.state[0], state.state[1], 'ro')
#    print(state.priority)
plt.axis('equal')
plt.show()
            
cursor.close()
conn.close()