from Environment import *
import TrajectoryGeneration as TG
from queue import PriorityQueue
import numpy as np 
import sqlite3
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle

# configuration - Q(x, y, theta)
# state - X(Q, v)
# time-state - TX(t,x,y,theta,v)
# path - (s,x,y,theta,k)


def path_forward(q0, q1, cursor):
    p, r = TG.calc_path(cursor, q0, q1)
    if r is not None and p[4] > 0.:
        path = TG.spiral3_calc(p, r=r, q=q0)
        return path, p, r
    return None, None, None


def path_reverse(q0, q1, cursor):
    p, r = TG.calc_path(cursor, q1, q0)
    if r is not None and p[4] > 0.:
        path = TG.spiral3_calc(p, r=r, q=q1)
        return path[::-1,:], p, r
    return None, None, None


def path_eval(path, k_m = 0.2):
    if path is not None:
        v_cost = np.where(path[:,4] > k_m, np.inf, 0.)
        return v_cost.sum()
    else:
        return np.inf

# dx - 1m - {0,1,...,100}
# dy - 1m - {0,1,...,100}
# dtheta - pi/8 - {0,1,2,...,15}
# dv - 1m/s - {-2,-1,0,1,2}
def forward_path_primitives(cursor):
    # forward
    q0 = (0.,0.,0.,0.)
    primitives = {}
    for i in range(1, 7):
        for j in range(-3, 4):
            for k in range(-8, 9):
                q1 = (i, j, k*np.pi/16, 0)
                path, p, r = path_forward(q0, q1, cursor)
                cost = path_eval(path)
                if not np.isinf(cost):
                    primitives[(i,j,k)] = (path, p, r)
    with open('forward_path_primitives3.pickle','wb') as f:  
        pickle.dump(primitives, f)
    # return primitives
    # print(len(primitives))
    # for q, path in primitives.items():
    #     plt.plot(q[0], q[1], 'co')
    #     plt.plot(path[:,1], path[:,2], color='magenta', linewidth=1.)
    # plt.axis('equal')
    # plt.show()


def reverse_path_primitives(cursor):
    # forward
    q0 = (0.,0.,0.,0.)
    primitives = {}
    for i in range(-6, 0):
        for j in range(-3, 4):
            for k in range(-8, 9):
                q1 = (i, j, k*np.pi/16, 0)
                path, p, r = path_reverse(q0, q1, cursor)
                cost = path_eval(path)
                if not np.isinf(cost):
                    primitives[(i,j,k)] = (path, p, r)
    with open('reverse_path_primitives3.pickle','wb') as f:  
        pickle.dump(primitives, f)
    # return primitives
    # print(len(primitives))
    # for q, path in primitives.items():
    #     plt.plot(q[0], q[1], 'co')
    #     plt.plot(path[:,1], path[:,2], color='magenta', linewidth=1.)
    # plt.axis('equal')
    # plt.show()


def load_forward_path_primitives():
    with open('forward_path_primitives3.pickle', 'rb') as f:
        primitives = pickle.load(f)
        return primitives
    return None


def load_reverse_path_primitives():
    with open('reverse_path_primitives3.pickle', 'rb') as f:
        primitives = pickle.load(f)
        return primitives
    return None


def test_load_path_primitives():
    primitives = {}
    primitives.update(load_forward_path_primitives())
    primitives.update(load_reverse_path_primitives())

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(0, 0, 'ro')
    for (i,j,k), (path, pr, r) in primitives.items():
        q = (i*1., j*1., k*np.pi/16., 0.)
        ax.plot(q[0], q[1], 'co')
        ax.plot(path[:,1], path[:,2], color='magenta', linewidth=1.)
    plt.axis('equal')
    ax.set_xlim([-7,7])
    ax.set_ylim([-4,4])
    # plt.savefig('img/reverse_path_primitives.png', dpi=600)
    plt.show()


def trajectory_forward(vi, vg, path, p, r, ai=0., cost_map=np.zeros((500,500)),  \
    truncate=False, p_lims=(0.2,0.15, 6.,-4., 2.1,-6.1,10.)):

    if vi >= 0 and vg > 0 and path is not None and p[4] > 0:
        u = TG.calc_velocity(vi, ai, vg, p[4])
        if u[3] is not None and u[3]>0:
            traj = TG.calc_trajectory(u, p, r, s=p[4], path=path, ref_time=0., ref_length=0.)
            cost = TG.eval_trajectory(traj, costmap=cost_map, truncate=truncate, p_lims=p_lims)
            if not np.isinf(cost):
                return traj, u
    return None, None


def trajectory_reverse(vi, vg, path, p, r, ai=0., cost_map=np.zeros((500,500)),  \
    truncate=False, p_lims=(0.2,0.15, 6.,-4., 2.1,-6.1,10.)):

    if vi <= 0 and vg < 0 and path is not None and p[4] > 0:
        u = TG.calc_velocity(vi, ai, vg, -p[4])
        if u[3] is not None and u[3]>0:
            traj = TG.calc_trajectory_reverse(u, p, r, s=p[4], path=path, ref_time=0., ref_length=0.)
            cost = TG.eval_trajectory(traj, costmap=cost_map, truncate=truncate, p_lims=p_lims)
            if not np.isinf(cost):
                return traj, u
    return None, None


def trajectory_stay(state_index, time=1.):
    if len(state_index) == 4:
        i, j, k, l = state_index
        from math import floor
        N = 1+floor(time*10)
        traj = np.zeros((N, 9))
        traj[:,0] = np.linspace(0., time, N)
        traj[:,1] = 0.
        traj[:,2] = i*1.
        traj[:,3] = j*1.
        traj[:,4] = k*np.pi/16.
        return traj, (0.,0.,0.,time)
    return None, None




# v in {-2,-1,0,1,2}
def motion_primitives():
    primitives = {} # {v:{(i,j,k,l):(traj, u)}}

    forward_path = load_forward_path_primitives()
    reverse_path = load_reverse_path_primitives()

    for v in [-2,-1,0,1,2]:
        control_set = {} # {(i,j,k,l):(traj, u)}
        if v > 0:
            for (i,j,k), (path, p, r) in forward_path.items():
                for l in [0,1,2]:
                    traj, u = trajectory_forward(v*1., l*1., path, p, r)
                    if traj is not None:
                        control_set[(i,j,k,l)] = (traj, u)
        elif v < 0:
            for (i,j,k), (path, p, r) in reverse_path.items():
                for l in [-2,-1,0]:
                    traj, u = trajectory_reverse(v*1., l*1., path, p, r)
                    if traj is not None:
                        control_set[(i,j,k,l)] = (traj, u)
        else:
            for l in [-2,-1,0,1,2]:
                if l == 0:
                    traj, u = trajectory_stay((0,0,0,0), 1.)
                    control_set[(0,0,0,0)] = (traj, u)
                elif l > 0:
                    for (i,j,k), (path, p, r) in forward_path.items():
                        traj, u = trajectory_forward(v*1., l*1., path, p, r)
                        if traj is not None:
                            control_set[(i,j,k,l)] = (traj, u)
                else:
                    for (i,j,k), (path, p, r) in reverse_path.items():
                        traj, u = trajectory_reverse(v*1., l*1., path, p, r)
                        if traj is not None:
                            control_set[(i,j,k,l)] = (traj, u)
        primitives[v] = control_set
        with open('motion_primitives.pickle','wb') as f:  
            pickle.dump(primitives, f)
    return primitives


def test_load_motion_primitives():
    primitives = {}
    with open('motion_primitives.pickle', 'rb') as f:
        primitives.update(pickle.load(f))

    fig = plt.figure()
    ax1 = fig.add_subplot(1,5,1)
    ax2 = fig.add_subplot(1,5,2)
    ax3 = fig.add_subplot(1,5,3)
    ax4 = fig.add_subplot(1,5,4)
    ax5 = fig.add_subplot(1,5,5)
    ax = [ax1, ax2, ax3, ax4, ax5]

    for index, v in enumerate([-2,-1,0,1,2]):
        control_set = primitives[v]
        for _, (traj, _) in control_set.items():
            ax[index].plot(traj[:,2], traj[:,3], color='magenta', linewidth=1.)
            ax[index].plot(traj[-1,2], traj[-1,3], 'co')

    plt.axis('equal')
    # ax.set_xlim([-7,7])
    # ax.set_ylim([-4,4])
    # plt.savefig('img/reverse_path_primitives.png', dpi=600)
    plt.show()



if __name__ == '__main__':

    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    # forward_path_primitives(cursor)
    # reverse_path_primitives(cursor)
    # test_load_path_primitives()
    # motion_primitives()
    test_load_motion_primitives()

    cursor.close()
    conn.close()