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


def calc_velocity(vi, vg, sg):
    # ai=0, ag=0
    # v(t) = a + bt + ct^2 + dt^3, t->[0, tg]
    # b == ai = 0
    # return : (a,c,d,tg)
    a, c, d, tg = vi, None, None, None
    v_s = vi + vg
    if abs(v_s) > 1.e-4:
        tg = 2*sg / v_s
        if tg > 0:
            c = -3*(vi - vg)*v_s**2 / (4*sg**2)
            d = -2*c / (3*tg)
    return (a,c,d,tg)


def calc_trajectory(u, p, r, path, s=None, ref_time=0., ref_length=0.):
    # u: (u0~u2, tg)
    # p: (p0~p3, sg)
    # r: (a,b,c,d)
    # s: path length
    # path: [(s,x,y,theta,k)]
    # return: array of points on trajectory - [(t,s,x,y,theta,k,dk,v,a)]
    u0, u1, u2, tg = u
    # p0, p1, p2, p3, sg = p
    a, b, c, d = r
    if s is None:
        s = p[4]
    trajectory = np.zeros((path.shape[0], 9)) # NX10 array
    trajectory[:,1:6] = path # s,x,y,theta,k
    trajectory[-1,0] = tg
    # time at given path length
    t_list = np.linspace(0., tg, path.shape[0])
    s_list = np.array([u0*t+u1*t**3/3+u2*t**4/4 for t in t_list])
    s2t = interp1d(s_list, t_list) # time @ given path length
    trajectory[1:-1, 0] = s2t(trajectory[1:-1, 1]) # t
    #
    trajectory[:,7] = np.array([u0+u1*t**2+u2*t**3 for t in trajectory[:,0]]) # v
    trajectory[:,8] = np.array([2*u1*t+3*u2*t**2 for t in trajectory[:,0]]) # a
    # trajectory[:,9] = 2*u2
    # dk/dt
    trajectory[:,6] = np.array([b+2*c*ss+3*d*ss**2 for ss in trajectory[:,1]])*trajectory[:,7]
    # revise the absolute time and length
    trajectory[:,0] += ref_time
    trajectory[:,1] += ref_length
    return trajectory


def calc_trajectory_reverse(u, p, r, path, s=None, ref_time=0., ref_length=0.):
    # u: (u0~u2, tg)
    # p: (p0~p3, sg)
    # r: (a,b,c,d)
    # s: path length
    # path: [(s,x,y,theta,k)]
    # return: array of points on trajectory - [(t,s,x,y,theta,k,dk,v,a)]
    u0, u1, u2, tg = u
    # p0, p1, p2, p3, sg = p
    a, b, c, d = r
    if s is None:
        s = p[4]
    trajectory = np.zeros((path.shape[0], 9)) # NX10 array
    trajectory[:,1:6] = path # s,x,y,theta,k
    trajectory[-1,0] = tg
    # time at given path length
    t_list = np.linspace(0., tg, path.shape[0])
    s_list = np.array([-u0*t-u1*t**3/3-u2*t**4/4 for t in t_list])
    s2t = interp1d(s_list, t_list) # time @ given path length
    trajectory[1:-1, 0] = s2t(trajectory[1:-1, 1]) # t
    #
    trajectory[:,7] = np.array([u0+u1*t**2+u2*t**3 for t in trajectory[:,0]]) # v
    trajectory[:,8] = np.array([2*u1*t+3*u2*t**2 for t in trajectory[:,0]]) # a
    # trajectory[:,9] = 2*u2
    # dk/dt
    trajectory[:,6] = np.array([b+2*c*ss+3*d*ss**2 for ss in trajectory[:,1]])*trajectory[:,7]
    # revise the absolute time and length
    trajectory[:,0] += ref_time
    trajectory[:,1] += ref_length
    return trajectory


def trajectory_forward(vi, vg, path, p, r, ai=0., cost_map=np.zeros((500,500)),  \
    truncate=False, p_lims=(0.2,0.15, 6.,-4., 2.1,-6.1,10.)):

    if vi >= 0 and vg > 0 and path is not None and p[4] > 0:
        # u = TG.calc_velocity(vi, ai, vg, p[4])
        u = calc_velocity(vi, vg, p[4])
        if u[3] is not None and u[3]>0:
            # traj = TG.calc_trajectory(u, p, r, s=p[4], path=path, ref_time=0., ref_length=0.)
            traj = calc_trajectory(u, p, r, s=p[4], path=path, ref_time=0., ref_length=0.)
            cost = TG.eval_trajectory(traj, costmap=cost_map, truncate=truncate, p_lims=p_lims)
            if not np.isinf(cost):
                return traj, u
    return None, None


def trajectory_reverse(vi, vg, path, p, r, ai=0., cost_map=np.zeros((500,500)),  \
    truncate=False, p_lims=(0.2,0.15, 6.,-4., 2.1,-6.1,10.)):

    if vi <= 0 and vg < 0 and path is not None and p[4] > 0:
        # u = TG.calc_velocity(vi, ai, vg, -p[4])
        u = calc_velocity(vi, vg, -p[4])
        if u[3] is not None and u[3]>0:
            # traj = TG.calc_trajectory_reverse(u, p, r, s=p[4], path=path, ref_time=0., ref_length=0.)
            traj = calc_trajectory_reverse(u, p, r, s=p[4], path=path, ref_time=0., ref_length=0.)
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
def motion_primitives_construction():
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
        with open('motion_primitives2.pickle','wb') as f:  
            pickle.dump(primitives, f)
    return primitives


def test_load_motion_primitives():
    primitives = {}
    with open('motion_primitives2.pickle', 'rb') as f:
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


class State:
    def __init__(self, index=None, state=None, time=np.inf, length=np.inf, cost=np.inf, heuristic_map=None, vehicle=None):
        if index is not None and len(index)==4:
            i, j, k, l = index
            x, y, theta, v = i*1., j*1., np.mod(k*np.pi/16., 2*np.pi), l*1.
        elif index is None and state is not None and len(state)==4:
            x, y, theta, v = state
            i, j, k, l = round(x), round(y), np.mod(round(theta*16./np.pi),32), round(v)
        else:
            x, y, theta, v = 0., 0., 0., 0.
            i, j, k, l = 0, 0, 0, 0
        self.index = (i,j,k,max(min(l, 2), -2))
        self.state = (x,y,theta,max(min(v, 2.), -2.))
        # self.x, self.y, self.theta, self.v = x, y, theta, max(min(int(v), 2), -2)
        self.time, self.length = time, length
        self.cost = cost
        if heuristic_map is not None:
            self.heuristic = 0. #query_heuristic(self, heuristic_map, vehicle)
        else:
            self.heuristic = 0.
        self.priority = self.cost + self.heuristic
        self.parent = None
        self.reach = False
        self.extend = False

    def RushTowardGoal(self, goal, cursor, cost_map=np.zeros((500,500)), vehicle=None, truncate=False, p_lims=(0.2,0.15, 6.,-4., 2.1,-6.1,10.)):
        """
        compute trajectory connect to goal state
        return : cost, trajectory - numpy array, N X 9
        """
        cost, traj = np.inf, None
        # relative coordinates
        cc, ss = np.cos(self.state[2]), np.sin(self.state[2])
        xr = (goal.state[0] - self.state[0])*cc + (goal.state[1] - self.state[1])*ss
        yr = -(goal.state[0] - self.state[0])*ss + (goal.state[1] - self.state[1])*cc
        thetar = np.mod(goal.state[2]-self.state[2], 2*np.pi)
        if thetar > np.pi:
            thetar -= 2*np.pi
        #
        if abs(thetar) < np.pi/2.:
            if self.index[3] >= 0 and xr > 0:
                path, p, r = path_forward((self.state[0], self.state[1], self.state[2], 0.), (goal.state[0], goal.state[1], goal.state[2], 0.), cursor)
                if path is not None:
                    u = calc_velocity(self.state[3], goal.state[3], p[4])
                    if u[3] is not None and u[3] > 0:
                        traj = calc_trajectory(u, p, r, s=p[4], path=path, ref_time=self.time, ref_length=self.length)
                        cost = TG.eval_trajectory(traj, vehicle=vehicle, costmap=cost_map, truncate=truncate, p_lims=p_lims)
            if self.index[3] <= 0 and xr < 0:
                path, p, r = path_reverse((self.state[0], self.state[1], self.state[2], 0.), (goal.state[0], goal.state[1], goal.state[2], 0.), cursor)
                if path is not None:
                    u = calc_velocity(self.state[3], goal.state[3], -p[4])
                    if u[3] is not None and u[3] > 0:
                        traj2 = calc_trajectory_reverse(u, p, r, s=p[4], path=path, ref_time=self.time, ref_length=self.length)
                        cost2 = TG.eval_trajectory(traj2, vehicle=vehicle, costmap=cost_map, truncate=truncate, p_lims=p_lims)
                        if cost2 < cost:
                            cost = cost2
                            traj = traj2
        #
        if self.cost + cost < goal.cost:
            self.extend = True
            goal.reach = True
            goal.parent = self
            goal.cost = self.cost + cost
            goal.priority = goal.cost + goal.heuristic
        return traj

    # def __eq__(self, other):
    #     return self.index == other.index

    def ControlSet(self, motion_primitives, cost_map=np.zeros((500,500)), heuristic_map=np.zeros((500,500)), vehicle=None, truncate=False, p_lims=(0.2,0.15, 6.,-4., 2.1,-6.1,10.)):
        """
        compute successors and trajectories according to motion_primitives
        motion_primitives : {v:{(i,j,k,l):(traj, u)}}
        return : [(state, traj)]
        """
        cc, ss = np.cos(self.state[2]), np.sin(self.state[2])
        two_pi = 2.*np.pi
        control_set = []
        local_control_set = motion_primitives[self.index[3]]
        for (i,j,k,l), (traj,_) in local_control_set.items():
            successor = State(state=(self.state[0]+i*cc-j*ss, self.state[1]+i*ss+j*cc, self.state[2]+k*np.pi/16., l*1.), heuristic_map=heuristic_map, vehicle=vehicle)
            x_t, y_t = self.state[0] + traj[:,2]*cc - traj[:,3]*ss, self.state[1] + traj[:,2]*ss + traj[:,3]*cc
            theta_t = np.mod(self.state[2] + traj[:,4], two_pi)
            traj[:,2], traj[:,3], traj[:,4] = x_t, y_t, theta_t
            cost = TG.eval_trajectory(traj, vehicle=vehicle, costmap=cost_map, truncate=truncate, p_lims=p_lims)
            if not np.isinf(cost):
                self.extend = True
                successor.reach = True
                successor.cost = self.cost + cost
                successor.priority = successor.cost + successor.heuristic
                successor.parent = self
                control_set.append((successor, traj))
        return control_set


def test_state():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    motion_primitives = {}
    with open('motion_primitives2.pickle', 'rb') as f:
        motion_primitives.update(pickle.load(f))

    start = State(index=(50,50,-1,-1), time=0., length=0., cost=0.)
    goal = State(index=(40,51,-2,-2))
    traj = start.RushTowardGoal(goal,cursor)

    control_set = start.ControlSet(motion_primitives)

    if control_set:
        for (_, ctraj) in control_set:
            plt.plot(ctraj[:,2], ctraj[:,3])
        plt.axis('equal')
        plt.show()

    # if traj is not None:
    #     plt.plot(traj[:,2], traj[:,3])
    #     plt.axis('equal')
    #     plt.show()
    # else:
    #     print("RushTowardGoal Fails")


    cursor.close()
    conn.close()




if __name__ == '__main__':

    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    # forward_path_primitives(cursor)
    # reverse_path_primitives(cursor)
    # test_load_path_primitives()
    # motion_primitives()
    # test_load_motion_primitives()
    test_state()

    cursor.close()
    conn.close()