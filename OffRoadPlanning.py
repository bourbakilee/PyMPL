import Environment as Env
from OnRoadPlanning import trajectory_interp
import TrajectoryGeneration as TG
from queue import PriorityQueue
import numpy as np 
from scipy.interpolate import interp1d
import sqlite3
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches
import pickle
import datetime
import cv2

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
        path[:,1:] = path[::-1,1:]
        return path, p, r                  # big error
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

    if vi >= 0 and vg >= 0 and path is not None and p[4] > 0:
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

    if vi <= 0 and vg <= 0 and path is not None and p[4] > 0:
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

    for v in [-1,0,1]:
        control_set = {} # {(i,j,k,l):(traj, u)}
        if v > 0:
            for (i,j,k), (path, p, r) in forward_path.items():
                for l in [0,1]:
                    traj, u = trajectory_forward(v*1., l*1., path, p, r)
                    if traj is not None:
                        control_set[(i,j,k,l)] = (traj, u)
        elif v < 0:
            for (i,j,k), (path, p, r) in reverse_path.items():
                for l in [-1,0]:
                    traj, u = trajectory_reverse(v*1., l*1., path, p, r)
                    if traj is not None:
                        control_set[(i,j,k,l)] = (traj, u)
        else:
            for l in [-1,0,1]:
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
        with open('motion_primitives3.pickle','wb') as f:  
            pickle.dump(primitives, f)
    return primitives


def test_load_motion_primitives():
    primitives = {}
    with open('motion_primitives2.pickle', 'rb') as f:
        primitives.update(pickle.load(f))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    control_set = primitives[-2]
    for (_,_,_, vg), (traj, _) in control_set.items():
        ax.plot(traj[:,2], traj[:,3], color='magenta', linewidth=1.)
        ax.plot(traj[-1,2], traj[-1,3], 'co')
        # ax[index].text(traj[-1,2], traj[-1,3], '{0}'.format(vg), fontsize=15)

    plt.axis('equal')
    ax.set_xlim([-7,7])
    # ax.set_ylim([-4,4])
    # plt.savefig('img/reverse_path_primitives.png', dpi=600)
    plt.show()


def query_heuristic(current, heuristic_map = np.zeros((1,500,500)), static = True, vehicle = None, resolution = 0.2, time_resolution = 0.1):
    """
    current - State (OffRoad)
    """
    if vehicle is None:
        vehicle = Env.Vehicle()
    traj = np.array([[current.time, current.length, current.state[0], current.state[1], current.state[2], 0., 0., current.state[3], 0.]])
    points = vehicle.covering_centers(traj)
    index = (points/resolution).astype(int)

    if static:
        if len(heuristic_map.shape) == 2:
            hm = heuristic_map
        else:
            hm = heuristic_map[0,:,:]
        try:
            h = 1.5*hm[index[0,1], index[0,0]] + hm[index[0,3], index[0,2]] + 0.5*hm[index[0,5], index[0,4]]
        except Exception as e:
            h = 1.e10
        return h/2.
    else:
        from math import floor
        t_i = floor(current.time / time_resolution)
        if 0 <= t_i < heuristic_map.shape[0]:
            hm1 = heuristic_map[t_i,:,:]
            hm2 = heuristic_map[t_i+1,:,:]
            try:
                h1 = 1.5*hm1[index[0,1], index[0,0]] + hm1[index[0,3], index[0,2]] + 0.5*hm1[index[0,5], index[0,4]]
                h2 = 1.5*hm2[index[0,1], index[0,0]] + hm2[index[0,3], index[0,2]] + 0.5*hm2[index[0,5], index[0,4]]
                h = (((t_i+1)*time_resolution-current.time)*h1 + (current.time-t_i*time_resolution)*h2) / time_resolution
            except Exception as e:
                h = 1.e10
        elif t_i == heuristic_map.shape[0]:
            hm = heuristic_map[t_i,:,:]
            try:
                h = 1.5*hm[index[0,1], index[0,0]] + hm[index[0,3], index[0,2]] + 0.5*hm[index[0,5], index[0,4]]
            except Exception as e:
                h = 1.e10
        else:
            h = 1.e10
        return h/2.


def eval_trajectory(traj, costmap, vehicle=Env.Vehicle(), resolution=0.2, time_resolution=0.1, static=True, weights=np.array([1.5, 2., 2., 3.]), p_lims=(0.2,0.15,20.01,-0.01,2.1,-6.1,9.81), truncate=False):
    """
    evaluate the cost of given trajctory
    trajectory - [(t, s, x, y, theta, k, dk, v, a)]
    weights - trajectory length, trajectory time, end velocity, environment
    p_lims - { k_m, dk_m, v_max, v_min, a_max, a_min, ac_m }
    """
    cost_matrix = np.zeros((traj.shape[0], 6)) # k, dk, v, a, a_c, env
    cost_matrix[:,0] = np.where(np.abs(traj[:,5])>p_lims[0], np.inf, 0.) # k
    cost_matrix[:,1] = np.where(np.abs(traj[:,6])>p_lims[1], np.inf, 0.) # dk
    cost_matrix[:,2] = np.where(traj[:,7]>p_lims[2], np.inf, 0.) # v
    cost_matrix[:,2] = np.where(traj[:,7]<p_lims[3], np.inf, 0.) # v
    cost_matrix[:,3] = np.where(traj[:,8]>p_lims[4], np.inf, 0.) # a
    cost_matrix[:,3] = np.where(traj[:,8]<p_lims[5], np.inf, 0.) # a
    cost_matrix[:,4] = np.where(np.abs(traj[:,7]**2 * traj[:,5])>p_lims[6], np.inf, 0.) # a_c
    cost_matrix[:,5] = weights[3] * TG.query_cost(traj,costmap,static=static,vehicle=vehicle,resolution=resolution, time_resolution=time_resolution)[:,0]
    cost_sum = cost_matrix.sum()
    if np.isinf(cost_sum):
        return np.inf
    return cost_sum*(traj[1,1]-traj[0,1]) + weights[0]*(traj[-1,1]-traj[0,1]) - 0.6*np.abs(traj[-1,7]) #+ weights[1]*(traj[-1,0] - traj[0,0]) #+ weights[2]*np.abs(traj[-1,7])


class State:
    def __init__(self, index=None, state=None, time=0., length=0., cost=np.inf, heuristic_map=None, vehicle=None, static=True, goal=None):
        if index is not None and len(index)==4:
            i, j, k, l = index
            x, y, theta, v = i*1., j*1., np.mod(k*np.pi/16., 2*np.pi), l*1.
        elif index is None and state is not None and len(state)==4:
            x, y, theta, v = state
            i, j, k, l = int(round(x)), int(round(y)), int(np.mod(round(theta*16./np.pi),32)), int(round(v))
        else:
            x, y, theta, v = 0., 0., 0., 0.
            i, j, k, l = 0, 0, 0, 0
        self.index = (i,j,k,max(min(l, 2), -2))
        self.state = (x,y,theta,max(min(v, 2.), -2.))
        # self.x, self.y, self.theta, self.v = x, y, theta, max(min(int(v), 2), -2)
        self.time, self.length = time, length
        self.cost = cost
        if heuristic_map is not None:
            self.heuristic = query_heuristic(self, heuristic_map=heuristic_map, vehicle=vehicle, static=static)
        else:
            self.heuristic = 0.
        # if goal is not None:
        #     d_theta = abs(np.mod((goal.state[2] - self.state[2]), 2*np.pi))
        #     self.heuristic += 10*min(d_theta, 2*np.pi-d_theta) + 2.*self.state[3]*(np.cos(self.state[2])*(self.state[0]-goal.state[0]) + np.sin(self.state[2])*(self.state[1]-goal.state[1]))
        self.priority = self.cost + self.heuristic
        self.parent = None
        self.reach = False
        self.extend = False

    def __lt__(self, other):
        return self.priority < other.priority

    @staticmethod
    def distance(s1, s2):
        return abs(s1.state[0] - s2.state[0]) + abs(s1.state[1] - s2.state[1]) + abs(s1.state[2] - s2.state[2]) + abs(s1.state[3] - s2.state[3])

    @staticmethod
    def RushTowardGoal(current, goal, cursor, edge_dict, pq, cost_map=np.zeros((500,500)), vehicle=None, truncate=False, p_lims=(0.2,0.15, 6.,-4., 2.1,-6.1,10.)):
        """
        compute trajectory connect to goal state
        return : cost, trajectory - numpy array, N X 9
        """
        cost, traj = np.inf, None
        # relative coordinates
        cc, ss = np.cos(current.state[2]), np.sin(current.state[2])
        xr = (goal.state[0] - current.state[0])*cc + (goal.state[1] - current.state[1])*ss
        yr = -(goal.state[0] - current.state[0])*ss + (goal.state[1] - current.state[1])*cc
        thetar = np.mod(goal.state[2]-current.state[2], 2*np.pi)
        if thetar > np.pi:
            thetar -= 2*np.pi
        #
        if abs(thetar) < np.pi/2.:
            if current.index[3] >= 0 and xr > 0:
                path, p, r = path_forward((current.state[0], current.state[1], current.state[2], 0.), (goal.state[0], goal.state[1], goal.state[2], 0.), cursor)
                if path is not None:
                    u = calc_velocity(current.state[3], goal.state[3], p[4])
                    if u[3] is not None and u[3] > 0:
                        traj = calc_trajectory(u, p, r, s=p[4], path=path, ref_time=current.time, ref_length=current.length)
                        cost = eval_trajectory(traj, vehicle=vehicle, costmap=cost_map, truncate=truncate, p_lims=p_lims)
            if current.index[3] <= 0 and xr < 0:
                path, p, r = path_reverse((current.state[0], current.state[1], current.state[2], 0.), (goal.state[0], goal.state[1], goal.state[2], 0.), cursor)
                if path is not None:
                    u = calc_velocity(current.state[3], goal.state[3], -p[4])
                    if u[3] is not None and u[3] > 0:
                        traj2 = calc_trajectory_reverse(u, p, r, s=p[4], path=path, ref_time=current.time, ref_length=current.length)
                        cost2 = eval_trajectory(traj2, vehicle=vehicle, costmap=cost_map, truncate=truncate, p_lims=p_lims)
                        if cost2 < cost:
                            cost = cost2
                            traj = traj2
        #
        if current.cost + cost < goal.cost:
            goal.time = traj[-1,0]
            goal.length = traj[-1,1]
            current.extend = True
            goal.reach = True
            if goal.parent is not None and (goal.parent, goal) in edge_dict:
                del edge_dict[(goal.parent, goal)]
            goal.parent = current
            goal.cost = current.cost + cost
            goal.priority = goal.cost + goal.heuristic
            edge_dict[(goal.parent, goal)] = traj
            pq.put(goal)


    # def __eq__(self, other):
    #     return self.index == other.index

    @staticmethod
    def ControlSet(current, motion_primitives, pq, node_dict, edge_dict, cost_map=np.zeros((500,500)), heuristic_map=np.zeros((500,500)), vehicle=None, truncate=False, p_lims=(0.2,0.15, 6.,-4., 2.1,-6.1,10.), goal=None):
        """
        compute successors and trajectories according to motion_primitives
        motion_primitives : {v:{(i,j,k,l):(traj, u)}}
        return : [(state, traj)]
        """
        cc, ss = np.cos(current.state[2]), np.sin(current.state[2])
        two_pi = 2.*np.pi
        # control_set = []
        local_control_set = motion_primitives[current.index[3]]
        current.extend = True
        for (i,j,k,l), (ctraj,_) in local_control_set.items():
            traj = np.zeros(ctraj.shape)
            traj[:,:] = ctraj[:,:]
            successor = State(state=(current.state[0]+i*cc-j*ss, current.state[1]+i*ss+j*cc, current.state[2]+k*np.pi/16., l*1.), time=current.time+traj[-1,0], length=current.length+traj[-1,1], heuristic_map=heuristic_map, vehicle=vehicle, goal=goal)
            if 0 <= successor.state[0] <= 100 and 0 <= successor.state[1]:
                state = None
                try:
                    state = node_dict[successor.index]
                except KeyError:
                    pass
                    # if State.distance(successor, distance) > 1.e-3:
                    #     successor = state
                    #     traj = State.RushTowardGoal(current, successor, cursor)
                if state is None:
                    x_t, y_t = current.state[0] + traj[:,2]*cc - traj[:,3]*ss, current.state[1] + traj[:,2]*ss + traj[:,3]*cc
                    theta_t = np.mod(current.state[2] + traj[:,4], two_pi)
                    traj[:,2], traj[:,3], traj[:,4] = x_t, y_t, theta_t
                    traj[:,0] += current.time
                    traj[:,1] += current.length
                    cost = eval_trajectory(traj, vehicle=vehicle, costmap=cost_map, truncate=truncate, p_lims=p_lims)
                    if successor.cost > current.cost + cost:
                        successor.reach = True
                        successor.cost = current.cost + cost
                        successor.priority = successor.cost + successor.heuristic
                        pq.put(successor)
                        if successor.parent is not None and (successor.parent, successor) in edge_dict:
                            del edge_dict[(successor.parent, successor)]
                        successor.parent = current
                        node_dict[successor.index] = successor
                        edge_dict[(current, successor)] = traj
                elif State.distance(successor, state)<1.e-3:
                    successor = state
                    x_t, y_t = current.state[0] + traj[:,2]*cc - traj[:,3]*ss, current.state[1] + traj[:,2]*ss + traj[:,3]*cc
                    theta_t = np.mod(current.state[2] + traj[:,4], two_pi)
                    traj[:,2], traj[:,3], traj[:,4] = x_t, y_t, theta_t
                    traj[:,0] += current.time
                    traj[:,1] += current.length
                    cost = eval_trajectory(traj, vehicle=vehicle, costmap=cost_map, truncate=truncate, p_lims=p_lims)
                    if successor.cost > current.cost + cost:
                        successor.reach = True
                        successor.cost = current.cost + cost
                        successor.priority = successor.cost + successor.heuristic
                        if successor.parent is not None and (successor.parent, successor) in edge_dict:
                            del edge_dict[(successor.parent, successor)]
                        successor.parent = current
                        pq.put(successor)
                        # node_dict[successor.index] = successor
                        edge_dict[(current, successor)] = traj
                else:
                    successor = state
                    State.RushTowardGoal(current, successor, cursor, edge_dict, pq)
                    # if traj is not None:
                    #     pq.put(successor)
                    #     # node_dict[successor.index] = successor
                    #     edge_dict[(current, successor)] = traj

        # return node_dict, edge_dict


def Astar(start, goal, cost_map, heuristic_map, vehicle, cursor, motion_primitives):
    """
    Open Aera Motion Planning, Static Environment
    """
    pq = PriorityQueue()
    pq.put(start)
    node_dict = {start.index:start, goal.index:goal}
    edge_dict = {}
    # graph = {} # {(state1, state2):trajectory}
    times = 0
    while times<200 and not goal.reach and not pq.empty():
        times += 1
        current = pq.get()
        current.extend = True
        State.RushTowardGoal(current=current, goal=goal, cursor=cursor, edge_dict=edge_dict, pq=pq, cost_map=cost_map, vehicle=vehicle)
        # if traj_g is not None:
        #     edge_dict[(current, goal)] = traj_g
        #     pq.put(goal)
        State.ControlSet(current=current,motion_primitives=motion_primitives, pq=pq, node_dict=node_dict, edge_dict=edge_dict, cost_map=cost_map, heuristic_map=heuristic_map,vehicle=vehicle, goal=goal)
        # control_set = State.ControlSet(current=current,motion_primitives=motion_primitives, cost_map=cost_map, heuristic_map=heuristic_map,vehicle=vehicle)
        # for (successor, traj) in control_set:
        #     edge_dict[(current, successor)] = traj
        #     pq.put(successor)
    if goal.reach:
        return True, node_dict, edge_dict
    else:
        return False, node_dict, edge_dict


def vehicle_circles():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # codes6 = [Path.MOVETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.CLOSEPOLY,
    #     ]

    # s_t = (20.,40.,np.pi/4.)
    # veh = Env.Vehicle(trajectory=np.array([[-1.,-1.,s_t[0], s_t[1], s_t[2], 0., 0.,0.,0.]]))
    # veh_verts = [tuple(veh.vertex[i]) for i in range(6)]
    # veh_verts.append(veh_verts[0])
    # veh_path = Path(veh_verts, codes6)
    # veh_patch = patches.PathPatch(veh_path, facecolor='white')
    # ax.add_patch(veh_patch)

    # centers = veh.covering_disk_centers()
    # # print(centers)
    # r = veh.covering_disk_radius()
    # # print(r)

    # for i in range(3):
    #     circ = plt.Circle((centers[i,0],centers[i,1]), radius=r, color='g',fill=False)
    #     ax.add_patch(circ)
    #     ax.plot(centers[i,0],centers[i,1],'ro')

    ws = Env.Workspace()
    # collision_filter = ws.collision_filter
    # ax.imshow(collision_filter, cmap=plt.cm.Blues, origin='lower', extent=(0,0.2*collision_filter.shape[0],0,0.2*collision_filter.shape[0]))
    cost_filter = ws.cost_filter
    ax.imshow(cost_filter, cmap=plt.cm.Blues, origin='lower', extent=(0,0.2*cost_filter.shape[0],0,0.2*cost_filter.shape[0]))

    # plt.axis('equal')
    plt.show()


def configuration_space():
    fig = plt.figure()
    
    # ax1 = fig.add_subplot(421)
    # ax2 = fig.add_subplot(422)
    # ax3 = fig.add_subplot(423)
    # ax4 = fig.add_subplot(424)
    # ax5 = fig.add_subplot(425)
    # ax6 = fig.add_subplot(426)
    # ax7 = fig.add_subplot(427)
    # ax8 = fig.add_subplot(428)
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)
    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)
    ax7 = fig.add_subplot(337)
    ax8 = fig.add_subplot(338)
    ax9 = fig.add_subplot(339)
    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    # boundary = [(0.,0.), (100.,0.), (100.,100.), (0.,100.)]
    obst1 = [(50.,15.), (75.,15.), (75.,40.), (50.,40.)]
    obst2 = [(60.,60.), (85.,60.), (85.,70.), (70.,70.), (70.,90.), (60.,90.)]
    obst3 = [(20.,20.), (40.,20.), (40.,40.), (20.,40.)]
    obst4 = [(15.,60.), (50.,60.), (50.,80.), (15.,80.)]

    ws = Env.Workspace()
    obst1_map = ws.grids_occupied_by_polygon(obst1)
    obst2_map = ws.grids_occupied_by_polygon(obst2)
    obst3_map = ws.grids_occupied_by_polygon(obst3)
    obst4_map = ws.grids_occupied_by_polygon(obst4)

    obst_map = obst1_map + obst2_map + obst3_map + obst4_map

    veh_filters = [ws.vehicle_filter(theta=np.pi*2*i/9) for i in range(9)]

    collision_maps = [cv2.filter2D(obst_map, -1, veh_filters[i]) for i in range(9)]
    collision_maps = [np.where(collision_maps[i]>1.e-6, 1., 0.) for i in range(9)]
    for i in range(9):
        # ax[i].imshow(collision_maps[i], cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
        ax[i].imshow(veh_filters[i], cmap=plt.cm.Blues, origin='lower', extent=(0,0.2*veh_filters[i].shape[0],0,0.2*veh_filters[i].shape[0]))
    plt.show()


def environment_builder2():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    boundary = [(0.,0.), (100.,0.), (100.,100.), (0.,100.)]
    obst1 = [(50.,15.), (75.,15.), (75.,40.), (50.,40.)]
    obst2 = [(60.,60.), (85.,60.), (85.,70.), (70.,70.), (70.,90.), (60.,90.)]
    obst3 = [(20.,20.), (40.,20.), (40.,40.), (20.,40.)]
    obst4 = [(15.,60.), (50.,60.), (50.,80.), (15.,80.)]

    # boundary_t = [(0.,0.), (100.,0.), (100.,100.), (0.,100.)]
    # boundary_t.append(boundary_t[0])
    # obst1_t = [(50.,15.), (75.,15.), (75.,40.), (50.,40.)]
    # obst1_t.append(obst1_t[0])
    # obst2_t = [(60.,60.), (85.,60.), (85.,70.), (70.,70.), (70.,90.), (60.,90.)]
    # obst2_t.append(obst2_t[0])
    # obst3_t = [(20.,20.), (40.,20.), (40.,40.), (20.,40.)]
    # obst3_t.append(obst3_t[0])
    # obst4_t = [(15.,60.), (50.,60.), (50.,80.), (15.,80.)]
    # obst4_t.append(obst4_t[0])

    # codes4 = [Path.MOVETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.CLOSEPOLY,
    #     ]
    # codes6 = [Path.MOVETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.CLOSEPOLY,
    #     ]

    # path_b = Path(boundary_t, codes4)
    # path1 = Path(obst1_t, codes4)
    # path2 = Path(obst2_t, codes6)
    # path3 = Path(obst3_t, codes4)
    # path4 = Path(obst4_t, codes4)

    # patch_b = patches.PathPatch(path_b, facecolor='green')
    # patch1 = patches.PathPatch(path1, facecolor='cyan')
    # patch2 = patches.PathPatch(path2, facecolor='cyan')
    # patch3 = patches.PathPatch(path3, facecolor='cyan')
    # patch4 = patches.PathPatch(path4, facecolor='cyan')

    # ax.add_patch(patch_b)
    # ax.add_patch(patch1)
    # ax.add_patch(patch2)
    # ax.add_patch(patch3)
    # ax.add_patch(patch4)

    ws = Env.Workspace()
    obst1_map = ws.grids_occupied_by_polygon(obst1)
    obst2_map = ws.grids_occupied_by_polygon(obst2)
    obst3_map = ws.grids_occupied_by_polygon(obst3)
    obst4_map = ws.grids_occupied_by_polygon(obst4)

    obst_map = obst1_map + obst2_map + obst3_map + obst4_map
    # ax.imshow(obst_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    collision_map = cv2.filter2D(obst_map, -1, ws.collision_filter)
    collision_map = np.where(collision_map>1.e-6, 1., 0.)
    # ax.imshow(collision_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    cost_map = cv2.filter2D(collision_map, -1, ws.cost_filter)
    cost_map += collision_map
    cost_map = np.where(cost_map>1., np.inf, cost_map)
    cost_map = np.where(cost_map<1.e-8, 0., cost_map)
    # with open('scenario_3/cost_map.pickle','wb') as f1:  
    #     pickle.dump(cost_map, f1)
    # ax.imshow(cost_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))

    goal = (80.,90.,np.pi/2.,0.)

    heuristic_map = Env.heuristic_map_constructor(goal, cost_map)
    # heuristic_map = np.where(heuristic_map>np.finfo('d').max, np.inf, heuristic_map)
    # heuristic_map = np.where(heuristic_map<0, heuristic_map.max(), heuristic_map)
    ax.imshow(heuristic_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    # with open('scenario_3/heuristic_map.pickle','wb') as f2:  
    #     pickle.dump(heuristic_map, f2)
    ax.plot(goal[0], goal[1], 'ro')

    # plt.axis('equal')
    # ax.set_xticks(np.arange(0,100,0.2))
    # ax.set_yticks(np.arange(0,100,0.2))
    # ax.grid()
    plt.show()


def navigation():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    motion_primitives = {}
    with open('motion_primitives3.pickle', 'rb') as f:
        motion_primitives.update(pickle.load(f))
        # {v:{(i,j,k,l):(traj, u)}}

    cost_map = np.zeros((500,500))
    with open('scenario_3/cost_map.pickle', 'rb') as f1:
        cost_map = pickle.load(f1)

    heuristic_map = np.zeros((500,500))
    with open('scenario_3/heuristic_map.pickle', 'rb') as f2:
        heuristic_map = pickle.load(f2)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    boundary_t = [(0.,0.), (100.,0.), (100.,100.), (0.,100.)]
    boundary_t.append(boundary_t[0])
    obst1_t = [(50.,15.), (75.,15.), (75.,40.), (50.,40.)]
    obst1_t.append(obst1_t[0])
    obst2_t = [(60.,60.), (85.,60.), (85.,70.), (70.,70.), (70.,90.), (60.,90.)]
    obst2_t.append(obst2_t[0])
    obst3_t = [(20.,20.), (40.,20.), (40.,40.), (20.,40.)]
    obst3_t.append(obst3_t[0])
    obst4_t = [(15.,60.), (50.,60.), (50.,80.), (15.,80.)]
    obst4_t.append(obst4_t[0])

    codes4 = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]
    codes6 = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]

    path_b = Path(boundary_t, codes4)
    path1 = Path(obst1_t, codes4)
    path2 = Path(obst2_t, codes6)
    path3 = Path(obst3_t, codes4)
    path4 = Path(obst4_t, codes4)

    patch_b = patches.PathPatch(path_b, facecolor='green', alpha=0.3)
    patch1 = patches.PathPatch(path1, facecolor='cyan')
    patch2 = patches.PathPatch(path2, facecolor='cyan')
    patch3 = patches.PathPatch(path3, facecolor='cyan')
    patch4 = patches.PathPatch(path4, facecolor='cyan')

    ax.add_patch(patch_b)
    ax.add_patch(patch1)
    ax.add_patch(patch2)
    ax.add_patch(patch3)
    ax.add_patch(patch4)

    s_t = (20.,40.,np.pi/4.)
    veh = Env.Vehicle(trajectory=np.array([[-1.,-1.,s_t[0], s_t[1], s_t[2], 0., 0.,0.,0.]]))
    veh_verts = [tuple(veh.vertex[i]) for i in range(6)]
    veh_verts.append(veh_verts[0])
    veh_path = Path(veh_verts, codes6)
    veh_patch = patches.PathPatch(veh_path, facecolor='red')
    # ax.add_patch(veh_patch)

    goal = State(state=(80.,90.,-np.pi/3.,0.))
    start = State(state=(10.,10.,0.,0.), cost=0., heuristic_map=heuristic_map)

    veh1 = Env.Vehicle(trajectory=np.array([[-1.,-1.,start.state[0], start.state[1], start.state[2], 0., 0.,0.,0.]]))
    veh2 = Env.Vehicle(trajectory=np.array([[-1.,-1.,goal.state[0], goal.state[1], goal.state[2], 0., 0.,0.,0.]]))
    veh1_verts = [tuple(veh1.vertex[i]) for i in range(6)]
    veh2_verts = [tuple(veh2.vertex[i]) for i in range(6)]
    veh1_verts.append(veh1_verts[0])
    veh2_verts.append(veh2_verts[0])
    veh1_path = Path(veh1_verts, codes6)
    veh2_path = Path(veh2_verts, codes6)
    # veh1_patch = patches.PathPatch(veh1_path, facecolor='red')
    # veh2_patch = patches.PathPatch(veh2_path, facecolor='purple')
    # ax.add_patch(veh1_patch)
    # ax.add_patch(veh2_patch)

    starttime = datetime.datetime.now()
    res, state_dict, traj_dict = Astar(start, goal, cost_map=cost_map, heuristic_map=heuristic_map, vehicle=None, cursor=cursor, motion_primitives=motion_primitives)
    endtime = datetime.datetime.now()
    print((endtime - starttime).total_seconds()*1000) # 3.4s
    print("Size of Graph: {0}".format(len(traj_dict))) # 846
    # with open('scenario_3/state_dict.pickle','wb') as f1:  
    #     pickle.dump(state_dict, f1)
    # with open('scenario_3/traj_dict.pickle','wb') as f2:  
    #     pickle.dump(traj_dict, f2)
    if res:
        print("Planning Successes.")
    else:
        print("Planning Fails.")
    # for (s1, s2), traj1 in traj_dict.items():
    #     plt.plot(traj1[:,2], traj1[:,3], 'navy')
    state = goal
    rows = 0
    while state.parent is not None:
        ax.plot(state.state[0], state.state[1], 'ro')
        ax.plot(state.parent.state[0], state.parent.state[1], 'ro')
        traj = traj_dict[(state.parent, state)]
        rows += traj.shape[0]
        ax.plot(traj[:,2], traj[:,3], color='magenta', linewidth=2.)
        state = state.parent
    
    final_traj=np.zeros((rows,9))
    state = goal
    while state.parent is not None:
        traj = traj_dict[(state.parent, state)]
        final_traj[(rows-traj.shape[0]):rows,:] = traj
        rows -= traj.shape[0]
        state = state.parent
    # with open('scenario_3/final_traj.pickle','wb') as f3:  
    #     pickle.dump(final_traj, f3)

    for i in range(61):
        state1 = trajectory_interp(final_traj, i*goal.time/60)
        if state1 is not None:
            obst_d1 = Env.Vehicle(trajectory=np.array([[-1.,-1.,state1[2], state1[3], state1[4], 0., 0.,0.,0.]]))
            verts_d1 = [tuple(obst_d1.vertex[i]) for i in range(6)]
            verts_d1.append(verts_d1[0])
            ax.add_patch(patches.PathPatch(Path(verts_d1, codes6), facecolor='blue', alpha=0.1+0.015*i))


    plt.axis('equal')
    plt.show()


    cursor.close()
    conn.close()


def environment_builder():
    free_area = [(20.,40.), (80.,40.), (80.,64.), (20.,64.)]
    door_area = [(80.,48.), (85.,48.), (85.,54.), (80.,54.)]
    codes4 = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]
    codes6 = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]

    ws = Env.Workspace()

    free_area_grids = ws.grids_occupied_by_polygon(free_area)
    door_area_grids = ws.grids_occupied_by_polygon(door_area)

    obst_area_grids = 1. - free_area_grids - door_area_grids
    obst_area_grids = np.where(obst_area_grids > 1.e-3, 1., 0.)
    obst_area_grids[320,:] = 1. # fix
    obst_area_grids[270:,400] = 1. # fix
    obst_area_grids[:240,400] = 1. # fix
    obst_area_grids[:,425] = 1. # fix
    obst_area_grids[270,400:] = 1. # fix

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.imshow(obst_area_grids, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    # ax.imshow(obst_area_grids[175:345, 75:425], cmap=plt.cm.Blues, origin='lower', extent=(15,85,35,69))
    ax.plot([20, 20], [40, 64], color='black', linewidth=2.)
    ax.plot([20, 80], [64, 64], color='black', linewidth=2.)
    ax.plot([80, 80], [64, 54], color='black', linewidth=2.)
    ax.plot([80, 85], [54, 54], color='black', linewidth=2.)
    ax.plot([85, 85], [54, 48], color='black', linewidth=2.)
    ax.plot([80, 85], [48, 48], color='black', linewidth=2.)
    ax.plot([80, 80], [40, 48], color='black', linewidth=2.)
    ax.plot([80, 20], [40, 40], color='black', linewidth=2.)
    #
    ax.plot([20, 20], [52, 64], color='magenta', linewidth=1.)
    ax.plot([26, 26], [52, 64], color='magenta', linewidth=1.)
    for i in range(5):
        ax.plot([20, 26], [52+3*i, 52+3*i], color='magenta', linewidth=1.)
    #
    ax.plot([28, 62], [41, 41], color='magenta', linewidth=1.)
    ax.plot([28, 62], [44, 44], color='magenta', linewidth=1.)
    for i in range(5):
        ax.plot([28+8.5*i, 28+8.5*i], [41, 44], color='magenta', linewidth=1.)
    # ax.plot([30, 60], [40, 40], color='magenta', linewidth=1.)
    # ax.plot([30, 60], [46, 46], color='magenta', linewidth=1.)
    # for i in range(11):
    #     ax.plot([30+3*i, 30+3*i], [40, 46], color='magenta', linewidth=1.)
    #
    ax.plot([52, 70], [56, 56], color='magenta', linewidth=1.)
    ax.plot([52, 70], [62, 62], color='magenta', linewidth=1.)
    for i in range(7):
        ax.plot([52+3*i, 52+3*i], [56, 62], color='magenta', linewidth=1.)
    #
    # configurations of vehicle obstacles
    cfgs = []
    for i in range(4):
        cfgs.append((21.5,53.5+3.*i, 0.))
    for i in range(4):
        cfgs.append((34.+8.5*i,42.5,np.pi))
    # for i in range(10):
    #     cfgs.append((31.5+3.*i,41.5,np.pi/2.))
    for i in range(6):
        cfgs.append((53.5+3.*i,60.5,-np.pi/2.))
    obsts = [Env.Vehicle(trajectory=np.array([[-1.,-1.,cfg_obst[0], cfg_obst[1], cfg_obst[2], 0., 0.,0.,0.]])) for cfg_obst in cfgs]
    obsts_verts =[]
    for obst in obsts:
        obst_verts = [tuple(obst.vertex[i]) for i in range(6)]
        obst_verts.append(obst_verts[0])
        obsts_verts.append(obst_verts)
    pathes = [Path(verts, codes6) for verts in obsts_verts]
    patch_list = [patches.PathPatch(path, facecolor='cyan') for path in pathes]
    # for patch in patch_list:
    #     ax.add_patch(patch)
    # configuration of vehicle
    start = (80.,51.,np.pi)
    goal = (42.5, 42.5, np.pi)
    #
    #
    for i, obst in enumerate(obsts):
        if i !=5:
            obst_area_grids += ws.grids_occupied_by_polygon(obst.vertex)
    # ax.imshow(obst_area_grids, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    # np.savetxt('scenario_2/obst_bitmap.txt', obst_area_grids, fmt='%i', delimiter=' ')

    collision_map = cv2.filter2D(obst_area_grids, -1, ws.collision_filter)
    collision_map = np.where(collision_map>1.e-6, 1., 0.)
    # ax.imshow(collision_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    # np.savetxt('scenario_2/collision_bitmap.txt', collision_map, fmt='%i', delimiter=' ')

    cost_map = cv2.filter2D(collision_map, -1, ws.cost_filter)
    cost_map += collision_map
    cost_map = np.where(cost_map>1., np.inf, cost_map)
    cost_map = np.where(cost_map<1.e-8, 0., cost_map)
    # costmap_save = np.where( cost_map >1., -1., cost_map)
    # with open('scenario_2/cost_map.pickle','wb') as f1:  
    #     pickle.dump(cost_map, f1)
    # ax.imshow(cost_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    # np.savetxt('scenario_2/cost_map.txt', cost_map, delimiter=' ')

    heuristic_map = Env.heuristic_map_constructor(goal, cost_map)
    # ax.imshow(heuristic_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    # with open('scenario_2/heuristic_map.pickle','wb') as f2:  
    #     pickle.dump(heuristic_map, f2)
    plt.axis('equal')
    plt.show()



def parking():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    motion_primitives = {}
    with open('motion_primitives3.pickle', 'rb') as f:
        motion_primitives.update(pickle.load(f))
        # {v:{(i,j,k,l):(traj, u)}}

    cost_map = np.zeros((500,500))
    with open('scenario_2/cost_map.pickle', 'rb') as f1:
        cost_map = pickle.load(f1)

    heuristic_map = np.zeros((500,500))
    with open('scenario_2/heuristic_map.pickle', 'rb') as f2:
        heuristic_map = pickle.load(f2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([20, 20], [40, 64], color='black', linewidth=2.)
    ax.plot([20, 80], [64, 64], color='black', linewidth=2.)
    ax.plot([80, 80], [64, 54], color='black', linewidth=2.)
    ax.plot([80, 85], [54, 54], color='black', linewidth=2.)
    ax.plot([85, 85], [54, 48], color='black', linewidth=2.)
    ax.plot([80, 85], [48, 48], color='black', linewidth=2.)
    ax.plot([80, 80], [40, 48], color='black', linewidth=2.)
    ax.plot([80, 20], [40, 40], color='black', linewidth=2.)
    #
    ax.plot([20, 20], [52, 64], color='magenta', linewidth=1.)
    ax.plot([26, 26], [52, 64], color='magenta', linewidth=1.)
    for i in range(5):
        ax.plot([20, 26], [52+3*i, 52+3*i], color='magenta', linewidth=1.)
    #
    ax.plot([28, 62], [41, 41], color='magenta', linewidth=1.)
    ax.plot([28, 62], [44, 44], color='magenta', linewidth=1.)
    for i in range(5):
        ax.plot([28+8.5*i, 28+8.5*i], [41, 44], color='magenta', linewidth=1.)
    # ax.plot([30, 60], [40, 40], color='magenta', linewidth=1.)
    # ax.plot([30, 60], [46, 46], color='magenta', linewidth=1.)
    # for i in range(11):
    #     ax.plot([30+3*i, 30+3*i], [40, 46], color='magenta', linewidth=1.)
    #
    ax.plot([52, 70], [56, 56], color='magenta', linewidth=1.)
    ax.plot([52, 70], [62, 62], color='magenta', linewidth=1.)
    for i in range(7):
        ax.plot([52+3*i, 52+3*i], [56, 62], color='magenta', linewidth=1.)
    #
    ax.imshow(cost_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))

    goal = State(state=(42.5, 42.5, np.pi,0.))
    start = State(state=(80.,51.,np.pi,0.), cost=0., heuristic_map=heuristic_map)
    
    plt.plot(start.state[0], start.state[1], 'ro')
    
    plt.plot(goal.state[0], goal.state[1], 'mo')

    res, state_dict, traj_dict = Astar(start, goal, cost_map=cost_map, heuristic_map=heuristic_map, vehicle=None, cursor=cursor, motion_primitives=motion_primitives)
    print("Size of Graph: {0}".format(len(traj_dict)))
    with open('scenario_2/state_dict.pickle','wb') as f1:  
        pickle.dump(state_dict, f1)
    with open('scenario_2/traj_dict.pickle','wb') as f2:  
        pickle.dump(traj_dict, f2)
    if res:
        print("Planning Successes.")
        # state = goal
        # while state != start:
        #     traj = g[(state.parent, state)]
        #     plt.plot(traj[:,2], traj[:,3], color='magenta', linewidth=1.)
    else:
        print("Planning Fails.")
    for _, state in state_dict.items():
        plt.plot(state.state[0], state.state[1], 'mo')
    for (s1, s2), traj1 in traj_dict.items():
        # plt.plot(s1.state[0], s1.state[1], 'ro')
        # plt.plot(s2.state[0], s2.state[1], 'ro')
        plt.plot(traj1[:,2], traj1[:,3])
    state = goal
    while state.parent is not None:
        plt.plot(state.state[0], state.state[1], 'ro')
        plt.plot(state.parent.state[0], state.parent.state[1], 'ro')
        traj = traj_dict[(state.parent, state)]
        plt.plot(traj[:,2], traj[:,3], color='navy', linewidth=2.)
        state = state.parent
    plt.axis('equal')
    plt.show()


    cursor.close()
    conn.close()


def test_state():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    motion_primitives = {}
    with open('motion_primitives2.pickle', 'rb') as f:
        motion_primitives.update(pickle.load(f))

    goal = State(index=(50,53,0,0))
    start = State(index=(50,50,0,0), time=0., length=0., cost=0., goal=goal)
    
    # traj = start.RushTowardGoal(goal,cursor)

    control_set = State.ControlSet(start, motion_primitives)

    if control_set:
        for (state, ctraj) in control_set:
            print(ctraj[-1,:])
            print(state.state)
            print(state.time)
            print(state.length)
            plt.plot(state.state[0], state.state[1], 'ro')
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


def test_astar():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    motion_primitives = {}
    with open('motion_primitives2.pickle', 'rb') as f:
        motion_primitives.update(pickle.load(f))

    goal = State(state=(20.,45.,0.,0.))
    start = State(state=(50.,50.,0.,0.), time=0., length=0., cost=0., goal=goal)
    plt.plot(start.state[0], start.state[1], 'ro')
    
    plt.plot(goal.state[0], goal.state[1], 'mo')

    res, state_dict, traj_dict = Astar(start, goal, cost_map=np.zeros((1,500,500)), heuristic_map=np.zeros((1,500,500)), vehicle=None, cursor=cursor, motion_primitives=motion_primitives)
    print("Size of Graph: {0}".format(len(traj_dict)))
    # with open('state_dict.pickle','wb') as f1:  
    #     pickle.dump(state_dict, f1)
    # with open('traj_dict.pickle','wb') as f2:  
    #     pickle.dump(traj_dict, f2)
    if res:
        print("Planning Successes.")
        # state = goal
        # while state != start:
        #     traj = g[(state.parent, state)]
        #     plt.plot(traj[:,2], traj[:,3], color='magenta', linewidth=1.)
    else:
        print("Planning Fails.")
    # for _, state in state_dict.items():
    #     plt.plot(state.state[0], state.state[1], 'mo')
    # for (s1, s2), traj1 in traj_dict.items():
    #     # plt.plot(s1.state[0], s1.state[1], 'ro')
    #     # plt.plot(s2.state[0], s2.state[1], 'ro')
    #     plt.plot(traj1[:,2], traj1[:,3])
    state = goal
    while state.parent is not None:
        plt.plot(state.state[0], state.state[1], 'ro')
        plt.plot(state.parent.state[0], state.parent.state[1], 'ro')
        traj = traj_dict[(state.parent, state)]
        plt.plot(traj[:,2], traj[:,3], color='navy', linewidth=2.)
        state = state.parent
    plt.axis('equal')
    plt.show()


    cursor.close()
    conn.close()


if __name__ == '__main__':

    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    # forward_path_primitives(cursor)
    # reverse_path_primitives(cursor)
    # test_load_path_primitives()
    # motion_primitives_construction()
    # test_load_motion_primitives()
    # test_state()
    # test_astar()
    # environment_builder()
    # parking()
    # environment_builder2()
    navigation()
    # configuration_space()
    # vehicle_circles()

    cursor.close()
    conn.close()