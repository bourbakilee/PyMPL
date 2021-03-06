# 2015.07.27 LI Yunsheng
# Trajectory Calculation and Trajectory Evaluation

import numpy as np
from numpy import matlib
from math import ceil, floor
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import Environment as Env # Environment module is not used here, because the trajectory evaluation procedure is just a trial and not include environment factors
# import sqlite3 # donnot import sqlite3, just add a parameter of database connection to the functions which needs

# k(s) = a + b*s + c*s**2 + d*s**3

def __a(p):
    # p = (p0 ~ p3, sg)
    return p[0]


def __b(p):
    return -(11*p[0]-18*p[1]+9*p[2]-2*p[3])/(2*p[4])


def __c(p):
    return 9*(2*p[0]-5*p[1]+4*p[2]-p[3])/(2*p[4]**2)


def __d(p):
    return -9*(p[0]-3*p[1]+3*p[2]-p[3])/(2*p[4]**3)


def __k(s,r):
    # r = (a,b,c,d) = (__a(p),__b(p),__c(p),__d(p))
    return r[0] + r[1]*s + r[2]*s**2 + r[3]*s**3


def __theta(s,r):
    return r[0]*s + r[1]*s**2/2 + r[2]*s**3/3 + r[3]*s**4/4


def __xy_calc(s, r, ref_step=8.0):
    # s: interval of integration - [0, s]
    # r: (a,b,c,d) - k(s) = a + b*s + c*s**2 + d*s**3
    # return: x(s), y(s)
    # if s >= 0:
    N = ceil(s/ref_step) #here exists a bug 2015.11.30
    # else:
    #     N = abs(floor(s/ref_step))
    # print(s)
    # print(N)
    s_N = np.linspace(0.,s,N+1)
    h = s/(8*N)
    F = 0.
    G = 0.
    theta = lambda t:__theta(t, r)
    for i in range(N):
        ss = np.linspace(s_N[i], s_N[i+1], 9)
        thetas = theta(ss)
        fs = np.cos(thetas)
        gs = np.sin(thetas)
        F += h/3*(fs[0] + 2*(fs[2]+fs[4]+fs[6]) + 4*(fs[1]+fs[3]+fs[5]+fs[7]) + fs[8])
        G += h/3*(gs[0] + 2*(gs[2]+gs[4]+gs[6]) + 4*(gs[1]+gs[3]+gs[5]+gs[7]) + gs[8])
    return F, G


# not used
def __s_t(t,u):
    # t - time
    # u - (u0~u2,tg)
    return u[0]*t + u[1]*t**2/2 + u[2]*t**3/3


def __Jacobian(p,r=None):
    # p = (p0~p3, sg)
    # r = (a,b,c,d)
    # return J: matrix(3,3)
    J = matlib.zeros((3,3))
    p0, p1, p2, p3, sg = p
    if r is None:
        r = (__a(p), __b(p), __c(p), __d(p))
    # a, b, c, d = r
    ss = np.linspace(0., sg, 9)
    thetas = __theta(ss[1:],r)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    c_p_1 = sg**2/24*np.array([1851/8192, 363/1024, 9963/8192, 51/64, 14475/8192, 891/1024, 13083/8192, 3/8])
    c_p_2 = sg**2/24*np.array([795/8192, 123/1024, 2187/8192, 3/64, -2325/8192, -405/1024, -10437/8192, -3/8])
    c_s_1 = sg/24*np.array([(2871*p0+1851*p1-795*p2+169*p3)/8192, \
                            (247*p0+363*p1-123*p2+25*p3)/1024, \
                            (4071*p0+9963*p1-2187*p2+441*p3)/8192, \
                            (15*p0+51*p1-3*p2+p3)/64, \
                            (3655*p0+14475*p1+2325*p2+25*p3)/8192, \
                            (231*p0+891*p1+405*p2+9*p3)/1024, \
                            (3927*p0+13083*p1+10437*p2+1225*p3)/8192, \
                            (p0+3*p1+3*p2+p3)/8])
    c_s_2 = np.array([1/6, 1/12, 1/6, 1/12, 1/6, 1/12, 1/6, 1/24])
    # dx_dp1, dx_dp2, dx_dsg; dy_dp1, dy_dp2, dy_dsg; dtheta_dp1, dtheta_dp2, dtheta_dsg
    J[0,0] = -np.sum(c_p_1*sin_t)
    J[0,1] = np.sum(c_p_2*sin_t)
    J[0,2] = 1/24 + np.sum(c_s_2*cos_t - c_s_1*sin_t)
    J[1,0] = np.sum(c_p_1*cos_t)
    J[1,1] = -np.sum(c_p_2*cos_t)
    J[1,2] = np.sum(c_s_2*sin_t + c_s_1*cos_t)
    J[2,0] = 0.375*sg
    J[2,1] = J[2,0]
    J[2,2] = c_s_1[-1]
    return J


def optimize(bd_con, init_val=None):
    # bd_con: boundary conditions - (k0, x1, y1, theta1, k1)
    # init_val: (p1, p2, sg)
    if init_val is None or init_val[2]<0:
        init_val = ((2*bd_con[0]+bd_con[4])/3, (bd_con[0]+2*bd_con[4])/3, np.sqrt(bd_con[1]**2+bd_con[2]**2)+5*min(np.abs(bd_con[3]), np.abs(2*np.pi-bd_con[3])))
    # revise the parameters
    init_val = list(init_val)
    init_val[0]= max(min(init_val[0],0.2),-0.2)
    init_val[1]= max(min(init_val[1],0.2),-0.2)
    init_val[2]= max(min(init_val[2],1000.),1.)
    # print('IterTimes: {0},P = {1}'.format(0, (init_val[0], init_val[1], init_val[2])))
    q_g = matlib.matrix([[bd_con[1]], [bd_con[2]], [bd_con[3]]])
    p = (bd_con[0], init_val[0], init_val[1], bd_con[4], init_val[2])
    r = (__a(p), __b(p), __c(p), __d(p))
    pp = matlib.matrix([[init_val[0]], [init_val[1]], [init_val[2]]])
    
    dq = matlib.matrix([[1.],[1.],[1.]])
    eps1, eps2 = 1.e-4, 1.e-6
    times = 0
    while (np.abs(dq[0,0])>eps1 or np.abs(dq[1,0])>eps1 or np.abs(dq[2,0])>eps2) and times<=100:
        times += 1
        J = __Jacobian(p, r)
        # print('J={0}'.format(J))
        theta_p = __theta(p[4], r)
        x_p, y_p = __xy_calc(p[4], r)
        # print('q = {0}'.format((x_p,y_p,theta_p)))
        dq = q_g -  matlib.matrix([[x_p],[y_p],[theta_p]])
        pp += J**-1*dq
        # revise the parameters
        pp[0,0]= max(min(pp[0,0],0.2),-0.2)
        pp[1,0]= max(min(pp[1,0],0.2),-0.2)
        pp[2,0]= max(min(pp[2,0],1000.),1.)
        # 
        # if pp[0,0] > 0.2:
        #     pp[0,0] = 0.2
        # elif pp[0,0] < -0.2:
        #     pp[0,0] = -0.2
        # if pp[1,0] > 0.2:
        #     pp[1,0] = 0.2
        # elif pp[1,0] < -0.2:
        #     pp[1,0] = -0.2
        # if pp[2,0] < 1.:
        #     pp[2,0] = 1.
        # elif pp[2,0] > 1000.:
        #     pp[2,0] = 1000.
        p = (bd_con[0], pp[0,0], pp[1,0], bd_con[4], pp[2,0])
        # print('p={0}'.format(p))
        r = (__a(p), __b(p), __c(p), __d(p))
        # print('r={0}'.format(r))
        # print('IterTimes: {0},P = {1}'.format(times, (p[1],p[2],p[4])))
    # print('IterTimes: {0}'.format(times))
    if times > 100:
        # pp = matlib.matrix([[-1.],[-1.],[-1.]])
        return None
    else:
        return pp[0,0], pp[1,0], pp[2,0]


def select_init_val(cursor, bd_con):
    # cursor: cursor of a connection to sqlite3 database
    # bd_con: boundary conditions - (k0,x1,y1,theta1,k1)
    # return: initial value of key - (p1,p2,sg)
    i, j, k, l, m = floor(bd_con[0]*40), floor(bd_con[1]*16/49), floor(bd_con[2]*0.16), floor(bd_con[3]*16/np.pi), floor(bd_con[4]*40)
    key_list = [(i,j,k,l,m),\
                (i+1,j,k,l,m),(i,j+1,k,l,m),(i,j,k+1,l,m),(i,j,k,l+1,m),(i,j,k,l,m+1),\
                (i+1,j+1,k,l,m),(i+1,j,k+1,l,m),(i+1,j,k,l+1,m),(i+1,j,k,l,m+1),(i,j+1,k+1,l,m),(i,j+1,k,l+1,m),(i,j+1,k,l,m+1),(i,j,k+1,l+1,m),(i,j,k+1,l,m+1),(i,j,k,l+1,m+1),\
                (i+1,j+1,k+1,l,m),(i+1,j+1,k,l+1,m),(i+1,j+1,k,l,m+1),(i+1,j,k+1,l+1,m),(i+1,j,k+1,l,m+1),(i,j+1,k+1,l+1,m),(i,j+1,k+1,l,m+1),(i+1,j,k,l+1,m+1),(i,j+1,k,l+1,m+1),(i,j,k+1,l+1,m+1),\
                (i+1,j+1,k+1,l+1,m),(i+1,j+1,k+1,l,m+1),(i+1,j+1,k,l+1,m+1),(i+1,j,k+1,l+1,m+1),(i,j+1,k+1,l+1,m+1),\
                (i+1,j+1,k+1,l+1,m+1)]
    val = None
    for key in key_list:
        cursor.execute('select p1, p2, sg from InitialGuessTable where \
        k0=? and x1=? and y1=? and theta1=? and k1=?', key)
        val = cursor.fetchone()
        if val is not None and val[2]>0:
            break
    return val


def calc_path(cursor, q0, q1):
    # cursor: cursor of connection to sqlite3 database
    # q0, q1: initial and goal configuration - (x,y,theta,k)
    # return: p(p0~p3,sg), r(a,b,c,d)
    cc = np.cos(q0[2])
    ss = np.sin(q0[2])
    x_r = (q1[0] - q0[0])*cc + (q1[1] - q0[1])*ss
    y_r = -(q1[0] - q0[0])*ss + (q1[1] - q0[1])*cc
    theta_r = np.mod(q1[2]-q0[2], 2*np.pi) # 
    if theta_r > np.pi:
        theta_r -= 2*np.pi
    # elif theta_r < -np.pi:
    #     theta_r += 2*np.pi
    bd_con = (q0[3], x_r, y_r, theta_r, q1[3])
    init_val = select_init_val(cursor, bd_con) #
    # print(init_val)
    pp = optimize(bd_con, init_val) #
    if pp is None or pp[2]<0:
        p, r = None, None
    else:
        p = (q0[3], pp[0], pp[1], q1[3], pp[2])
        r = (__a(p), __b(p), __c(p), __d(p))
    return p, r


def calc_path_no_init_val(q0, q1):
    # q0, q1: initial and goal configuration - (x,y,theta,k)
    # return: p(p0~p3,sg), r(a,b,c,d)
    cc = np.cos(q0[2])
    ss = np.sin(q0[2])
    x_r = (q1[0] - q0[0])*cc + (q1[1] - q0[1])*ss
    y_r = -(q1[0] - q0[0])*ss + (q1[1] - q0[1])*cc
    theta_r = np.mod(q1[2]-q0[2], 2*np.pi) # 
    if theta_r > np.pi:
        theta_r -= 2*np.pi
    bd_con = (q0[3], x_r, y_r, theta_r, q1[3])
    pp = optimize(bd_con) #
    print(pp)
    if pp is None or pp[2]<0:
        p, r = None, None
    else:
        p = (q0[3], pp[0], pp[1], q1[3], pp[2])
        r = (__a(p), __b(p), __c(p), __d(p))
    return p, r


def calc_velocity(v0, a0, vg, sg):
    # v(t) = q0 + q1*t + q2*t**2
    # return: q0~q2, tg
    u0, u1, u2, tg = v0, a0, None, None
    delta = (2*v0+vg)**2 + 6*a0*sg
    if delta >= -1.e-6:
        if abs(a0) > 1.e-4:
            if sg>0.:
                tg = (np.sqrt(max(0.,delta)) - 2*v0 -vg)/a0
            elif sg<0.:
                tg = (-np.sqrt(max(0.,delta)) - 2*v0 -vg)/a0
        else:
            tg = 3*sg/(2*v0+vg)
    if tg is not None:
        u2 = (vg - v0 - a0*tg)/tg**2
    return (u0,u1,u2,tg)


def spiral3_calc(p, r=None, s=None, q=(0.,0.,0.), ref_delta_s=0.1):
    # 
    # p: (p0~p3, sg)
    # r: (a,b,c,d)
    # q0=(0,0,0)
    # return: NX5 array - [s,x,y,theta,k]
    if r is None:
        r = (__a(p), __b(p), __c(p), __d(p))
    if s is None:
        s = p[4]
    N = ceil(s / ref_delta_s)
    line = np.zeros((N+1,5))
    delta_s = s / N
    line[:,0] = np.linspace(0, s ,N+1) # s
    line[:,4] = __k(line[:,0], r) # k
    line[:,3] = np.mod(q[2] + __theta(line[:,0], r), 2*np.pi) # theta q[2] + __theta(line[:,0], r)
    cos_t = np.cos(line[:,3])
    sin_t = np.sin(line[:,3])
    d_x = (cos_t[0:N] + cos_t[1:N+1])/2 * delta_s
    d_y = (sin_t[0:N] + sin_t[1:N+1])/2 * delta_s
    line[0,1], line[0,2] = q[0], q[1]
    for i in range(1,N+1):
        line[i,1] = line[i-1,1] + d_x[i-1] # x
        line[i,2] = line[i-1,2] + d_y[i-1] # y
    return line


def calc_trajectory(u, p, r=None, s=None, path=None, q0=None, ref_time=0., ref_length=0.):
    # u: (u0~u2, tg)
    # p: (p0~p3, sg)
    # r: (a,b,c,d)
    # s: path length
    # path: [(s,x,y,theta,k)]
    # q0: (x0,y0,theta0)
    # return: array of points on trajectory - [(t,s,x,y,theta,k,dk,v,a)]
    u0, u1, u2, tg = u
    if r is None:
        r = (__a(p), __b(p), __c(p), __d(p))
    # p0, p1, p2, p3, sg = p
    a, b, c, d = r
    if s is None:
        s = p[4]
    #
    if path is None:
        path = spiral3_calc(p, r, s, q0) # NX5 array
    trajectory = np.zeros((path.shape[0], 9)) # NX10 array
    trajectory[:,1:6] = path # s,x,y,theta,k
    trajectory[-1,0] = tg
    # time at given path length
    t_list = np.linspace(0., tg, path.shape[0])
    #     s_list = __s_t(t_list, u)
    s_list = np.array([u0*t+u1*t**2/2+u2*t**3/3 for t in t_list])
    s2t = interp1d(s_list, t_list) # time @ given path length
    trajectory[1:-1, 0] = s2t(trajectory[1:-1, 1]) # t
    #
    trajectory[:,7] = np.array([u0+u1*t+u2*t**2 for t in trajectory[:,0]]) # v
    trajectory[:,8] = np.array([u1+2*u2*t for t in trajectory[:,0]]) # a
    # trajectory[:,9] = 2*u2
    # dk/dt
    trajectory[:,6] = np.array([b+2*c*ss+3*d*ss**2 for ss in trajectory[:,1]])*trajectory[:,7]
    # revise the absolute time and length
    trajectory[:,0] += ref_time
    trajectory[:,1] += ref_length
    return trajectory


def calc_trajectory_reverse(u, p, r=None, s=None, path=None, q0=None, ref_time=0., ref_length=0.):
    # u: (u0~u2, tg)
    # p: (p0~p3, sg)
    # r: (a,b,c,d)
    # s: path length
    # path: [(s,x,y,theta,k)]
    # q0: (x0,y0,theta0)
    # return: array of points on trajectory - [(t,s,x,y,theta,k,dk,v,a)]
    u0, u1, u2, tg = u
    if r is None:
        r = (__a(p), __b(p), __c(p), __d(p))
    # p0, p1, p2, p3, sg = p
    a, b, c, d = r
    if s is None:
        s = p[4]
    #
    if path is None:
        path = spiral3_calc(p, r, s, q0) # NX5 array
    trajectory = np.zeros((path.shape[0], 9)) # NX10 array
    trajectory[:,1:6] = path # s,x,y,theta,k

    trajectory[-1,0] = tg
    trajectory[1:-1, 0] = np.array([ fsolve(lambda t: u0*t+u1*t**2/2.+u2*t**3/3.+ss, 2*ss/u0)[0] for ss in trajectory[1:-1, 1]])# t
    

    #
    trajectory[:,7] = np.array([u0+u1*t+u2*t**2 for t in trajectory[:,0]]) # v
    trajectory[:,8] = np.array([u1+2*u2*t for t in trajectory[:,0]]) # a
    # trajectory[:,9] = 2*u2
    # dk/dt
    trajectory[:,6] = np.array([b+2*c*ss+3*d*ss**2 for ss in trajectory[:,1]])*trajectory[:,7]
    # revise the absolute time and length
    trajectory[:,0] += ref_time
    trajectory[:,1] += ref_length
    return trajectory



# query cost from costmap
# traj: array of points on trajectory - [(t,s,x,y,theta,k,dk,v,a)]
def query_cost(traj, costmap, static = True, vehicle = None, resolution = 0.2, time_resolution = 0.1):
    if vehicle is None:
        vehicle = Env.Vehicle()
    points = vehicle.covering_centers(traj)
    index = (points/resolution).astype(int)

    cost = np.zeros((traj.shape[0],1))
    if static:
        if len(costmap.shape) == 2:
            cm = costmap
        else:
            cm = costmap[0,:,:]
        for i in range(traj.shape[0]):
            try:
                cost[i] = 1.5*costmap[index[i,1], index[i,0]] + costmap[index[i,3], index[i,2]] + 0.5*costmap[index[i,5], index[i,4]]
            except Exception as e:
                cost[i] = 1.e10
    else: # dynamic
        t_i = np.floor(traj[:,0]/time_resolution).astype(int)
        for i in range(traj.shape[0]):
            if 0 <= t_i[i] < costmap.shape[0]-1:
                cm1 = costmap[t_i[i], :, :]
                cm2 = costmap[t_i[i]+1, :, :]
                try:
                    cost1 = 1.5*cm1[index[i,1], index[i,0]] + cm1[index[i,3], index[i,2]] + 0.5*cm1[index[i,5], index[i,4]]
                    cost2 = 1.5*cm2[index[i,1], index[i,0]] + cm2[index[i,3], index[i,2]] + 0.5*cm2[index[i,5], index[i,4]]
                    cost[i] = (((t_i[i]+1)*time_resolution-traj[i,0])*cost1 + (traj[i,0]-t_i[i]*time_resolution)*cost2) / time_resolution
                except Exception as e:
                    cost[i] = 1.e10
            elif t_i[i] == costmap.shape[0]-1:
                cm = costmap[t_i[i], :, :]
                try:
                    cost[i] = 1.5*cm[index[i,1], index[i,0]] + cm[index[i,3], index[i,2]] + 0.5*cm[index[i,5], index[i,4]]
                except Exception as e:
                    cost[i] = 1.e10
            else:
                cost[i] = 1.e10
    return cost


# this trajectory evaluation procedure is just a trial, the formal one has been writen in C++ ~~~~
# trajectory evaluation - not only check the collision, also truncate the collision part of trajectory
# 
def eval_trajectory(trajectory, costmap, vehicle=Env.Vehicle(), road=None, resolution=0.2, time_resolution = 0.1, truncate=True, static=True, \
    weights=np.array([5., 10., 0.05, 0.2, 0.2, 0.2, 10., 0.5, 10., -2.]), p_lims=(0.2,0.15,20.01,-0.01,2.1,-6.1,9.81)):
    # trajectory: array of points on trajectory - [(t,s,x,y,theta,k,dk,v,a)]
    # weights: weights for (k, dk, v, a, a_c, l, env, j, t, s)
    # p_lims - { k_m, dk_m, v_max, v_min, a_max, a_min, ac_m } = (0.2,0.1,20.,0.,2.,-6.,10.) - (np.inf, np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
    # return: cost
    #
    delta_s = trajectory[1,1] - trajectory[0,1]
    time = trajectory[-1,0] - trajectory[0,0]
    length = trajectory[-1,1] - trajectory[0,1]
    jerk = (trajectory[1,8] - trajectory[0,8])/(trajectory[1,0] - trajectory[0,0])
    # print(delta_s, time, length, jerk)
    # cost matrix
    cost_matrix = np.zeros((trajectory.shape[0],7)) # k, dk, v, a, a_c, off_set(x,y), env(t,x,y,theta)
    cost_matrix[:,0] = weights[0] * np.abs(trajectory[:,5]) # k
    cost_matrix[:,1] = weights[1] * np.abs(trajectory[:,6]) # dk
    cost_matrix[:,2] = weights[2] * trajectory[:,7] # v
    cost_matrix[:,3] = weights[3] * trajectory[:,8]**2 # a
    cost_matrix[:,4] = weights[4] * np.abs(trajectory[:,5]) * trajectory[:,7]**2 # a_c
    if road is not None:
        # false
        l_list = road.xy2sl(trajectory[:,2], trajectory[:,3])[1]
        # print(l_list)
        cost_matrix[:,5] = weights[5] * np.abs(l_list) # should also consider target lane offset
    if costmap is not None:
        cost_matrix[:,6] = weights[6]*query_cost(trajectory, costmap=costmap, vehicle=vehicle, resolution=resolution, time_resolution = 0.1, static=static)[:,0]
    # cost matrix revise
    cost_matrix[:,0] = np.where(np.abs(trajectory[:,5])>p_lims[0], np.inf, cost_matrix[:,0]) # k
    cost_matrix[:,1] = np.where(np.abs(trajectory[:,6])>p_lims[1], np.inf, cost_matrix[:,1]) # dk
    cost_matrix[:,2] = np.where(trajectory[:,7]>p_lims[2], np.inf, cost_matrix[:,2]) # v
    cost_matrix[:,2] = np.where(trajectory[:,7]<p_lims[3], np.inf, cost_matrix[:,2]) # v
    cost_matrix[:,3] = np.where(trajectory[:,8]>p_lims[4], np.inf, cost_matrix[:,3]) # a
    cost_matrix[:,3] = np.where(trajectory[:,8]<p_lims[5], np.inf, cost_matrix[:,3]) # a
    cost_matrix[:,4] = np.where(np.abs(cost_matrix[:,4])>weights[4]*p_lims[6], np.inf, cost_matrix[:,4]) # a_c
    # cost synthesize 
    if truncate:
        row_cost = cost_matrix.sum(axis=1)
        N = row_cost.shape[0]
        M = 0
        for i in range(N):
            if not np.isinf(row_cost[i]):
                M += 1
            else:
                break
        Row = 2*M // 3
        if M == N: # and trajectory[-1,1] - trajectory[0,1]>3.:
            return row_cost.sum()*delta_s + weights[7]*np.abs(jerk)*length + weights[8]*time + weights[9]*length, trajectory, False
        elif 2<M<N and trajectory[Row-1,1] - trajectory[0,1]>2.:
            return row_cost[0:Row].sum()*delta_s + (weights[7]*np.abs(jerk)+weights[9])*(trajectory[Row-1,1] - trajectory[0,1]) + weights[8]*(trajectory[Row-1,0]-trajectory[0,0]), trajectory[0:Row,:], True
        else:
            return 0., None, True
    else:
        return cost_matrix.sum()*delta_s + weights[7]*np.abs(jerk)*length + weights[8]*time + weights[9]*length # +  weights[9]*length/(weights[8]*time)



# if __name__ == '__main__':
    # test
    # p0, p1, p2, p3, sg = 0.01, 0.0070893846923453458, 0.0056488100020089405, -0.01, 109.61234579137816
    # p = (p0,p1,p2,p3,sg)
    # a, b, c, d = __a(p), __b(p), __c(p), __d(p)
    # r = (a,b,c,d)
    # theta = lambda s: __theta(s,r)
    # x, y = __xy_calc(sg,theta)
    # t = theta(sg)
    # J = __Jacobian(p,r)
    # print(p)
    # print(r)
    # print(x,y)
    # print(J)
    # print(J**-1*(matlib.matrix([[100.-x],[40.-y],[np.pi/6-t]])))
    # bd_con = (0.01, 100., 40., np.pi/6, -0.01)
    # init_val = (0.01/3, -0.01/3, np.sqrt(100.**2+40.**2)+5*np.pi/6)
    # pp = optimize(bd_con, init_val)
    # # print('pp={0}'.format(pp))
    # p = (bd_con[0], pp[0], pp[1], bd_con[4], pp[2])
    # r = (__a(p), __b(p), __c(p), __d(p))
    # # x_p, y_p = __xy_calc(p[4], r)
    # # theta_p = __theta(p[4], r)
    # # print('x={0}, y={1}, theta={2}'.format(x_p,y_p,theta_p))
    # # print('err: dx={0}, dy={1}, dtheta={2}'.format(bd_con[1]-x_p, bd_con[2]-y_p, bd_con[3]-theta_p))
    # path = spiral3_calc(p,r)
    # # print(path)
    # u = calc_velocity(5.,0.2,10.,p[4])
    # # print(u)
    # trajectory = calc_trajectory(u,p,r,path=path)
    # print(trajectory[:,0], trajectory[:,7])
    # cost = eval_trajectory(trajectory)
    # print(cost)
