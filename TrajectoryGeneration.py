# 2015.11.27 LI Yunsheng
# 轨迹计算、轨迹评价

import numpy as np
from numpy import matlib
from math import ceil, floor
# import sqlite3


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
    N = ceil(s/ref_step) #此处有bug 2015.11.30
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


def __Jacobian(p,r=None):
    # p = (p0~p3, sg)
    # r = (a,b,c,d)
    # return J: matrix(3,3)
    J = matlib.zeros((3,3))
    p0, p1, p2, p3, sg = p
    if r is None:
        r = (__a(p), __b(p), __c(p), __d(p))
    a, b, c, d = r
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
    q_g = matlib.matrix([[bd_con[1]], [bd_con[2]], [bd_con[3]]])
    p = (bd_con[0], init_val[0], init_val[1], bd_con[4], init_val[2])
    r = (__a(p), __b(p), __c(p), __d(p))
    pp = matlib.matrix([[init_val[0]], [init_val[1]], [init_val[2]]])
    dq = matlib.matrix([[1.],[1.],[1.]])
    eps1, eps2 = 1.e-6, 1.e-8
    times = 0
    while (np.abs(dq[0,0])>eps1 or np.abs(dq[1,0])>eps1 or np.abs(dq[2,0])>eps2) and times<=500:
        times += 1
        J = __Jacobian(p, r)
        # print('J={0}'.format(J))
        theta_p = __theta(p[4], r)
        x_p, y_p = __xy_calc(p[4], r)
        dq = q_g -  matlib.matrix([[x_p],[y_p],[theta_p]])
        pp += J**-1*dq
        # 检查参数边界条件
        if pp[0,0] > 0.2:
            pp[0,0] = 0.2
        elif pp[0,0] < -0.2:
            pp[0,0] = -0.2
        if pp[1,0] > 0.2:
            pp[1,0] = 0.2
        elif pp[1,0] < -0.2:
            pp[1,0] = -0.2
        if pp[2,0] < 1.:
            pp[2,0] = 1.
        elif pp[2,0] > 400.:
            pp[2,0] = 400.
        p = (bd_con[0], pp[0,0], pp[1,0], bd_con[4], pp[2,0])
        # print('p={0}'.format(p))
        r = (__a(p), __b(p), __c(p), __d(p))
        # print('r={0}'.format(r))
    # print('IterTimes: {0}'.format(times))
    if times > 500:
        pp = matlib.matrix([[-1.],[-1.],[-1.]])
    return pp[0,0], pp[1,0], pp[2,0]


# 考虑修改接口参数为p,r...
def spiral_calc(fun, length,q=None,ref_delta_s=0.1):
    # fun: k(s)
    # q0=(0,0,0)
    # return: [s,l,x,y,theta,k]
    N = int(np.ceil(length/ref_delta_s))
    line = np.zeros((N,6))
    delta_s = length / (N-1)
    s_list = np.linspace(0,length,N)
    line[:,0] = s_list # s
    line[:,5] = fun(s_list) # k
    d_theta = (line[0:N-1,5] + line[1:N,5])/2 * delta_s
    for i in range(1,N):
        line[i,4] = line[i-1,4] + d_theta[i-1] # theta
    cos_t = np.cos(line[:,4])
    sin_t = np.sin(line[:,4])
    d_x = (cos_t[0:N-1] + cos_t[1:N])/2 * delta_s
    d_y = (sin_t[0:N-1] + sin_t[1:N])/2 * delta_s
    for i in range(1,N):
        line[i,2] = line[i-1,2] + d_x[i-1] # x
        line[i,3] = line[i-1,3] + d_y[i-1] # y
    if q is not None:
        sin_x = np.sin(q[2])*line[:,2]
        cos_x = np.cos(q[2])*line[:,2]
        sin_y = np.sin(q[2])*line[:,3]
        cos_y = np.cos(q[2])*line[:,3]
        line[:,2] = q[0] + cos_x - sin_y
        line[:,3] = q[1] + sin_x + cos_y
        line[:,4] = np.mod((line[:,4] + q[2]), 2*np.pi)
    return line


def select_init_val(cursor, bd_con):
    # cursor: cursor of a connection to sqlites database
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
    theta_r = np.mod(q1[2]-q0[2], 2*np.pi)
    bd_con = (q0[3], x_r, y_r, theta_r, q1[3])
    init_val = select_init_val(cursor, bd_con)
    pp = optimize(bd_con, init_val)
    if pp is None or pp[2]<0:
        return None
    else:
        p = (q0[3], pp[0], pp[1], q1[2], pp[3])
        r = (__a(p), __b(p), __c(p), __d(p))
        return p, r


if __name__ == '__main__':
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
    bd_con = (0.01, 100., 40., np.pi/6, -0.01)
    init_val = (0.01/3, -0.01/3, np.sqrt(100.**2+40.**2)+5*np.pi/6)
    pp = optimize(bd_con, init_val)
    print('pp={0}'.format(pp))
    p = (bd_con[0], pp[0], pp[1], bd_con[4], pp[2])
    r = (__a(p), __b(p), __c(p), __d(p))
    x_p, y_p = __xy_calc(p[4], r)
    theta_p = __theta(p[4], r)
    print('x={0}, y={1}, theta={2}'.format(x_p,y_p,theta_p))
    print('err: dx={0}, dy={1}, dtheta={2}'.format(bd_con[1]-x_p, bd_con[2]-y_p, bd_con[3]-theta_p))
