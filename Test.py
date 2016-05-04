from Environment import *
from OnRoadPlanning import *
import TrajectoryGeneration as TG
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches
import sqlite3
from math import floor, ceil

def test_vehicle():
    veh = Vehicle()
    print(veh.covering_disk_radius())
    print(veh.covering_disk_centers())

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    verts = [tuple(veh.vertex[i]) for i in range(6)]
    verts.append(verts[0])
    codes = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path,facecolor='green')
    ax1.add_patch(patch)
    plt.axis('equal')
    plt.axis('off')
    plt.show()



def road_profile(s):
    #s<=220
    #return 0.01-0.00038611167606780294*s+4.4656981495145228e-6*s**2-1.4071316854528154e-8*s**3
    #s<=110
    return 0.01-0.000242811907598*s+6.42266190994e-6*s**2-5.3571326595e-8*s**3


def test():
    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, q=(0.,0.,0.))
    print(center_line[-1,:])


def test_road():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, s=70., q=(5.,30.,np.pi/9))
    road = Road(center_line, ref_grid_width=1, ref_grid_length=2.)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax1.plot(center_line[:,1], center_line[:,2], color='red', linestyle='--', linewidth=2.)
    ax1.plot(center_line[:,1], center_line[:,2], color='red', linewidth=1.)
    ax1.plot(road.lateral_lines[:,0], road.lateral_lines[:,1],color='red', linewidth=1.)
    ax1.plot(road.lateral_lines[:,-2], road.lateral_lines[:,-1],color='green', linewidth=1.)

    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.)
        else:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=0.3)
    for i in range(road.grid_num_longitudinal+1):
        ax1.plot(road.lateral_lines[:,2*i], road.lateral_lines[:,2*i+1],color='black', linewidth=0.3)
    #
    cfg0 = road.ij2xy(2,-4)
    veh = Vehicle(trajectory=np.array([[-1.,-1.,cfg0[0], cfg0[1], cfg0[2], cfg0[3], 0.,5.,0.]]))
    verts0 = [tuple(veh.vertex[i]) for i in range(6)]
    verts0.append(verts0[0])

    cfg1 = road.ij2xy(15, -4)
    cfg2 = road.ij2xy(15, 0)
    cfg3 = road.ij2xy(30, 0)
    cfg4 = road.ij2xy(30., 4)
    obst1 = Vehicle(trajectory=np.array([[-1.,-1.,cfg1[0], cfg1[1], cfg1[2], cfg1[3], 0.,0.,0.]]))
    obst2 = Vehicle(trajectory=np.array([[-1.,-1.,cfg2[0], cfg2[1], cfg2[2], cfg2[3], 0.,0.,0.]]))
    obst3 = Vehicle(trajectory=np.array([[-1.,-1.,cfg3[0], cfg3[1], cfg3[2], cfg3[3], 0.,0.,0.]]))
    obst4 = Vehicle(trajectory=np.array([[-1.,-1.,cfg4[0], cfg4[1], cfg4[2], cfg4[3], 0.,0.,0.]]))
    verts1 = [tuple(obst1.vertex[i]) for i in range(6)]
    verts1.append(verts1[0])
    verts2 = [tuple(obst2.vertex[i]) for i in range(6)]
    verts2.append(verts2[0])
    verts3 = [tuple(obst3.vertex[i]) for i in range(6)]
    verts3.append(verts3[0])
    verts4 = [tuple(obst4.vertex[i]) for i in range(6)]
    verts4.append(verts4[0])
    codes = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]
    path0 = Path(verts0, codes)
    patch0 = patches.PathPatch(path0,facecolor='cyan')
    ax1.add_patch(patch0)
    path1 = Path(verts1, codes)
    patch1 = patches.PathPatch(path1,facecolor='cyan')
    ax1.add_patch(patch1)
    path2 = Path(verts2, codes)
    patch2 = patches.PathPatch(path2,facecolor='cyan')
    ax1.add_patch(patch2)
    path3 = Path(verts3, codes)
    patch3 = patches.PathPatch(path3,facecolor='cyan')
    ax1.add_patch(patch3)
    path4 = Path(verts4, codes)
    patch4 = patches.PathPatch(path4,facecolor='cyan')
    ax1.add_patch(patch4)

    #
    ax1.plot([cfg0[0], cfg1[0], cfg2[0], cfg3[0], cfg4[0]], [cfg0[1], cfg1[1], cfg2[1], cfg3[1], cfg4[1]], 'ro')
    for q1 in [cfg1, cfg2]:
        p, r = TG.calc_path(cursor, cfg0, q1)
        line = TG.spiral3_calc(p, r=r, q=cfg0)
        ax1.plot(line[:,1], line[:,2], color='magenta', linewidth=2)
    for q0 in [cfg1, cfg2]:
        for q1 in [cfg3, cfg4]:
            p, r = TG.calc_path(cursor, q0, q1)
            line = TG.spiral3_calc(p, r=r, q=q0)
            ax1.plot(line[:,1], line[:,2], color='magenta', linewidth=2)
    #
    plt.xlabel('$x (m)$', fontsize=20)
    plt.ylabel('$y (m)$', fontsize=20)
    plt.axis('equal')
    # plt.axis('off')
    # plt.savefig('img/road_grid.png', dpi=600)
    # plt.savefig('img/road_shape.png', dpi=600)
    # plt.savefig('img/road_spiral3.png', dpi=600)
    plt.show()

    cursor.close()
    conn.close()


def test_reverse_traj():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    state_start = State(time=0., length =0., x=0.,y=0.,theta=0., k=0.01, v=-2.,acc=-0.1)
    state_goal = State(x=-20.,y=-5.,theta=-np.pi/6., k=0.01, v=-2.)
    traj = trajectory_reverse(state_start, state_goal, cursor)

    if traj is not None:
        plt.plot(traj[:,2], traj[:,3])
        plt.axis('equal')
        plt.show()
    else:
        print("Failed")

    cursor.close()
    conn.close()


def test_open():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    state_start = State(time=0., length =0., x=40.,y=40.,theta=0., k=0., v=0.,acc=0.)
    state_dict = {}
    traj_dict = {}
    outs = state_start.successors_open(state_dict, traj_dict, cursor, times=[2.])
    print(len(outs))

    for state in outs:
        plt.plot(state.x, state.y, 'ro')
    plt.plot(state_start.x, state_start.y, 'ro')
    for _, traj in traj_dict.items():
        plt.plot(traj[:,2], traj[:,3])
    plt.show()

    # state_goal = State(x=45., y=39., theta=0., k=0., v=1.)
    # traj = trajectory(state_start, state_goal, cursor)
    # if traj is not None:
    #     plt.plot(traj[:,2], traj[:,3])
    #     plt.axis('equal')
    #     plt.show()
    # else:
    #     print("Trajectory not exists")


    cursor.close()
    conn.close()



def test_transition():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, s=70., q=(5.,30.,np.pi/9))
    road = Road(center_line, ref_grid_width=0.5, ref_grid_length=1.)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax1.plot(center_line[:,1], center_line[:,2], color='red', linestyle='--', linewidth=2.)
    ax1.plot(center_line[:,1], center_line[:,2], color='red', linewidth=1.)
    ax1.plot(road.lateral_lines[:,0], road.lateral_lines[:,1],color='red', linewidth=1.)
    ax1.plot(road.lateral_lines[:,-2], road.lateral_lines[:,-1],color='green', linewidth=1.)

    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.)
        else:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=0.3)
    for i in range(road.grid_num_longitudinal+1):
        ax1.plot(road.lateral_lines[:,2*i], road.lateral_lines[:,2*i+1],color='black', linewidth=0.3)
    #
    cfg0 = road.ij2xy(4, 0)
    veh = Vehicle(trajectory=np.array([[-1.,-1.,cfg0[0], cfg0[1], cfg0[2], cfg0[3], 0.,5.,0.]]))
    # verts0 = [tuple(veh.vertex[i]) for i in range(6)]
    # verts0.append(verts0[0])
    # codes = [Path.MOVETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.LINETO,
    #     Path.CLOSEPOLY,
    #     ]
    # path0 = Path(verts0, codes)
    # patch0 = patches.PathPatch(path0,facecolor='cyan')
    # ax1.add_patch(patch0)
    #
    cm = np.zeros((500,500))
    hm = np.zeros((500,500))
    goal = State(road=road,r_i=15,r_j=-2,v=6.)
    goal2 = State(road=road,r_s=goal.r_s+0.3,r_l=goal.r_l+0.1,v=6.)
    goal1 = State(road=road,r_s=goal.r_s+0.7,r_l=goal.r_l+0.3,v=6.)
    ax1.text(goal1.x, goal1.y+0.4, 'g1', fontsize=15) 
    ax1.text(goal2.x, goal2.y-0.4, 'g2', fontsize=15) 
    ax1.plot(goal1.x, goal1.y, 'mo')
    ax1.plot(goal2.x, goal2.y, 'mo')
    state1 = State(time=0.,length=0.,road=road,r_i=4,r_j=0,v=5.,cost=0., heuristic_map=hm)
    state2 = State(time=0.,length=0.,road=road,r_i=5,r_j=-4,v=5.,cost=0., heuristic_map=hm)
    ax1.text(state1.x, state1.y+0.4, 's1', fontsize=15) 
    ax1.text(state2.x, state2.y-0.4, 's2', fontsize=15) 
    ax1.plot(state1.x, state1.y, 'mo')
    ax1.plot(state2.x, state2.y, 'mo')
    # ax1.plot(state.x, state.y, 'rs')
    # for s1 in [state1, state2]:
    #     for s2 in [goal1,goal2]:
    #         traj = trajectory(s1,s2,cursor)
    #         if traj is not None:
    #             ax1.plot(traj[:,2], traj[:,3], linewidth=2)
    #             ax1.plot(traj[0,2], traj[0,3], 'mo')
    #             ax1.plot(traj[-1,2], traj[-1,3], 'mo')
    traj1 = trajectory(state1,goal1,cursor)
    traj2 = trajectory(state2,goal2,cursor)
    traj3 = trajectory(state2,goal1,cursor)
    ax1.plot(traj1[:,2], traj1[:,3], linewidth=2., color='red')
    ax1.plot(traj2[:,2], traj2[:,3], linewidth=2., color='black')
    ax1.plot(traj3[:,2], traj3[:,3], 'b--', linewidth=2.)
    # node_dict = {}
    # # successors = state.successors(state_dict=node_dict, road=road, goal=goal, vehicle=veh, heuristic_map=hm, accs=[-3., -2., -1., 0., 1.], v_offset=[-1.,-0.5, 0., 0.5, 1.],times=[3.])
    # successors = state.successors(state_dict=node_dict, road=road, goal=goal, vehicle=veh, heuristic_map=hm, times = [2.5]) # accs=[ 0.], v_offset=[-1.,-0.5, 0., 0.5, 1.],times=[1.,2.,4.]
    # print("number of successors: {0}".format(len(successors)))
    # for successor in successors:
    #     # p, r = TG.calc_path(cursor, state.q, successor.q)
    #     # if p is not None and p[4]>0:
    #     #     line = TG.spiral3_calc(p, r=r, q=state.q)
    #     #     ax1.plot(successor.x, successor.y, 'mo')
    #     #     ax1.plot(line[:,1], line[:,2], linewidth=2)
    #     traj1 = trajectory(state,successor,cursor)
    #     if traj1 is not None:
    #         _, traj, _ = TG.eval_trajectory(traj1,costmap=cm,road=road)
    #         if traj is not None:
    #             ax1.plot(traj[:,2], traj[:,3], linewidth=2)
    #             ax1.plot(traj[-1,2], traj[-1,3], 'mo')
    #
    # ax1.plot([cfg0[0], cfg1[0], cfg2[0], cfg3[0], cfg4[0]], [cfg0[1], cfg1[1], cfg2[1], cfg3[1], cfg4[1]], 'ro')
    # for q1 in [cfg1, cfg2]:
    #     p, r = TG.calc_path(cursor, cfg0, q1)
    #     line = TG.spiral3_calc(p, r=r, q=cfg0)
    #     ax1.plot(line[:,1], line[:,2], color='magenta', linewidth=2)
    # for q0 in [cfg1, cfg2]:
    #     for q1 in [cfg3, cfg4]:
    #         p, r = TG.calc_path(cursor, q0, q1)
    #         line = TG.spiral3_calc(p, r=r, q=q0)
    #         ax1.plot(line[:,1], line[:,2], color='magenta', linewidth=2)
    #
    plt.xlabel('$x (m)$', fontsize=20)
    plt.ylabel('$y (m)$', fontsize=20)
    plt.axis('equal')
    # plt.axis('off')
    # plt.savefig('img/road_grid.png', dpi=600)
    # plt.savefig('img/road_shape.png', dpi=600)
    # plt.savefig('img/road_spiral3.png', dpi=600)
    plt.show()

    cursor.close()
    conn.close()




def test_workspace():
    # veh = Vehicle(trajectory=np.array([[-1.,-1.,2.,30.,0.,0.,0.,0.,0.,0.]]))
    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, s=100., q=(2.,15.,0))
    road = Road(center_line=center_line)
    #
    cfg = road.ij2xy(5,-4)
    traj = np.array([[-1,-1,cfg[0],cfg[1],cfg[2],cfg[3],0.,0.,0.,0.]])
    veh = Vehicle(trajectory=traj)
    #
    lane_costs = [0.3,0.6,0.9]
    #static_obsts
    cs1 = road.ij2xy(20,8)
    cs2 = road.ij2xy(50,-6)
    traj1 = np.array([[-1,-1,cs1[0],cs1[1],cs1[2],cs1[3],0.,0.,0.,0.]])
    traj2 = np.array([[-1,-1,cs2[0],cs2[1],cs2[2],cs2[3],0.,0.,0.,0.]])
    ob1 = Vehicle(trajectory=traj1)
    ob2 = Vehicle(trajectory=traj2)
    obsts = [ob1,ob2]
    #
    ws = Workspace(vehicle=veh, road=road, lane_costs=lane_costs, static_obsts=obsts)
    # test_disk = ws.disk_filter(2)
    # test_veh = ws.vehicle_filter(theta = np.pi/4)
    #
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    lane_map = 0.2*ws.lane_grids[0] + 0.4*ws.lane_grids[1]
    lane_map = np.where(lane_map > 0.4, 0.4, lane_map)
    lane_map += 0.6*ws.lane_grids[2]
    lane_map = np.where(lane_map > 0.6, 0.6, lane_map)


    ax1.imshow(lane_map, cmap=plt.cm.Greens, origin="lower",extent=(0.,100.,0.,100.))
    # ax1.colorbar()
    import pylab as pl
    mycmap = plt.cm.Greens
    mynames=['low cost dense','intermediate cost dense','high cost dense']
    for entry in [0.2,0.4,0.6]:
        mycolor = mycmap(255)
        pl.plot(0, 0, "-", c=mycolor, alpha=entry, label=mynames[int(entry*5-1)], linewidth=8)
    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=2.)
        # else:
        #     ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=1.)
    # for i in range(road.grid_num_longitudinal+1):
    #     ax1.plot(road.lateral_lines[:,2*i], road.lateral_lines[:,2*i+1],color='black', linewidth=0.3)
    #ax1.imshow(ws.cost_filter, cmap=plt.cm.Greens, origin="lower",extent=(0.,ws.resolution*ws.cost_filter.shape[1],0.,ws.resolution*ws.cost_filter.shape[0]))
    # np.savetxt('cost_filter.txt',ws.cost_filter,fmt='%i',delimiter=' ')
    # plt.savefig('img/road_cost.png', dpi=600)
    # plt.axis('equal')
    plt.legend()
    plt.xlabel('$x (m)$', fontsize=20)
    plt.ylabel('$y (m)$', fontsize=20)

    # plt.savefig('img/road_cost.png', dpi=600)
    plt.show()


def test_traj():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()
    q0 = (100.,100.,0., 0.02)
    q1 = (125.,130.,np.pi/6,-0.02)
    p,r = TG.calc_path(cursor,q0,q1)
    print('p={0},r={1}'.format(p,r))
    cursor.close()
    conn.close()


def test_database():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()
    k0,k1=0.,0.
    cursor.execute('select p1,p2,sg from InitialGuessTable where k0=? and k1=? and y1>=0 and theta1>=0',(int(k0*40),int(k1*40)))
    pps = cursor.fetchall()
    t=0
    for pp in pps:
        if pp[2]>0:
            t+=1
            path = TG.spiral3_calc(p=(k0,pp[0],pp[1],k1,pp[2]))
            plt.plot(path[:,1],path[:,2])
    print(t)
    plt.title('k0 = {0}, k1 = {1}'.format(k0,k1))
    plt.axis('equal')
    plt.savefig('img/initialguesstable(k0={0}k1={1}).png'.format(k0,k1),dpi=600)
    plt.show()
    cursor.close()
    conn.close()


def test_coordinate_trasform():
    from math import pi 
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    q0 = (5., 0., pi/30, -0.01)
    q1 = (55., 20., pi/30, 0.01)

    p, r = TG.calc_path(cursor, q0, q1)
    print('p={0},r={1}'.format(p,r))

    sg = p[4]
    # print(sg)

    line = TG.spiral3_calc(p, r=r, q=q0)
    print(line[-1,:])
    P0 = q0[0:2]
    P1 = TG.spiral3_calc(p, r=r, s=sg/3, q=q0)[-1, 1:3]
    P2 = TG.spiral3_calc(p, r=r, s=2*sg/3, q=q0)[-1, 1:3]
    P3 = TG.spiral3_calc(p, r=r, s=sg, q=q0)[-1, 1:3]

    # plt.figure(figsize=(80,60))
    plt.plot(line[:,1], line[:,2], color='black', linewidth=4)
    # plt.plot(P0[0], P0[1], 'go', linewidth=4)
    # plt.plot(P1[0], P1[1], 'go', linewidth=4)
    # plt.plot(P2[0], P2[1], 'go', linewidth=4)
    # plt.plot(P3[0], P3[1], 'go', linewidth=4)
    plt.scatter(P0[0], P0[1], s=200)
    plt.scatter(P1[0], P1[1], s=200)
    plt.scatter(P2[0], P2[1], s=200)
    plt.scatter(P3[0], P3[1], s=200)
    plt.xlabel('$x (m)$', fontsize=40)
    plt.ylabel('$y (m)$', fontsize=40)
    plt.text(P0[0]-3, P0[1]+3,'$p_0=k(0)$', fontsize=40)
    plt.text(P1[0]-3, P1[1]-3,'$p_1=k(\\frac{s_g}{3})$', fontsize=40)
    plt.text(P2[0]-3, P2[1]-3,'$p_2=k(\\frac{2s_g}{3})$', fontsize=40)
    plt.text(P3[0]-3, P3[1]-3,'$p_3=k(s_g)$', fontsize=40)

    plt.axis('equal')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # plt.savefig('img/coordinate_transform.png',dpi=600)
    plt.show()

    cursor.close()
    conn.close()


def test_diff_init_val():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    q0 = (20., 20., -np.pi/6., -0.01)
    q1 = (70., 30., np.pi/6., -0.01)

    p1, r1 = TG.calc_path(cursor, q0, q1)
    p2, r2 = TG.calc_path_no_init_val(q0,q1)
    print('p={0},r={1}'.format(p1,r1))
    print('p={0},r={1}'.format(p2,r2))

    
    line1 = TG.spiral3_calc(p1, r=r1, q=q0)
    line2 = TG.spiral3_calc(p2, r=r2, q=q0)
    
    # plt.figure(figsize=(80,60))
    plt.plot(line1[:,1], line1[:,2], color='black', linewidth=4)
    plt.plot(line2[:,1], line2[:,2], color='black', linewidth=4)
    
    plt.xlabel('$x (m)$', fontsize=40)
    plt.ylabel('$y (m)$', fontsize=40)
    # plt.text(P0[0]-3, P0[1]+3,'$p_0=k(0)$', fontsize=40)
    # plt.text(P1[0]-3, P1[1]-3,'$p_1=k(\\frac{s_g}{3})$', fontsize=40)
    # plt.text(P2[0]-3, P2[1]-3,'$p_2=k(\\frac{2s_g}{3})$', fontsize=40)
    # plt.text(P3[0]-3, P3[1]-3,'$p_3=k(s_g)$', fontsize=40)

    plt.axis('equal')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # plt.savefig('img/coordinate_transform.png',dpi=600)
    plt.show()

    cursor.close()
    conn.close()


def test_optimize():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    

    bd_con = (-0.01, 70., 30., np.pi/6, 0.01)
    init_val = (0., 0., 100)
    # init_val = (-0.0033333333333333335, 0.0033333333333333335, 78.775724936630581)

    pp1 = TG.optimize(bd_con, init_val=init_val)
    p1 = (bd_con[0], pp1[0], pp1[1], bd_con[4], pp1[2])
    r1 = (TG.__a(p1), TG.__b(p1), TG.__c(p1), TG.__d(p1))
    print('p1 = {0}'.format(p1))




    # q0 = (0., 0., 0., -0.01)
    # q1 = (70., 30., np.pi/6., 0.01)

    # p2, r2 = TG.calc_path(cursor, q0, q1)
    # print('p2 = {0}'.format(p2))


    line1 = TG.spiral3_calc(p1, r=r1, q=(0.,0.,0.))
    # line2 = TG.spiral3_calc(p2, r=r2, q=q0)

    plt.plot(line1[:,1], line1[:,2], color='black', linewidth=4)
    # plt.plot(line2[:,1], line2[:,2], color='green', linewidth=4)

    plt.axis('equal')
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    # plt.savefig('img/coordinate_transform.png',dpi=600)
    plt.show()


    cursor.close()
    conn.close()


def test_cost_fun_discrete():

    from math import pi 
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    q0 = (5., 0., 0., 0.01)
    q1 = (55., 20., pi/6, -0.01)

    p, r = TG.calc_path(cursor, q0, q1)
    print('p={0},r={1}'.format(p,r))

    sg = p[4]


    line = TG.spiral3_calc(p, r=r, q=q0)

    P0 = q0[0:3]
    P1 = TG.spiral3_calc(p, r=r, s=sg/3, q=q0)[-1, 1:4]
    P2 = TG.spiral3_calc(p, r=r, s=2*sg/3, q=q0)[-1, 1:4]
    P3 = TG.spiral3_calc(p, r=r, s=sg, q=q0)[-1, 1:4]

    ####################
    veh0 = Vehicle(trajectory=np.array([[-1.,-1.,P0[0],P0[1],P0[2],0.,0.,0.,0.]]))
    veh1 = Vehicle(trajectory=np.array([[-1.,-1.,P1[0],P1[1],P1[2],0.,0.,0.,0.]]))
    veh2 = Vehicle(trajectory=np.array([[-1.,-1.,P2[0],P2[1],P2[2],0.,0.,0.,0.]]))
    veh3 = Vehicle(trajectory=np.array([[-1.,-1.,P3[0],P3[1],P3[2],0.,0.,0.,0.]]))


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    ax1.plot(line[:,1], line[:,2], color='black', linewidth=4)
    # ax2.plot(line[:,1], line[:,2], color='black', linewidth=4)

    verts0 = [tuple(veh0.vertex[i]) for i in range(6)]
    verts0.append(verts0[0])
    verts1 = [tuple(veh1.vertex[i]) for i in range(6)]
    verts1.append(verts1[0])
    verts2 = [tuple(veh2.vertex[i]) for i in range(6)]
    verts2.append(verts2[0])
    verts3 = [tuple(veh3.vertex[i]) for i in range(6)]
    verts3.append(verts3[0])

    codes = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]

    path0 = Path(verts0, codes)
    path1 = Path(verts1, codes)
    path2 = Path(verts2, codes)
    path3 = Path(verts3, codes)

    patch0 = patches.PathPatch(path0,facecolor='green')
    patch1 = patches.PathPatch(path1,facecolor='green')
    patch2 = patches.PathPatch(path2,facecolor='green')
    patch3 = patches.PathPatch(path3,facecolor='green')

    ax1.add_patch(patch0)
    ax1.add_patch(patch1)
    ax1.add_patch(patch2)
    ax1.add_patch(patch3)

    plt.axis('equal')
    # plt.axis('off')
    plt.show()


    cursor.close()
    conn.close()


def test_different_initval():
    # conn = sqlite3.connect('InitialGuessTable.db')
    # cursor = conn.cursor()

    q0 = (0.,0.,0.,0.07)
    q1 = (43.21, 10.53, 0.47, -0.03)

    # init_val = (-0.011680200827827, 0.027105283154636, 45.369339753738281)
    init_val = (-0.010248903904925, 0.000396612698255, 45.158514552345096)
    bd_con = (0.07, 43.21, 10.53, 0.47, -0.03)
    pp = TG.optimize(bd_con, init_val=init_val)
    print(pp)

    # p, r = TG.calc_path(cursor, q0, q1)
    # print('p={0},r={1}'.format(p,r))


    # line = TG.spiral3_calc(p, r=r, q=q0)
    # print('Goal Configuration: {0}'.format(line[-1,:]))

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # # ax2 = fig.add_subplot(122)

    # ax1.plot(line[:,1], line[:,2], color='black', linewidth=4)

    # plt.axis('equal')
    # # plt.axis('off')
    # plt.show()

    # cursor.close()
    # conn.close()


def test_long_path():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    q0 = (10., 50., 0., 0.)
    q1 = (800., 10., 0., 0)

    p, r = TG.calc_path(cursor, q0, q1)
    print('p={0},r={1}'.format(p,r))

    line = TG.spiral3_calc(p, r=r, q=q0)
    print('Goal Configuration: {0}'.format(line[-1,:]))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    ax1.plot(line[:,1], line[:,2], color='black', linewidth=4)

    plt.axis('equal')
    # plt.axis('off')
    plt.show()

    cursor.close()
    conn.close()


def test_traj_sampling():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    q0 = (10., 50., 0., 0.)
    q1 = (40., 60., 0., 0)

    p, r = TG.calc_path(cursor, q0, q1)
    # print('p={0},r={1}'.format(p,r))

    line = TG.spiral3_calc(p, r=r, q=q0)
    # print('Goal Configuration: {0}'.format(line[-1,:]))
    # print(line.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    ax1.plot(line[:,1], line[:,2], color='navy', linewidth=2.)
    for i in range(65):
        # ax1.plot(line[i*5, 1], line[i*5, 2], 'ro')
        k = floor(324/64**2*i**2)
        ax1.plot(line[k, 1], line[k, 2], 'ro')
    ax1.plot(line[-1, 1], line[-1, 2], 'ro')

    plt.axis('equal')
    # plt.axis('off')
    plt.show()

    cursor.close()
    conn.close()


if __name__ == '__main__':
    # test_vehicle()
    # test_road()
    # test_open()
    test_transition()
    # test_workspace()
    # test()
    # test_traj()
    # test_coordinate_trasform()
    # test_diff_init_val()
    # test_optimize()
    # test_cost_fun_discrete()
    # test_different_initval()
    # test_long_path()
    # test_traj_sampling()
