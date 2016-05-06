from Environment import *
import TrajectoryGeneration as TG
from OnRoadPlanning import State, Astar, trajectory_stay, trajectory_interp
from OnRoadPlanning import trajectory as trajectory_forward
import cv2
from math import ceil, floor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
import matplotlib.patches as patches
import sqlite3
from queue import PriorityQueue
import datetime
import pickle


def senarios_4():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, s=100.,q=(3.,25.,0.))
    road = Road(center_line, ref_grid_width=0.5, ref_grid_length=1.)

    # ax.plot(center_line[:,1], center_line[:,2], color='red', linewidth=1.)
    ax.plot(road.lateral_lines[:,0], road.lateral_lines[:,1],color='green', linewidth=1.)
    ax.plot(road.lateral_lines[:,-2], road.lateral_lines[:,-1],color='green', linewidth=1.)
    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.)

    #
    # ws = Workspace(road=road, lane_costs=[0.4,0.1,0.2])
    # cost_map_base = ws.lane_map
    # ax.imshow(ws.lane_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))

    s1 = State(time=0.,length=0.,road=road,r_s=25.,r_l=road.lane_width, v=15.)
    g1 = State(road=road,r_s=95.,r_l=road.lane_width,v=15.)
    traj1 = trajectory_forward(s1,g1,cursor)
    # print(traj1[-1,:])
    # ax.plot(traj1[:,2], traj1[:,3], color='navy', linewidth=2.)

    s2 = State(time=0.,length=0.,road=road,r_s=20.,r_l=0.,v=6.)
    g2 = State(road=road,r_s=95.,r_l=0.,v=6.)
    traj2 = trajectory_forward(s2,g2,cursor)
    # print(traj2[-1,:])
    # ax.plot(traj2[:,2], traj2[:,3], color='navy', linewidth=2.)

    cfg3 = road.sl2xy(30.,-road.lane_width)
    obst_s = Vehicle(trajectory=np.array([[-1.,-1.,cfg3[0], cfg3[1], cfg3[2], cfg3[3], 0.,0.,0.]]))

    cfg0 = road.sl2xy(5.,0.)
    veh0 = Vehicle(trajectory=np.array([[-1.,-1.,cfg0[0], cfg0[1], cfg0[2], cfg0[3], 0.,0.,0.]]))
    cfg1 = road.sl2xy(90.,0.)
    veh1 = Vehicle(trajectory=np.array([[-1.,-1.,cfg1[0], cfg1[1], cfg1[2], cfg1[3], 0.,0.,0.]]))

    ax.plot(cfg0[0], cfg0[1], 'ko')
    ax.text(cfg0[0], cfg0[1]+0.4, 'Start')
    ax.plot(cfg1[0], cfg1[1], 'ko')
    ax.text(cfg1[0], cfg1[1]+0.4, 'Goal')

    #
    codes6 = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]

    verts_s = [tuple(obst_s.vertex[i]) for i in range(6)]
    verts_s.append(verts_s[0])
    ax.add_patch(patches.PathPatch(Path(verts_s, codes6), facecolor='cyan'))

    verts0 = [tuple(veh0.vertex[i]) for i in range(6)]
    verts0.append(verts0[0])
    ax.add_patch(patches.PathPatch(Path(verts0, codes6), facecolor='green', alpha=0.5))

    verts1 = [tuple(veh1.vertex[i]) for i in range(6)]
    verts1.append(verts1[0])
    ax.add_patch(patches.PathPatch(Path(verts1, codes6), facecolor='red', alpha=0.5))

    for i in range(20):
        state1 = trajectory_interp(traj1, i*0.2)
        state2 = trajectory_interp(traj2, i*0.2)
        if state1 is not None:
            obst_d1 = Vehicle(trajectory=np.array([[-1.,-1.,state1[2], state1[3], state1[4], 0., 0.,0.,0.]]))
            verts_d1 = [tuple(obst_d1.vertex[i]) for i in range(6)]
            verts_d1.append(verts_d1[0])
            ax.add_patch(patches.PathPatch(Path(verts_d1, codes6), facecolor='cyan', alpha=(i+1)/20.))
        if state2 is not None:
            obst_d2 = Vehicle(trajectory=np.array([[-1.,-1.,state2[2], state2[3], state2[4], 0., 0.,0.,0.]]))
            verts_d2 = [tuple(obst_d2.vertex[i]) for i in range(6)]
            verts_d2.append(verts_d2[0])
            ax.add_patch(patches.PathPatch(Path(verts_d2, codes6), facecolor='cyan', alpha=(i+1)/20.))

    #
    # cost_maps = np.zeros((150,500,500))
    # grids_s = ws.grids_occupied_by_polygon(obst_s.vertex)
    # lane_grids = sum(ws.lane_grids)
    # lane_grids = np.where(lane_grids>1.,1., lane_grids)
    # off_road_map = 1. - lane_grids
    # grids_s += off_road_map

    # for i in range(150):
    #     obst_map = np.zeros((500,500))
    #     obst_map += grids_s
    #     state1 = trajectory_interp(traj1, i/10.)
    #     state2 = trajectory_interp(traj2, i/10.)
    #     if state1 is not None:
    #         obst_d1 = Vehicle(trajectory=np.array([[-1.,-1.,state1[2], state1[3], state1[4], 0., 0.,0.,0.]]))
    #         grids_d1 = ws.grids_occupied_by_polygon(obst_d1.vertex)
    #         obst_map += grids_d1
    #     if state2 is not None:
    #         obst_d2 = Vehicle(trajectory=np.array([[-1.,-1.,state2[2], state2[3], state2[4], 0., 0.,0.,0.]]))
    #         grids_d2 = ws.grids_occupied_by_polygon(obst_d2.vertex)
    #         obst_map += grids_d2

    #     collision_map = cv2.filter2D(obst_map, -1, ws.collision_filter)
    #     collision_map = np.where(collision_map>1.e-6, 1., 0.)
    #     cost_map = cv2.filter2D(collision_map, -1, ws.cost_filter)
    #     cost_map += collision_map
    #     cost_map = np.where(cost_map>1., np.inf, cost_map)
    #     cost_map = np.where(cost_map<1.e-8, 0., cost_map)
    #     cost_map += cost_map_base
    #     cost_maps[i,:,:] = cost_map
    # with open('scenario_4/cost_maps.pickle','wb') as f1:  
    #     pickle.dump(cost_maps, f1)




    # plt.xlabel('$x (m)$', fontsize=20)
    # plt.ylabel('$y (m)$', fontsize=20)
    plt.axis('equal')
    # plt.axis('off')
    # plt.savefig('scenario_4/obstacles2.png', dpi=600)
    plt.show()

    cursor.close()
    conn.close()


def test_cost_maps():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, s=100.,q=(3.,25.,0.))
    road = Road(center_line, ref_grid_width=0.5, ref_grid_length=1.)

    cost_maps = np.zeros((150,500,500))
    with open('scenario_4/cost_maps.pickle', 'rb') as f1:
        cost_maps = pickle.load(f1)

    heuristic_maps = np.zeros((150,500,500))
    goal = tuple(road.sl2xy(90.,0.))
    for i in range(150):
        hm = heuristic_map_constructor(goal, cost_maps[i,:,:])
        heuristic_maps[i,:,:] = hm
    # with open('scenario_4/heuristic_maps.pickle','wb') as f2:  
    #     pickle.dump(heuristic_maps, f2)

    fig = plt.figure()
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)
    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)
    ax7 = fig.add_subplot(337)
    ax8 = fig.add_subplot(338)
    ax9 = fig.add_subplot(339)

    ax = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

    for i in range(9):
        # cost_map = cost_maps[int(i*10),:,:]
        # cost_map = np.where(cost_map>np.finfo('d').max, 1., cost_map)
        # ax[i].imshow(cost_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
        heuristic_map = heuristic_maps[int(i*10),:,:]
        ax[i].imshow(heuristic_map, cmap=plt.cm.Blues, origin='lower', extent=(0,100,0,100))
    plt.show()

    cursor.close()
    conn.close()


def scenario_4_sim():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    cost_maps = np.zeros((150,500,500))
    with open('scenario_4/cost_maps.pickle', 'rb') as f1:
        cost_maps = pickle.load(f1)

    heuristic_maps = np.zeros((150,500,500))
    with open('scenario_4/heuristic_maps.pickle', 'rb') as f2:
        heuristic_maps = pickle.load(f2)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, s=100.,q=(3.,25.,0.))
    road = Road(center_line, ref_grid_width=0.5, ref_grid_length=1.)

    # ax.plot(center_line[:,1], center_line[:,2], color='red', linewidth=1.)
    ax.plot(road.lateral_lines[:,0], road.lateral_lines[:,1],color='green', linewidth=1.)
    ax.plot(road.lateral_lines[:,-2], road.lateral_lines[:,-1],color='green', linewidth=1.)
    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.)

    goal = State(road=road, r_s=90., r_l=0., v=10.,static=False)
    start = State(time=0., length=0., road=road, r_s=5., r_l=0., v=10.,cost=0.,heuristic_map=heuristic_maps, static=False)
    veh = Vehicle(trajectory=np.array([[-1.,-1.,start.x, start.y, start.theta, start.k, 0.,0.,0.]]))


    # # weights: weights for (k, dk, v, a, a_c, l, env, j, t, s)
    weights = np.array([5., 10., -0.1, 10., 0.1, 1., 50., 5, 30., -2.])

    starttime = datetime.datetime.now()
    res, state_dict, traj_dict = Astar(start, goal, road, cost_maps, veh, heuristic_maps, cursor, static=False, weights=weights)
    endtime = datetime.datetime.now()

    print((endtime - starttime).seconds)
    print(res)
    print(len(state_dict))
    print(len(traj_dict))

    # for _ , traj in traj_dict.items():
        # ax.plot(traj[:,2], traj[:,3], traj[:,0], color='navy', linewidth=0.3)
        # ax.plot(traj[:,2], traj[:,3], color='blue', linewidth=1.)
    # for _, state in state_dict.items():
    #     if state != start and state != goal:
    #         ax.plot(state.x, state.y, 'go')
            # ax.text(state.x, state.y,'{0:.2f}'.format(state.cost))
    state = goal
    rows = 0
    while state.parent is not None:
        traj = traj_dict[(state.parent, state)]
        ax.plot(traj[:,2], traj[:,3], color='magenta', linewidth=3.)
        rows += traj.shape[0]
        ax.plot(state.x, state.y, 'go')
        ax.plot(state.parent.x, state.parent.y, 'go')
        state = state.parent
        # ax.plot(traj[:,2], traj[:,3], traj[:,0], color='teal', linewidth=1.)
        
        # ax.plot(traj[:,0], traj[:,7], color='black', linewidth=0.5)
    # print(rows)
    final_traj=np.zeros((rows,9))
    state = goal
    # row = 0
    while state.parent is not None:
        traj = traj_dict[(state.parent, state)]
        print(traj.shape[0])
        final_traj[(rows-traj.shape[0]):rows,:] = traj
        rows -= traj.shape[0]
        # row += traj.shape[0]
        state = state.parent
    with open('scenario_4/final_traj.pickle','wb') as f3:  
        pickle.dump(final_traj, f3)

    #
    #################
    s1 = State(time=0.,length=0.,road=road,r_s=25.,r_l=road.lane_width, v=15.)
    g1 = State(road=road,r_s=95.,r_l=road.lane_width,v=15.)
    traj1 = trajectory_forward(s1,g1,cursor)
    # print(traj1[-1,:])
    # ax.plot(traj1[:,2], traj1[:,3], color='navy', linewidth=2.)

    s2 = State(time=0.,length=0.,road=road,r_s=20.,r_l=0.,v=6.)
    g2 = State(road=road,r_s=95.,r_l=0.,v=6.)
    traj2 = trajectory_forward(s2,g2,cursor)
    # print(traj2[-1,:])
    # ax.plot(traj2[:,2], traj2[:,3], color='navy', linewidth=2.)

    cfg3 = road.sl2xy(30.,-road.lane_width)
    obst_s = Vehicle(trajectory=np.array([[-1.,-1.,cfg3[0], cfg3[1], cfg3[2], cfg3[3], 0.,0.,0.]]))

    codes6 = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]

    verts_s = [tuple(obst_s.vertex[i]) for i in range(6)]
    verts_s.append(verts_s[0])
    ax.add_patch(patches.PathPatch(Path(verts_s, codes6), facecolor='cyan'))

    for i in range(31):
        state1 = trajectory_interp(traj1, i*goal.time/30)
        state2 = trajectory_interp(traj2, i*goal.time/30)
        state3 = trajectory_interp(final_traj, i*goal.time/30)
        if state1 is not None:
            obst_d1 = Vehicle(trajectory=np.array([[-1.,-1.,state1[2], state1[3], state1[4], 0., 0.,0.,0.]]))
            verts_d1 = [tuple(obst_d1.vertex[i]) for i in range(6)]
            verts_d1.append(verts_d1[0])
            ax.add_patch(patches.PathPatch(Path(verts_d1, codes6), facecolor='cyan', alpha=0.1+0.03*i))
        if state2 is not None:
            obst_d2 = Vehicle(trajectory=np.array([[-1.,-1.,state2[2], state2[3], state2[4], 0., 0.,0.,0.]]))
            verts_d2 = [tuple(obst_d2.vertex[i]) for i in range(6)]
            verts_d2.append(verts_d2[0])
            ax.add_patch(patches.PathPatch(Path(verts_d2, codes6), facecolor='cyan', alpha=0.1+0.03*i))
        if state3 is not None:
            obst_d3 = Vehicle(trajectory=np.array([[-1.,-1.,state3[2], state3[3], state3[4], 0., 0.,0.,0.]]))
            verts_d3 = [tuple(obst_d3.vertex[i]) for i in range(6)]
            verts_d3.append(verts_d3[0])
            ax.add_patch(patches.PathPatch(Path(verts_d3, codes6), facecolor='blue', alpha=0.1+0.03*i))




    plt.axis('equal')
    # plt.axis('off')
    # plt.savefig('scenario_4/obstacles2.png', dpi=600)
    plt.show()

    cursor.close()
    conn.close()

def senarios_1():
    # database connection
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    # plot
    fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(212)

    # road center line points
    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, s=100.,q=(3.,25.,0.))
    # p = (0.,0.,0.,0.,90.) # (p0~p3, sg)
    # center_line = TG.spiral3_calc(p, q=(5.,50.,0.))
    # np.savetxt('scenario_1/road_center_line.txt', center_line, delimiter='\t')
    # print(center_line)

    # road
    road = Road(center_line)

    ax1.plot(center_line[:,1], center_line[:,2], color='maroon', linestyle='--', linewidth=1.)

    ax1.plot(road.lateral_lines[:,0], road.lateral_lines[:,1],color='green', linewidth=1.)
    ax1.plot(road.lateral_lines[:,-2], road.lateral_lines[:,-1],color='green', linewidth=1.)
    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.)
        # else:
        #     ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=0.3)
    # for i in range(road.grid_num_longitudinal+1):
    #     ax1.plot(road.lateral_lines[:,2*i], road.lateral_lines[:,2*i+1],color='black', linewidth=0.3)

    # vehicle
    cfg0 = road.sl2xy(5.,0.)
    cfg5 = road.sl2xy(90.,0.)
    veh = Vehicle(trajectory=np.array([[-1.,-1.,cfg0[0], cfg0[1], cfg0[2], cfg0[3], 0.,5.,0.]]))
    veh5 = Vehicle(trajectory=np.array([[-1.,-1.,cfg5[0], cfg5[1], cfg5[2], cfg5[3], 0.,5.,0.]]))

    # ax1.plot(cfg0[0], cfg0[1], 'ko')
    # ax1.text(cfg0[0], cfg0[1]+0.4, 'Start')
    # # ax1.plot(cfg5[0], cfg5[1], 'ko')
    # ax1.text(cfg5[0], cfg5[1]+0.4, 'Goal')

    codes6 = [Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]
    # verts0 = [tuple(veh.vertex[i]) for i in range(6)]
    # verts0.append(verts0[0])
    # ax1.add_patch(patches.PathPatch(Path(verts0, codes6), facecolor='green', alpha=0.5))

    # verts5 = [tuple(veh5.vertex[i]) for i in range(6)]
    # verts5.append(verts5[0])
    # ax1.add_patch(patches.PathPatch(Path(verts5, codes6), facecolor='red', alpha=0.5))


    # workspace
    ws = Workspace(vehicle=veh, road=road)
    road_lane_bitmap0 = ws.lane_grids[0]
    road_lane_bitmap1 = ws.lane_grids[1]
    road_lane_bitmap2 = ws.lane_grids[2]
    # write the lane bitmaps into files
    # np.savetxt('road_lane_bitmap0.txt', road_lane_bitmap0, fmt='%i',delimiter=' ')
    # np.savetxt('road_lane_bitmap1.txt', road_lane_bitmap1, fmt='%i',delimiter=' ')
    # np.savetxt('road_lane_bitmap2.txt', road_lane_bitmap2, fmt='%i',delimiter=' ')
    # road bitmap
    road_bitmap = road_lane_bitmap0 + road_lane_bitmap1 + road_lane_bitmap2
    road_bitmap = np.where(road_bitmap>1.e-6, 1., 0.)
    # np.savetxt('road_bitmap.txt', road_bitmap, fmt='%i', delimiter=' ')
    # base bitmap
    base = 1. - road_bitmap
    # base = np.where(base>1.e-6, np.inf, 0)
    # np.savetxt('base_bitmap.txt', base, fmt='%i', delimiter=' ')

    # static obstacles
    cfg1 = road.sl2xy(30., 0.)
    cfg2 = road.sl2xy(30., -road.lane_width)
    cfg3 = road.sl2xy(65.,0.)
    cfg4 = road.sl2xy(65., road.lane_width)
    obst1 = Vehicle(trajectory=np.array([[-1.,-1.,cfg1[0], cfg1[1], cfg1[2], cfg1[3], 0.,0.,0.]]))
    obst2 = Vehicle(trajectory=np.array([[-1.,-1.,cfg2[0], cfg2[1], cfg2[2], cfg2[3], 0.,0.,0.]]))
    obst3 = Vehicle(trajectory=np.array([[-1.,-1.,cfg3[0], cfg3[1], cfg3[2], cfg3[3], 0.,0.,0.]]))
    obst4 = Vehicle(trajectory=np.array([[-1.,-1.,cfg4[0], cfg4[1], cfg4[2], cfg4[3], 0.,0.,0.]]))

    verts1 = [tuple(obst1.vertex[i]) for i in range(6)]
    verts1.append(verts1[0])
    ax1.add_patch(patches.PathPatch(Path(verts1, codes6), facecolor='cyan'))
    verts2 = [tuple(obst2.vertex[i]) for i in range(6)]
    verts2.append(verts2[0])
    ax1.add_patch(patches.PathPatch(Path(verts2, codes6), facecolor='cyan'))
    verts3 = [tuple(obst3.vertex[i]) for i in range(6)]
    verts3.append(verts3[0])
    ax1.add_patch(patches.PathPatch(Path(verts3, codes6), facecolor='cyan'))
    verts4 = [tuple(obst4.vertex[i]) for i in range(6)]
    verts4.append(verts4[0])
    ax1.add_patch(patches.PathPatch(Path(verts4, codes6), facecolor='cyan'))

    base += ws.grids_occupied_by_polygon(obst1.vertex)
    base += ws.grids_occupied_by_polygon(obst2.vertex)
    base += ws.grids_occupied_by_polygon(obst3.vertex)
    base += ws.grids_occupied_by_polygon(obst4.vertex)
    base = np.where(base>1.e-6, 1.,0.)
    # np.savetxt('scenario_1/static_bitmap.txt', base, fmt='%i', delimiter=' ')
    
    # collision map
    collision_map = cv2.filter2D(base, -1, ws.collision_filter)
    collision_map = np.where(collision_map>1.e-6, 1., 0.)
    # np.savetxt('scenario_1/collision_bitmap.txt', collision_map, fmt='%i', delimiter=' ')

    # cost map
    cost_map = cv2.filter2D(collision_map, -1, ws.cost_filter)
    cost_map += collision_map
    cost_map = np.where(cost_map>1., np.inf, cost_map)
    cost_map = np.where(cost_map<1.e-8, 0., cost_map)
    # costmap_save = np.where( cost_map >1., -1., cost_map)
    # np.savetxt('scenario_1/cost_map.txt', costmap_save, delimiter='\t')

    # plot
    # costmap_plot = np.where( cost_map >1., 1., cost_map)
    # ax1.imshow(costmap_plot, cmap=plt.cm.Reds, origin="lower",extent=(0.,ws.resolution*ws.row,0.,ws.resolution*ws.column))
    
    
    # heuristic map
    goal_state = State(road=road, r_s=90., r_l=0., v=10.)
    # ax1.scatter(goal_state.x, goal_state.y, c='r')
    ax1.plot(goal_state.x, goal_state.y, 'rs')

    heuristic_map = heuristic_map_constructor(goal_state, cost_map)
    # hm_save = np.where(heuristic_map > np.finfo('d').max, -1., heuristic_map)
    # np.savetxt('scenario_1/heuristic_map.txt', hm_save, delimiter='\t')

    start_state = State(time=0., length=0., road=road, r_s=5., r_l=0., v=10.,cost=0., heuristic_map=heuristic_map)
    # ax1.scatter(start_state.x, start_state.y, c='r')
    ax1.plot(start_state.x, start_state.y, 'rs')
    # ax1.imshow(heuristic_map, cmap=plt.cm.Reds, origin="lower",extent=(0.,ws.resolution*ws.row,0.,ws.resolution*ws.column))

    # # weights: weights for (k, dk, v, a, a_c, l, env, j, t, s)
    weights = np.array([5., 10., -0.1, 10., 0.1, 0.1, 50., 5, 40., -4.])

    starttime = datetime.datetime.now()
    res, state_dict, traj_dict = Astar(start_state, goal_state, road, cost_map, veh, heuristic_map, cursor, weights=weights)
    endtime = datetime.datetime.now()

    print((endtime - starttime).seconds)
    print(res)
    print(len(state_dict))
    print(len(traj_dict))
    # print(goal_state.time, goal_state.length, goal_state.cost, start_state.heuristic, goal_state.heuristic)
    # # True
    # 168
    # 175
    # 8.78814826688 76.409797813 2701.06684421 1559.33663366 0.0

    # for _ , traj in traj_dict.items():
        # ax1.plot(traj[:,2], traj[:,3], traj[:,0], color='navy', linewidth=0.3)
        # ax1.plot(traj[:,2], traj[:,3], color='navy', linewidth=1.)
    # for _, state in state_dict.items():
    #     if state != start_state and state != goal_state:
    #         ax1.plot(state.x, state.y, 'go')
            # ax1.text(state.x, state.y,'{0:.2f}'.format(state.cost))
    state = goal_state
    rows = 0 
    while state.parent is not None:
        traj = traj_dict[(state.parent, state)]
        ax1.plot(traj[:,2], traj[:,3], color='magenta', linewidth=3.)
        rows += traj.shape[0]
        ax1.plot(state.x, state.y, 'go')
        ax1.plot(state.parent.x, state.parent.y, 'go')
        state = state.parent

    final_traj=np.zeros((rows,9))
    state = goal_state
    while state.parent is not None:
        traj = traj_dict[(state.parent, state)]
        final_traj[(rows-traj.shape[0]):rows,:] = traj
        rows -= traj.shape[0]
        state = state.parent

    # with open('scenario_1/final_traj.pickle','wb') as f3:  
    #     pickle.dump(final_traj, f3)

    for i in range(31):
        state1 = trajectory_interp(final_traj, i*goal_state.time/30)
        if state1 is not None:
            obst_d1 = Vehicle(trajectory=np.array([[-1.,-1.,state1[2], state1[3], state1[4], 0., 0.,0.,0.]]))
            verts_d1 = [tuple(obst_d1.vertex[i]) for i in range(6)]
            verts_d1.append(verts_d1[0])
            ax1.add_patch(patches.PathPatch(Path(verts_d1, codes6), facecolor='blue', alpha=0.1+0.03*i))


    # close database connection
    cursor.close()
    conn.close()

    #
    # plt.legend()
    plt.axis('equal')
    # plt.savefig('scenario_1/astar_3.png', dpi=600)
    plt.show()


def env_plot():
    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # road center line points
    p = (0.,0.,0.,0.,90.) # (p0~p3, sg)
    center_line = TG.spiral3_calc(p, q=(5.,50.,0.))
    # print(center_line)

    # road
    road = Road(center_line)

    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.5)
        else:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=0.3)
    for i in range(road.grid_num_longitudinal+1):
        ax1.plot(road.lateral_lines[:,2*i], road.lateral_lines[:,2*i+1],color='black', linewidth=0.3)

    # vehicle
    cfg0 = road.sl2xy(5.,0.)
    veh = Vehicle(trajectory=np.array([[-1.,-1.,cfg0[0], cfg0[1], cfg0[2], cfg0[3], 0.,5.,0.]]))


    # workspace
    ws = Workspace(vehicle=veh, road=road)

    # static obstacles
    cfg1 = road.sl2xy(25., 0.)
    cfg2 = road.sl2xy(25., -road.lane_width)
    cfg3 = road.sl2xy(55.,0.)
    cfg4 = road.sl2xy(55., road.lane_width)
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
    path1 = Path(verts1, codes)
    patch1 = patches.PathPatch(path1,facecolor='green')
    ax1.add_patch(patch1)
    path2 = Path(verts2, codes)
    patch2 = patches.PathPatch(path2,facecolor='green')
    ax1.add_patch(patch2)
    path3 = Path(verts3, codes)
    patch3 = patches.PathPatch(path3,facecolor='green')
    ax1.add_patch(patch3)
    path4 = Path(verts4, codes)
    patch4 = patches.PathPatch(path4,facecolor='green')
    ax1.add_patch(patch4)
    plt.axis('equal')
    plt.show()


def costmap_plot():
    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # road center line points
    p = (0.,0.,0.,0.,90.) # (p0~p3, sg)
    center_line = TG.spiral3_calc(p, q=(5.,50.,0.))
    # print(center_line)

    # road
    road = Road(center_line)

    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.5)
        else:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=0.3)
    for i in range(road.grid_num_longitudinal+1):
        ax1.plot(road.lateral_lines[:,2*i], road.lateral_lines[:,2*i+1],color='black', linewidth=0.3)

    # vehicle
    cfg0 = road.sl2xy(5.,0.)
    veh = Vehicle(trajectory=np.array([[-1.,-1.,cfg0[0], cfg0[1], cfg0[2], cfg0[3], 0.,5.,0.]]))


    # workspace
    ws = Workspace(vehicle=veh, road=road)
    road_lane_bitmap0 = ws.lane_grids[0]
    road_lane_bitmap1 = ws.lane_grids[1]
    road_lane_bitmap2 = ws.lane_grids[2]
    # write the lane bitmaps into files
    # np.savetxt('road_lane_bitmap0.txt', road_lane_bitmap0, fmt='%i',delimiter=' ')
    # np.savetxt('road_lane_bitmap1.txt', road_lane_bitmap1, fmt='%i',delimiter=' ')
    # np.savetxt('road_lane_bitmap2.txt', road_lane_bitmap2, fmt='%i',delimiter=' ')
    # road bitmap
    road_bitmap = road_lane_bitmap0 + road_lane_bitmap1 + road_lane_bitmap2
    road_bitmap = np.where(road_bitmap>1.e-6, 1., 0.)
    # np.savetxt('road_bitmap.txt', road_bitmap, fmt='%i', delimiter=' ')
    # base bitmap
    base = 1. - road_bitmap
    # base = np.where(base>1.e-6, np.inf, 0)
    # np.savetxt('base_bitmap.txt', base, fmt='%i', delimiter=' ')

    # static obstacles
    cfg1 = road.sl2xy(25., 0.)
    cfg2 = road.sl2xy(25., -road.lane_width)
    cfg3 = road.sl2xy(55.,0.)
    cfg4 = road.sl2xy(55., road.lane_width)
    obst1 = Vehicle(trajectory=np.array([[-1.,-1.,cfg1[0], cfg1[1], cfg1[2], cfg1[3], 0.,0.,0.]]))
    obst2 = Vehicle(trajectory=np.array([[-1.,-1.,cfg2[0], cfg2[1], cfg2[2], cfg2[3], 0.,0.,0.]]))
    obst3 = Vehicle(trajectory=np.array([[-1.,-1.,cfg3[0], cfg3[1], cfg3[2], cfg3[3], 0.,0.,0.]]))
    obst4 = Vehicle(trajectory=np.array([[-1.,-1.,cfg4[0], cfg4[1], cfg4[2], cfg4[3], 0.,0.,0.]]))
    base += ws.grids_occupied_by_polygon(obst1.vertex)
    base += ws.grids_occupied_by_polygon(obst2.vertex)
    base += ws.grids_occupied_by_polygon(obst3.vertex)
    base += ws.grids_occupied_by_polygon(obst4.vertex)
    base = np.where(base>1.e-6, 1.,0.)
    # np.savetxt('scenario_1/static_bitmap.txt', base, fmt='%i', delimiter=' ')
    
    # collision map
    collision_map = cv2.filter2D(base, -1, ws.collision_filter)
    collision_map = np.where(collision_map>1.e-6, 1., 0.)
    # np.savetxt('scenario_1/collision_bitmap.txt', collision_map, fmt='%i', delimiter=' ')

    # cost map
    cost_map = cv2.filter2D(collision_map, -1, ws.cost_filter)
    cost_map += collision_map
    cost_map = np.where(cost_map>1., np.inf, cost_map)
    cost_map = np.where(cost_map<1.e-16, 0., cost_map)
    # np.savetxt('scenario_1/cost_grayscale_map.txt', cost_map, fmt='%1.6f', delimiter='\t')

    # plot
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    costmap_plot = np.where( cost_map >1., 1., cost_map)
    ax1.imshow(costmap_plot, cmap=plt.cm.Reds, origin="lower",extent=(0.,ws.resolution*ws.row,0.,ws.resolution*ws.column))
    ax1.plot(center_line[:,1], center_line[:,2], color='maroon', linestyle='--', linewidth=2.)
    plt.axis('equal')
    plt.show()



def extend_plot():
    # database connection
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # road center line points
    p = (0.,0.,0.,0.,90.) # (p0~p3, sg)
    center_line = TG.spiral3_calc(p, q=(5.,50.,0.))
    # print(center_line)

    # road
    road = Road(center_line)

    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.5)
        else:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=0.3)
    for i in range(road.grid_num_longitudinal+1):
        ax1.plot(road.lateral_lines[:,2*i], road.lateral_lines[:,2*i+1],color='black', linewidth=0.3)

    # vehicle
    cfg0 = road.sl2xy(5.,0.)
    veh = Vehicle(trajectory=np.array([[-1.,-1.,cfg0[0], cfg0[1], cfg0[2], cfg0[3], 0.,5.,0.]]))


    # workspace
    ws = Workspace(vehicle=veh, road=road)
    road_lane_bitmap0 = ws.lane_grids[0]
    road_lane_bitmap1 = ws.lane_grids[1]
    road_lane_bitmap2 = ws.lane_grids[2]
    # write the lane bitmaps into files
    # np.savetxt('road_lane_bitmap0.txt', road_lane_bitmap0, fmt='%i',delimiter=' ')
    # np.savetxt('road_lane_bitmap1.txt', road_lane_bitmap1, fmt='%i',delimiter=' ')
    # np.savetxt('road_lane_bitmap2.txt', road_lane_bitmap2, fmt='%i',delimiter=' ')
    # road bitmap
    road_bitmap = road_lane_bitmap0 + road_lane_bitmap1 + road_lane_bitmap2
    road_bitmap = np.where(road_bitmap>1.e-6, 1., 0.)
    # np.savetxt('road_bitmap.txt', road_bitmap, fmt='%i', delimiter=' ')
    # base bitmap
    base = 1. - road_bitmap
    # base = np.where(base>1.e-6, np.inf, 0)
    # np.savetxt('base_bitmap.txt', base, fmt='%i', delimiter=' ')

    # static obstacles
    cfg1 = road.sl2xy(25., 0.)
    cfg2 = road.sl2xy(25., -road.lane_width)
    cfg3 = road.sl2xy(55.,0.)
    cfg4 = road.sl2xy(55., road.lane_width)
    obst1 = Vehicle(trajectory=np.array([[-1.,-1.,cfg1[0], cfg1[1], cfg1[2], cfg1[3], 0.,0.,0.]]))
    obst2 = Vehicle(trajectory=np.array([[-1.,-1.,cfg2[0], cfg2[1], cfg2[2], cfg2[3], 0.,0.,0.]]))
    obst3 = Vehicle(trajectory=np.array([[-1.,-1.,cfg3[0], cfg3[1], cfg3[2], cfg3[3], 0.,0.,0.]]))
    obst4 = Vehicle(trajectory=np.array([[-1.,-1.,cfg4[0], cfg4[1], cfg4[2], cfg4[3], 0.,0.,0.]]))
    base += ws.grids_occupied_by_polygon(obst1.vertex)
    base += ws.grids_occupied_by_polygon(obst2.vertex)
    base += ws.grids_occupied_by_polygon(obst3.vertex)
    base += ws.grids_occupied_by_polygon(obst4.vertex)
    base = np.where(base>1.e-6, 1.,0.)
    # np.savetxt('scenario_1/static_bitmap.txt', base, fmt='%i', delimiter=' ')
    
    # collision map
    collision_map = cv2.filter2D(base, -1, ws.collision_filter)
    collision_map = np.where(collision_map>1.e-6, 1., 0.)
    # np.savetxt('scenario_1/collision_bitmap.txt', collision_map, fmt='%i', delimiter=' ')

    # cost map
    cost_map = cv2.filter2D(collision_map, -1, ws.cost_filter)
    cost_map += collision_map
    cost_map = np.where(cost_map>1., np.inf, cost_map)
    cost_map = np.where(cost_map<1.e-16, 0., cost_map)
    # np.savetxt('scenario_1/cost_grayscale_map.txt', cost_map, fmt='%1.6f', delimiter='\t')

    # plot
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    costmap_plot = np.where( cost_map >1., 1., cost_map)
    ax1.imshow(costmap_plot, cmap=plt.cm.Reds, origin="lower",extent=(0.,ws.resolution*ws.row,0.,ws.resolution*ws.column))
    ax1.plot(center_line[:,1], center_line[:,2], color='maroon', linestyle='--', linewidth=2.)


    count = 0
    start_state = State(road=road, r_s=5., r_l=0., v=5.)
    ax1.plot(start_state.x, start_state.y, 'rs')
    

    current = start_state
    outs = current.out_set(road)
    for (i,j,v,a) in outs:
        # print(i,j,v,a)
        next_state = State(road=road, r_i=i, r_j=j, v=v)
        # print(current.q, next_state.q)
        p, r = TG.calc_path(cursor, current.q, next_state.q)
        if r is not None:
            if p[4]>0:
                u = TG.calc_velocity(current.v, a, v, p[4])
                if u[3] is not None and u[3]>0:
                    path = TG.spiral3_calc(p,r,q=current.q,ref_delta_s=0.2)
                    traj = TG.calc_trajectory(u,p,r,s=p[4],path=path,q0=current.q, ref_time=current.time, ref_length=current.length)
                    # if next_state == goal_state:
                    #     cost = TG.eval_trajectory(traj, cost_map, vehicle=veh, road=road, truncate=False)
                    # else:
                    cost, traj = TG.eval_trajectory(traj, cost_map, vehicle=veh, road=road)
                    if not np.isinf(cost) and traj is not None:
                        count += 1
                        next_state.update(cost, traj, current, road)
                        # plot
                        ax1.plot(traj[:,2], traj[:,3],  linewidth=1.)
                        ax1.text(traj[-1,2], traj[-1,3],'{0:.2f}'.format(cost))

    
    # close database connection
    cursor.close()
    conn.close()

    #
    # plt.legend()
    plt.axis('equal')
    # plt.savefig('scenario_1/planning_result.png', dpi=600)
    plt.show()





if __name__ == '__main__':
    # env_plot()
    # costmap_plot()
    # extend_plot()
    senarios_1()
    # senarios_4()
    # test_cost_maps()
    # scenario_4_sim()
