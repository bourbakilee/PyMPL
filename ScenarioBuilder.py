from Environment import *
import TrajectoryGeneration as TG
import cv2
from math import ceil, floor
import random
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches
import sqlite3





def senarios_1():
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
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.)
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
    ax1.plot(center_line[:,1], center_line[:,2], color='maroon', linestyle='--', linewidth=1.5)


    count = 0
    traj_dict = {} #

    start_state = State(road=road, r_s=5., r_l=0., v=5.)
    ax1.plot(start_state.x, start_state.y, 'rs')
    goal_state = State(road=road, r_s=80., r_l=0., v=5.)
    ax1.plot(goal_state.x, goal_state.y, 'rs')

    state_list = [start_state]
    while not goal_state.extended and len(state_list)>0:
        # print(len(state_list))
        current = state_list.pop(0)
        outs = current.control_set(road=road, goal=goal_state)
        for next_state in outs:
            p, r = TG.calc_path(cursor, current.q, next_state.q)
            if r is not None and p[4]>0:
                u = TG.calc_velocity(current.v, next_state.a, next_state.v, p[4])
                if u[3] is not None and u[3]>0:
                    path = TG.spiral3_calc(p,r,q=current.q,ref_delta_s=0.2)
                    traj = TG.calc_trajectory(u,p,r,s=p[4],path=path,q0=current.q, ref_time=current.time, ref_length=current.length)
                    if next_state == goal_state:
                        cost = TG.eval_trajectory(traj, cost_map, vehicle=veh, road=road, truncate=False)
                    else:
                        cost, traj = TG.eval_trajectory(traj, cost_map, vehicle=veh, road=road)
                    if not np.isinf(cost) and traj is not None:
                        count += 1
                        next_state.update(cost, traj, current, road)
                        state_list.append(next_state)
                        traj_dict[(current, next_state)] = traj #
                        # plot
                        ax1.plot(traj[:,2], traj[:,3], color='black', linewidth=0.5)
                        # ax1.text(traj[-1,2], traj[-1,3],'{0:.2f}'.format(cost))
    print(count)
    state = goal_state
    while state.parent is not None:
        traj = traj_dict[(state.parent, state)]
        state = state.parent
        ax1.plot(traj[:,2], traj[:,3], color='teal', linewidth=1.)

    
    # close database connection
    cursor.close()
    conn.close()

    #
    # plt.legend()
    plt.axis('equal')
    # plt.savefig('scenario_1/scenario_1.png', dpi=600)
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