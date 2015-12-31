from Environment import *
import TrajectoryGeneration as TG
import cv2
from math import ceil, floor
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches
import sqlite3

def road_center_line():
    pass


def lane_maps():
    pass


def static_map():
    pass


def dynamic_map(time):
    pass


def collision_map():
    pass


def cost_map(time):
    pass


if __name__ == '__main__':
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
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='blue', linewidth=1.)
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
    cfg1 = road.sl2xy(30., 0.)
    cfg2 = road.sl2xy(30., -road.lane_width)
    cfg3 = road.sl2xy(55.,0.)
    cfg4 = road.sl2xy(55, road.lane_width)
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
    cost_map = np.where(cost_map>1., 0.6, cost_map)
    cost_map = np.where(cost_map<1.e-16, 0., cost_map)
    # np.savetxt('scenario_1/cost_grayscale_map.txt', cost_map, fmt='%1.6f', delimiter='\t')

    # plot
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    ax1.imshow(cost_map, cmap=plt.cm.Greens, origin="lower",extent=(0.,ws.resolution*ws.row,0.,ws.resolution*ws.column))
    ax1.plot(center_line[:,1], center_line[:,2], color='red', linestyle='--', linewidth=2.)

    # action space
    # print('q0:{0}'.format(cfg0))
    v0 = 5.
    # a0 = 0.5
    l0 = 0.
    s0 = 5.
    for n1 in [-2.,-1.,0.,1.]: # v1 = v0 + n1* dv/dt
        for n2 in [-2.,-1.,0.,1.,2.]: # l1 = l0 + n2* 0.5
            for n3 in [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]:
                v1 = min(max(v0 + n1*2*n3, 0.), 20.)
                l1 = min(max(l0 + n2/8*n3, -road.width/2), road.width/2)
                s1 = min(max(s0 + (v0+v1)/2, 0.), road.length)
                r_i = int(s1/road.grid_length)
                r_j = int(l1/road.grid_width)
                q1 = road.ij2xy(r_i,r_j)
                p, r = TG.calc_path(cursor, cfg0, q1)
                if r is not None:
                    # print(p,r)
                    spiral3_path = TG.spiral3_calc(p,r, q=cfg0, ref_delta_s=0.2)
                    # print(spiral3_path[-1,:])
                    u = TG.calc_velocity(v0,n1*2,v1,p[4])
                    # print(u)
                    traj = TG.calc_trajectory(u,p,r,s=p[4],path=spiral3_path,q0=cfg0)
                    # print(traj[:,6])
                    # print(traj[:,7]**2*traj[:,5])
                    cost=TG.eval_trajectory(traj, cost_map, vehicle=veh,road=road)
                    # ax1.plot(spiral3_path[:,1], spiral3_path[:,2],  linewidth=3., label='{0}'.format(cost))
                    # if not np.isinf(cost):
                    ax1.plot(spiral3_path[:,1], spiral3_path[:,2],  linewidth=3.)
                    ax1.text(spiral3_path[-1,1],spiral3_path[-1,2],'{0:.2f}'.format(cost))
    
    
    # close database connection
    cursor.close()
    conn.close()

    #
    # plt.legend()
    plt.axis('equal')
    plt.show()
