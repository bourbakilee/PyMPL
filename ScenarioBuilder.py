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


class State:
    def __init__(self, time=0., length=0.,road=None, r_i=None, r_j=None, r_s=None, r_l=None, x=0., y=0., theta=0., k=0., v=0.):
        if road is not None:
            if r_i is not None and r_j is not None:
                self.r_i = r_i # int
                self.r_j = r_j # int
                self.r_s = road.longitudinal_biases[r_i]
                self.r_l = road.lateral_biases[r_j + road.grid_num_lateral//2]
                self.x, self.y, self.theta, self.k = road.sl2xy(self.r_s, self.r_l)
            elif r_s is not None and r_l is not None:
                self.r_s = r_s
                self.r_l = r_l
                self.r_i = int(r_s/road.grid_length)
                self.r_j = int(r_l/road.grid_width)
                self.x, self.y, self.theta, self.k = road.sl2xy(self.r_s, self.r_l)
            else:
                self.x, self.y, self.theta, self.k = x, y, theta, k
                self.r_s, self.r_l = road.xy2sl(x,y)
                self.r_i, self.r_j = int(self.r_s/road.grid_length), int(self.r_l/road.grid_width)
        else:
            self.x, self.y, self.theta, self.k = x, y, theta, k
            self.r_s, self.r_l = -1., -1.
            self.r_i, self.r_j = -1, -1
        self.v = v
        self.q = np.array([self.x, self.y, self.theta, self.k])
        # self.dk = 0.
        # self.a = 0.
        # self.current_lane = 0 # int
        # self.target_lane = 0 # int
        # 
        # must be updated
        self.time = time
        self.length = length
        self.extended = False
        self.cost = 0.
        self.heuristic = 0.
        self.priority = 0.


    def __cmp__(self, other):
        return cmp( self.priority , other.priority )


    # traj - [t,s,x,y,theta,k,dk,v,a]
    # ref cost - parent node's cost
    def update(self, cost, traj, parent, road=None):
        self.x, self.y, self.theta, self.k = traj[-1,2:6]
        if road is not None:
            self.r_s, self.r_l = road.xy2sl(self.x,self.y)
            self.r_i, self.r_j = int(self.r_s/road.grid_length), int(self.r_l/road.grid_width)
        self.v = traj[-1,7]
        self.q = np.array([self.x, self.y, self.theta, self.k])
        self.time = traj[-1,0]
        self.length = traj[-1,1]
        self.extended = True
        self.cost = cost + parent.cost
        # self.heuristic -> update
        self.priority = self.heuristic + cost


    # {(i, j, v)}
    def out_set(self, road, accs=[-4., -2., 0., 2.], v_offset=[-1., -0.5, 0., 0.5, 1.], times=[1., 2., 4.]):
        # road is not None
        # accs = [-2., -1., 0., 1.]
        # v_offset = [-0.5, 0., 0.5]
        # times = [1., 2., 4. , 7.]
        outs = []
        for n1 in accs:
            for n2 in v_offset:
                for n3 in times:
                    # v = min(max(self.v + n1*n3, 0.), 20.)
                    # l = min(max(self.r_l + n2*n3, -road.width/2), road.width/2)
                    # s = min(max(self.r_s + (self.v+v)/2*n3, 0.), road.length)
                    v = self.v + n1*n3
                    l = self.r_l + n2*n3
                    s = self.r_s + (self.v+v)/2*n3
                    # x = self.x + s*np.cos(self.theta) - l*np.sin(self.theta)
                    # y = self.y + s*np.sin(self.theta) + l*np.cos(self.theta)
                    # s, l = road.xy2sl(x,y)
                    if -1.e-6<v<20+1.e-6 and -1.e-6<s<road.length+1.e-6 and -road.width/2<l<road.width/2:
                        i = int(s/road.grid_length)
                        j = int(l/road.grid_width)
                        if i > self.r_i:
                            outs.append((i,j,v,n1))
        return list(set(outs))



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

    # action space
    # print('q0:{0}'.format(cfg0))
    # v0 = 5.
    # # a0 = 0.5
    # l0 = 0.
    # s0 = 5.
    # valid = 0
    # for n1 in [-2.,-1.,0.,1.]: # v1 = v0 + n1* dv/dt
    #     for n2 in [-2.,-1.,0.,1.,2.]: # l1 = l0 + n2* 0.5
    #         for n3 in [1.,2.,4.,7.]:
    #             v1 = min(max(v0 + n1*2*n3, 0.), 20.)
    #             l1 = min(max(l0 + n2/2*n3, -road.width/2), road.width/2)
    #             s1 = min(max(s0 + (v0+v1)/2*n3, 0.), road.length)
    #             r_i = int(s1/road.grid_length)
    #             r_j = int(l1/road.grid_width)
    #             q1 = road.ij2xy(r_i,r_j)
    #             p, r = TG.calc_path(cursor, cfg0, q1)
    #             if r is not None:
    #                 # print(p,r)
    #                 spiral3_path = TG.spiral3_calc(p,r, q=cfg0, ref_delta_s=0.2)
    #                 # print(spiral3_path[-1,:])
    #                 u = TG.calc_velocity(v0,n1*2,v1,p[4])
    #                 # print(u)
    #                 if u[3] is not None and u[3]>0:
    #                     valid+=1
    #                     traj = TG.calc_trajectory(u,p,r,s=p[4],path=spiral3_path,q0=cfg0)
    #                     # print(traj[:,6])
    #                     # print(traj[:,7]**2*traj[:,5])
    #                     cost, traj=TG.eval_trajectory(traj, cost_map, vehicle=veh,road=road)
    #                     # ax1.plot(spiral3_path[:,1], spiral3_path[:,2],  linewidth=3., label='{0}'.format(cost))
    #                     if not np.isinf(cost) and traj is not None:
    #                         ax1.plot(traj[:,2], traj[:,3],  linewidth=3.)
    #                         ax1.text(traj[-1,2], traj[-1,3],'{0:.2f}'.format(cost))
    # print(valid)

    count = 0
    start_state = State(road=road, r_s=5., r_l=0., v=5.)
    ax1.plot(start_state.x, start_state.y, 'rs')
    goal_state = State(road=road, r_s=80., r_l=0., v=5.)
    ax1.plot(goal_state.x, goal_state.y, 'rs')
    state_list = [start_state]
    while len(state_list)>0:
        # print(len(state_list))
        current = state_list.pop(0)
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
                            state_list.append(next_state)
                            # plot
                            ax1.plot(traj[:,2], traj[:,3],  linewidth=1.)
                            # ax1.text(traj[-1,2], traj[-1,3],'{0:.2f}'.format(cost))
        next_state = goal_state
        p, r = TG.calc_path(cursor, current.q, next_state.q)
        if r is not None:
            if p[4]>0:
                u = TG.calc_velocity(current.v, a, v, p[4])
                if u[3] is not None and u[3]>0:
                    path = TG.spiral3_calc(p,r,q=current.q,ref_delta_s=0.2)
                    traj = TG.calc_trajectory(u,p,r,s=p[4],path=path,q0=current.q, ref_time=current.time, ref_length=current.length)
                    cost = TG.eval_trajectory(traj, cost_map, vehicle=veh, road=road, truncate=False)
                    if not np.isinf(cost) and traj is not None:
                        count += 1
                        next_state.update(cost, traj, current, road)
                        state_list.append(next_state)
                        # plot
                        ax1.plot(traj[:,2], traj[:,3],  linewidth=1.)
                        # ax1.text(traj[-1,2], traj[-1,3],'{0:.2f}'.format(cost))
        # if count > 2000:
        if goal_state.extended:
            break
    print(count)

    
    # close database connection
    cursor.close()
    conn.close()

    #
    # plt.legend()
    plt.axis('equal')
    # plt.savefig('scenario_1/planning_result.png', dpi=600)
    plt.show()
