# 2015.11.16, LI Yunsheng

from math import floor, ceil, sqrt
import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

class Vehicle():
    def __init__(self, wheelbase=2.94, front_offset=0.988, rear_offset=1.28, width=2.029, trajectory=np.array([[-1.,-1.,49.9,49.9,np.pi/6.,0.,0.,0.,0.]])):
        # trajectory: N X 10 array - [ [t, s, x, y, theta, k, dk, v, a] ]
        # state: 1 X 6 vector - [t, x, y, theta, k, v]
        self.trajectory = trajectory
        self.state = np.array([trajectory[0,0], trajectory[0,2], trajectory[0,3], trajectory[0,4], trajectory[0,5], trajectory[0,7]])
        self.traj_fun = None if trajectory[0,0] < 0. else interp1d(self.trajectory[:,0],self.trajectory[:,1:].T,kind='linear')
        # self.traj_fun2 = None if trajectory[0,1] < 0. else interp1d(self.trajectory[:,1],np.array([self.trajectory[:,0],self.trajectory[:,2],self.trajectory[:,3],self.trajectory[:,4],self.trajectory[:,5],self.trajectory[:,6],self.trajectory[:,7],self.trajectory[:,8],self.trajectory[:,9]]).T)
        # geometric parameters
        self.wheelbase = wheelbase
        self.front_offset = front_offset
        self.rear_offset = rear_offset
        self.length = wheelbase + front_offset + rear_offset
        self.width = width
        
        #
        self.center_of_rear_axle = self.state[1:3]
        self.heading = self.state[3]
        c, s = np.cos(self.heading), np.sin(self.heading)
        self.geometric_center = self.center_of_rear_axle + (wheelbase+front_offset-rear_offset)/2*np.array([c,s])
        length = self.length
        self.vertex = self.geometric_center + 0.5*np.array([
            [-c*length-s*width, -s*length+c*width],
            [-c*length+s*width,-s*length-c*width],
            [c*(length-0.3*width)+s*width, s*(length-0.3*width)-c*width],
            [c*length+s*0.7*width,s*length-c*0.7*width],
            [c*length-s*0.7*width, s*length+c*0.7*width],
            [c*(length-0.3*width)-s*width, s*(length-0.3*width)+c*width]
            ])


    def update(self,time):
        if self.traj_fun is not None and time is not None and time <= self.trajectory[-1,0]:
            tmp = self.traj_fun(time) # [s, x, y, theta, k, dk, v, a, j]
            self.state = np.array([time, tmp[1], tmp[2], tmp[3], tmp[4], tmp[6]])
            self.center_of_rear_axle = self.state[1:3]
            self.heading = self.state[3]
            c, s = np.cos(self.heading), np.sin(self.heading)
            self.geometric_center = self.center_of_rear_axle + (wheelbase+front_offset-rear_offset)/2*np.array([c,s])
            length = self.length
            self.vertex = self.geometric_center + 0.5*np.array([
                [-c*length-s*width, -s*length+c*width],
                [-c*length+s*width,-s*length-c*width],
                [c*(length-0.3*width)+s*width, s*(length-0.3*width)-c*width],
                [c*length+s*0.7*width,s*length-c*0.7*width],
                [c*length-s*0.7*width, s*length+c*0.7*width],
                [c*(length-0.3*width)-s*width, s*(length-0.3*width)+c*width]
                ])
        # else:
        #     print("invalid time:", time)


    def recover(self):
        trajectory = self.trajectory
        length = self.length
        width = self.width
        wheelbase = self.wheelbase
        self.state = np.array([trajectory[0,0], trajectory[0,2], trajectory[0,3], trajectory[0,4], trajectory[0,5], trajectory[0,7]])
        self.center_of_rear_axle = self.state[1:3]
        self.heading = self.state[3]
        c, s = np.cos(self.heading), np.sin(self.heading)
        self.geometric_center = self.center_of_rear_axle + (wheelbase+front_offset-rear_offset)/2*np.array([c,s])
        self.vertex = self.geometric_center + 0.5*np.array([
            [-c*length-s*width, -s*length+c*width],
            [-c*length+s*width,-s*length-c*width],
            [c*(length-0.3*width)+s*width, s*(length-0.3*width)-c*width],
            [c*length+s*0.7*width,s*length-c*0.7*width],
            [c*length-s*0.7*width, s*length+c*0.7*width],
            [c*(length-0.3*width)-s*width, s*(length-0.3*width)+c*width]
            ])

    #
    def covering_disk_radius(self):
        return np.sqrt(self.length**2/9. + self.width**2)/2

    #
    def covering_disk_centers(self):
        distance = 2.*self.length/3.
        direction = np.array([np.cos(self.heading),np.sin(self.heading)])
        centers = np.zeros((3,2))
        centers[1] = self.geometric_center
        centers[0] = centers[1] - distance * direction
        centers[2] = centers[1] + distance * direction
        return centers

    # traj: array of points on trajectory - [(t,s,x,y,theta,k,dk,v,a)]
    def covering_centers(self, traj):
        points = np.zeros((traj.shape[0], 6))
        sin_t = np.sin(traj[:,4])
        cos_t = np.cos(traj[:,4])
        points[:,2] = traj[:,2] + cos_t*(self.wheelbase+self.front_offset-self.rear_offset)/2. # x2
        points[:,3] = traj[:,3] + sin_t*(self.wheelbase+self.front_offset-self.rear_offset)/2. # y2
        points[:,0] = points[:,2] + self.length*2/3*cos_t
        points[:,1] = points[:,3] + self.length*2/3*sin_t
        points[:,4] = points[:,2] - self.length*2/3*cos_t
        points[:,5] = points[:,3] - self.length*2/3*sin_t
        return points



class Road():
    def __init__(self, center_line, lane_width=3.5, ref_grid_width=0.5, ref_grid_length=1.):
        # center_line: Nx5 array, [[s, x, y, theta, k],...]
        # # current_lane: {1,0,-1} - {right, middle, left}
        # lane_number = 3
        self.length = center_line[-1,0]
        self.lane_width = lane_width
        self.width = lane_width * 3.
        self.center_line = center_line
        # self.current_lane = current_lane
        self.center_line_fun = interp1d(center_line[:,0], center_line[:,1:].T, kind='linear') # return [x,y,theta,k]
        self.grid_num_per_lane = 2 * ceil(ceil(lane_width / ref_grid_width) / 2) # lateral direction
        self.grid_num_lateral = self.grid_num_per_lane*3 # 横向网格数目
        self.grid_num_longitudinal = ceil(self.length/ref_grid_length) # 纵向网格数目
        self.grid_width = lane_width / self.grid_num_per_lane
        self.grid_length = self.length/self.grid_num_longitudinal
        #
        self.lateral_biases = np.linspace(-self.width/2., self.width/2., self.grid_num_lateral+1)
        self.longitudinal_biases = np.linspace(0., self.length, self.grid_num_longitudinal+1)
        # 计算纵横向网格线，方便绘图
        longitudinal_lines = np.zeros((self.center_line.shape[0], 2*(self.grid_num_lateral+1))) #[[x,y],...]
        for i in range(self.grid_num_lateral+1):
            longitudinal_lines[:,(2*i):(2*i+2)] = self.lateral_biasing_line(self.lateral_biases[i])
        self.longitudinal_lines = longitudinal_lines

        lateral_lines = np.zeros((ceil(self.width/0.1), 2*(self.grid_num_longitudinal+1))) #[[x,y],...]
        for i in range(self.grid_num_longitudinal+1):
            lateral_lines[:,(2*i):(2*i+2)] = self.longitudinal_biasing_line(self.longitudinal_biases[i])
        self.lateral_lines = lateral_lines
        #


    def lateral_biasing_line(self,lateral_bias):
        # lateral_bias \in (-self.width/2, self.width/2)
        # return Nx2 array (x,y): x=x0-l*sin(theta), y=y0+l*cos(theta)
        return self.center_line[:,1:3] + lateral_bias*np.array([-np.sin(self.center_line[:,3]), np.cos(self.center_line[:,3])]).T


    def longitudinal_biasing_line(self,longitudinal_bias):
        # longitudinal_bias \in (0,self.length)
        # return Mx2 array (x,y)
        lateral_bias = np.linspace(-self.width/2., self.width/2.,np.ceil(self.width/0.1))
        base_station = self.sl2xy(longitudinal_bias, 0) #(x,y,theta,k)
        return base_station[0:2] + np.array([-np.sin(base_station[2])*lateral_bias, np.cos(base_station[2])*lateral_bias]).T


    def sl2xy(self, s, l):
        # s - longitudinal bias, l - lateral bias
        # return [x,y,theta,k]
        if 0<=s<=self.length:
            station = self.center_line_fun(s)
            k = 0. if abs(station[3])<1.e-10 else (station[3]**-1-l)**-1
            return np.array([station[0]-l*np.sin(station[2]), station[1]+l*np.cos(station[2]), station[2], k])
        else:
            return None


    def ij2xy(self,i,j):
        # return [x,y,theta,k]
        if abs(j) <= self.grid_num_lateral//2 and 0<=i<=self.grid_num_longitudinal:
            return self.sl2xy(self.longitudinal_biases[i], self.lateral_biases[j+self.grid_num_lateral//2])
        else:
            return None


    def __xys(self,x,y,s):
        tmp = self.center_line_fun(s)
        return (x-tmp[0])*np.cos(tmp[2]) + (y-tmp[1])*np.sin(tmp[2])


    def xy2sl(self, x, y):
        try:
            f = lambda s: self.__xys(x,y,s)
            s0 = fsolve(f, self.length/2)
            tmp = self.center_line_fun(s0)
            if abs(tmp[2]) < 1.e-4:
                l0 = (y-tmp[1])/np.cos(tmp[2])
            else:
                l0 = (tmp[0]-x)/np.sin(tmp[2])
            # return l0
            return np.array([s0,l0])
        except ValueError:
            return np.array([np.inf, np.inf])

# gaussian filter 2D
def fspecial_gauss(size, sigma=1):

    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()



class Workspace():
    def __init__(self,base=np.zeros((500,500)), resolution=0.2, vehicle=Vehicle(), road=None, current_lane=0, target_lane=0, lane_costs=[0.3,0.6,0.9], static_obsts=None, moving_obsts=None):
        # current_lane: {0,1,2} - {right, middle, left}, default: 0
        self.base = base
        self.row = base.shape[0]
        self.column = base.shape[1]
        self.resolution = resolution
        self.static_obsts = static_obsts # list of static vehicles
        self.moving_obsts = moving_obsts # list of moving vehicles
        self.road = road
        self.current_lane = current_lane
        self.target_lane = target_lane
        self.lane_grids = self.grids_of_lanes(self.road)
        self.lane_costs = lane_costs
        self.lane_map = self.__lane_map()
        self.vehicle = vehicle
        #
        # self.disk = self.disk_filter()
        self.collision_filter = self.disk_filter(r=vehicle.covering_disk_radius())
        cost_filter = self.disk_filter(1.5)
        cost_filter *= fspecial_gauss(cost_filter.shape[0])
        self.cost_filter = cost_filter/cost_filter.sum() # consider using gaussianblur filter method
        #
        self.time = 0. if self.moving_obsts is not None else None
        self.delta_t = 0.05
        self.time_series = np.linspace(0.,10.,201) if self.moving_obsts is not None else None
        self.static_map = self.__static_map()
        self.env_map = self.__env_map()
        self.collision_map = self.__collision_map(flt=self.collision_filter)
        self.cost_map = self.__cost_map(flt=self.cost_filter)
        self.cost_maps = self.__cost_maps()


    def update(self, time):
        if self.moving_obsts is not None and time >= 0:
            self.time = time
            for i in range(len(self.moving_obsts)):
                self.moving_obsts[i].update(time)
            self.env_map = self.__env_map()
            self.collision_map = self.__collision_map(flt=self.collision_filter)
            self.cost_map = self.__cost_map(flt=self.cost_filter)


    def set_current_lane(self, current_lane):
        self.current_lane = current_lane


    def set_target_lane(self, target_lane):
        self.target_lane = target_lane


    def set_lane_costs(self, lane_costs):
        # lane_costs: [v_right, v_center, v_left]
        self.lane_costs = lane_costs
        self.lane_map = self.__lane_map()
        self.cost_map = self.__cost_map(flt=cost_filter)


    def disk_filter(self, r=None):
        # r <= 10m, O(10.1, 10.1)
        if r is None:
            r = self.vehicle.covering_disk_radius()
        R = np.zeros((101,101))
        k0 = ceil(r/self.resolution - 0.5)
        # N = 2 * k0 + 1 # 滤波器是NxN方阵，N为奇数
        # 中间一行
        for j in range(50-k0, 51+k0):
            R[50,j] = 1.
        #向上\下
        for i in range(0, k0):
            rr = sqrt( r**2 - ((i+0.5)*self.resolution)**2 )
            k = ceil(rr/self.resolution - 0.5)
            for j in range(50-k, 51+k):
                R[51+i, j] = 1.
                R[49-i, j] = 1.
        F = R[(50-k0):(51+k0), (50-k0):(51+k0)]
        N = F.sum()
        return F/N


    def vehicle_filter(self,theta=0.):
        veh = Vehicle(trajectory=np.array([[-1.,-1.,49.9,49.9,theta,0.,0.,0.,0.,0.]]))
        #R = np.zeros((81,81))
        r = sqrt(veh.width**2 + (veh.length + veh.wheelbase)**2) / 2
        k0 = ceil(r/self.resolution - 0.5)
        #
        # grids_list = []
        # for i in range(len(veh.vertex)):
        #     j = (i+1) % (len(veh.vertex))
        #     #
        #     j_m1 = floor(veh.vertex[i,0]/self.resolution)
        #     j_m2 = floor(veh.vertex[j,0]/self.resolution)
        #     if j_m1 == j_m2:
        #         grids = [(j_m1,floor(veh.vertex[i,1]/self.resolution)), (j_m2,floor(veh.vertex[j,1]/self.resolution))]
        #     else:
        #         f = interp1d([veh.vertex[i,0], veh.vertex[j,0]], [veh.vertex[i,1], veh.vertex[j,1]])
        #         if veh.vertex[i,0] < veh.vertex[j,0]:
        #             grids = self.grids_occupied_by_line(f,veh.vertex[i,0], veh.vertex[j,0])
        #         else:
        #             grids = self.grids_occupied_by_line(f,veh.vertex[j,0], veh.vertex[i,0])
        #     grids_list.append(grids)
        F = self.grids_occupied_by_polygon(veh.vertex)
        N = F.sum()
        return F[(249-k0):(250+k0),(249-k0):(250+k0)]/N


    def grids_occupied_by_line(self, f, x1, x2):
        # 0 <= x1 < x2 <= 100
        j_min, j_max = floor(x1/self.resolution), floor(x2/self.resolution) #列
        # f 单调 : 0 <= f(x1) , f(x2) <= 100
        i_min, i_max = floor(f(x1)/self.resolution), floor(f(x2)/self.resolution) #行
        #
        if x1 > x2:
            j_min, j_max = j_max, j_min
            i_min, i_max = i_max, i_min
        #
        grids = [(j_min, i_min), (j_max, i_max)]
        for j in range(j_min+1, j_max+1):
            i = floor(f(j*self.resolution)/self.resolution)
            grids.append((j-1,i))
            grids.append((j,i))
        return grids #[(列，行)]


    def grids_encircled_by_lines(self,grids_list):
        R = np.zeros((500,500))
        grids = []
        for i in range(len(grids_list)):
            grids.extend(grids_list[i])
        grids.sort()
        i = 0
        j = 1
        while i < len(grids):
            while grids[j][0] == grids[i][0]:
                j += 1
                if j == len(grids):
                    break
            for k in range(grids[i][1], grids[j-1][1]+1):
                if 0<=k<500 and 0<=grids[i][0]<500:
                    R[k,grids[i][0]] = 1.
            if j == len(grids):
                break
            else:
                i = j
                j += 1
        return R


    def grids_occupied_by_polygon(self,plg):
        # plg : List of vertex [(x1,y1),(x2,y2),...,(xn,yn)]
        grids_list=[]
        for i in range(len(plg)):
            j = (i+1) % (len(plg))
            j_m1 = floor(plg[i][0] / self.resolution)
            j_m2 = floor(plg[j][0] / self.resolution)
            if j_m1 == j_m2:
                grids = [(j_m1,floor(plg[i][1]/self.resolution)), (j_m2,floor(plg[j][1]/self.resolution))]
            else:
                f = interp1d([plg[i][0], plg[j][0]], [plg[i][1], plg[j][1]])
                # if plg[i][0] < plg[j][0]:
                grids = self.grids_occupied_by_line(f,plg[i][0], plg[j][0])
                # else:
                #     grids = self.grids_occupied_by_line(f,plg[j][0], plg[i][0])
            grids_list.append(grids)
        return self.grids_encircled_by_lines(grids_list)


# bug: vertical line
    def grids_of_lanes(self, road):
        # return: list of matrix map
        # if road is None:
        #     return None
        # else:
        ll = road.longitudinal_lines
        nl = road.grid_num_per_lane
        # print(nl)
        map_list = []
        f1 = interp1d(ll[:,0], ll[:,1])
        for i in range(3):
            grids_list = []
            f2 = interp1d(ll[:,2*(i+1)*nl], ll[:,1+2*(i+1)*nl])

            x0, x1 = floor(ll[0,2*i*nl]/self.resolution), floor(ll[0,2*(i+1)*nl]/self.resolution)
            if x0 != x1:
                g = interp1d([ll[0,2*i*nl], ll[0,2*(i+1)*nl]], [ll[0, 1+2*i*nl], ll[0, 1+2*(i+1)*nl]])
                grids1 = self.grids_occupied_by_line(g, ll[0,2*i*nl], ll[0,2*(i+1)*nl])
            else:
                grids1 = [(x0, floor(ll[0, 1+2*i*nl]/self.resolution)), (x1, floor(ll[0, 1+2*(i+1)*nl]/self.resolution))]

            x0, x1 = floor(ll[-1,2*i*nl]/self.resolution), floor(ll[-1,2*(i+1)*nl]/self.resolution)
            if x0!=x1:
                h = interp1d([ll[-1,2*i*nl], ll[-1,2*(i+1)*nl]], [ll[-1, 1+2*i*nl], ll[-1, 1+2*(i+1)*nl]])
                grids2 = self.grids_occupied_by_line(h, ll[-1,2*i*nl], ll[-1,2*(i+1)*nl])
            else:
                grids2 = [(x0, floor(ll[-1, 1+2*i*nl]/self.resolution)), (x1, floor(ll[-1, 1+2*(i+1)*nl]/self.resolution))]

            grids_list.append(self.grids_occupied_by_line(f1, ll[0,2*i*nl], ll[-1,2*i*nl]))
            grids_list.append(self.grids_occupied_by_line(f2, ll[0,2*(i+1)*nl], ll[-1,2*(i+1)*nl]))
            grids_list.append(grids1)
            grids_list.append(grids2)
            map_list.append(self.grids_encircled_by_lines(grids_list))
            f1 = f2
        # print(map_list)
        return map_list


    def __static_map(self):
        if not self.static_obsts:
            return self.base
        else:
            R = self.base
            for i in range(len(self.static_obsts)):
                R += self.grids_occupied_by_polygon(self.static_obsts[i].vertex)
            R = np.where(R > 0., 1., 0.)
            return R


    def __env_map(self):
        R = self.static_map
        if self.moving_obsts is not None:
            for i in range(len(self.moving_obsts)):
                R += self.grids_occupied_by_polygon(self.moving_obsts[i].vertex)
            R = np.where(R > 0., 1., 0.)
        return R


    def __collision_map(self, flt=None):
        if flt is None:
            flt = self.collision_filter
        eps = 1.e-6
        Env_map = self.env_map
        Dest = cv2.filter2D(Env_map, -1, flt)
        Dest = np.where(Dest > eps, 1., 0.)
        return Dest


    def __lane_map(self):
        if self.lane_costs is not None:
            v1 = min(self.lane_costs)
            p1 = self.lane_costs.index(v1)
            v3 = max(self.lane_costs)
            p3 = self.lane_costs.index(v3)
            p2 = [i for i in [0,1,2] if i!=p1 and i!=p3][0]
            v2 = self.lane_costs[p2]
            lm = v1*self.lane_grids[p1] + v2*self.lane_grids[p2]
            lm = np.where(lm>v2, (v1+v2)/2, lm)
            lm += v3*self.lane_grids[p3]
            lm = np.where(lm>v3, (v2+v3)/2, lm)
            return lm
        else:
            return None


    def __cost_map(self, flt=None, scale=255.):
        if flt is None:
            flt = self.cost_filter
        eps = 1.e-6
        cost_map = self.collision_map
        cost_map += cv2.filter2D(cost_map, -1, flt)
        cost_map = np.where(cost_map<eps, 0., cost_map)
        cost_map = scale * np.where(cost_map>1, 1., cost_map)
        for i in range(3):
            cost_map += self.lane_costs[i] * self.lane_grids[i]
        cost_map = np.where(cost_map>255., 255., cost_map)
        return cost_map


    def __cost_maps(self):
        if self.time_series is not None:
            cost_maps = np.zeros((201,500,500))
            for i in range(201):
                self.update(self.time_series[i])
                cost_maps[i]=self.cost_map
            return cost_maps


    # cost-map search
    # current - (x,y)
    # goal - (xg, yg)
    # map - 500 X 500 grayscale map
    @staticmethod
    def costmap_heuristic(current, goal, map):
        heuristic = 0.




