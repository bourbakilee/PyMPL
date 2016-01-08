from Environment import *
import TrajectoryGeneration as TG
from queue import PriorityQueue


class State:
    def __init__(self, time=np.inf, length=np.inf,road=None, r_i=None, r_j=None, r_s=None, r_l=None, \
        x=0., y=0., theta=0., k=0., v=0., acc=0., cost=np.inf, vehicle=None, heuristic_map = None):
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
                self.r_i = int(round(r_s/road.grid_length))
                self.r_j = int(round(r_l/road.grid_width))
                self.x, self.y, self.theta, self.k = road.sl2xy(self.r_s, self.r_l)
            else:
                self.x, self.y, self.theta, self.k = x, y, theta, k
                self.r_s, self.r_l = road.xy2sl(x,y)[0:2,0]
                self.r_i, self.r_j = int(round(self.r_s/road.grid_length)), int(round(self.r_l/road.grid_width))
        else:
            self.x, self.y, self.theta, self.k = x, y, theta, k
            self.r_s, self.r_l = -1., -1.
            self.r_i, self.r_j = -1, -1
        self.v = v
        self.q = np.array([self.x, self.y, self.theta, self.k])
        # self.dk = 0.
        self.a = acc # not actual acceleration
        # self.current_lane = 0 # int
        # self.target_lane = 0 # int
        # 
        # must be updated
        self.time = time
        self.length = length
        self.reach = False
        self.extend = False
        self.parent = None
        self.cost = cost
        if heuristic_map is not None:
            self.heuristic = query_heuristic(self, heuristic_map, vehicle)
        else:
            self.heuristic = 0.
        self.priority = self.cost + self.heuristic


    # def __cmp__(self, other):
    #     return cmp( self.priority , other.priority )
    def __lt__(self, other):
        return self.priority < other.priority


    # update if traj successfully connect from parent to self
    # cost must not be inf
    def update(self, parent, cost, traj):
        if not parent.extend:
            parent.extend = True
        if self.cost > cost + parent.cost:
            self.time = traj[-1,0]
            self.length = traj[-1,1]
            self.reach = True
            self.parent = parent
            self.cost = cost + parent.cost
            self.priority = self.heuristic + cost
            return True
        return False


    def update2(self, parent, cost, traj, road, heuristic_map):
        self.x, self.y, self.theta, self.k = traj[-1,2:6]
        self.r_s, self.r_l = road.xy2sl(self.x,self.y)[0:2,0]
        self.r_i, self.r_j = int(round(self.r_s/road.grid_length)), int(round(self.r_l/road.grid_width))
        self.v = traj[-1,7]
        self.a = traj[-1,8]
        self.q = np.array([self.x, self.y, self.theta, self.k])
        self.heuristic = query_heuristic(self, heuristic_map)
        # return self.update(parent, cost, traj)


    # @staticmethod
    def distance(self, g):
        return abs(self.x - g.x) + abs(self.y - g.y) + abs(self.theta - g.theta) + abs(self.k - g.k) + abs(self.v - g.v)


    # traj - [t,s,x,y,theta,k,dk,v,a]
    # ref cost - parent node's cost
    # state_dict - {(i,j,k):state(i,j,v)}
    @staticmethod
    def post_process(current, successor, goal, cost, traj, truncated, pq, state_dict, traj_dict, vehicle, road, costmap, heuristic_map,cursor, weights=np.array([5., 10., 0.05, 0.2, 0.2, 0.2, 10., 0.5, 10., -2.])):
        if not truncated:
            if successor.update(current, cost, traj):
                pq.put(successor)
                if successor != goal:
                    state_dict[(successor.r_i, successor.r_j, int(round(successor.v/2)))] = successor
                traj_dict[(current, successor)] = traj
        else:
            successor.update2(current, cost, traj, road, heuristic_map)
            i, j, k = successor.r_i, successor.r_j, int(round(successor.v/2))
            try:
                state = state_dict[(i,j,k)]
                if state.distance(successor) > 1.e-4:
                    traj = trajectory(current, state, cursor)
                    if traj is not None:
                        cost = TG.eval_trajectory(traj, costmap, vehicle=vehicle, road=road, truncate=False, weights=weights)
                        if not np.isinf(cost):
                            successor = state 
                        else:
                            successor = None
                    else:
                        successor = None
                else:
                    successor = state 
            except KeyError:
                state_dict[(i,j,k)] = successor
            finally:
                if successor is not None:
                    if successor.update(current, cost, traj):
                        pq.put(successor)
                        traj_dict[(current, successor)] = traj

        
        


    # {next_state}
    # state_dict
    def successors(self, state_dict, road, goal, vehicle, heuristic_map, accs=[-4., -2., 0., 2.], v_offset=[-1., -0.5, 0., 0.5, 1.], times=[1., 2., 4.], \
        p_lims=(0.2,0.15,20.+1.e-6,-1.e-6,2.1,-6.,6.)):
        # road is not None
        # goal state is not None
        # accs = [-2., -1., 0., 1.]
        # v_offset = [-0.5, 0., 0.5]
        # times = [1., 2., 4. , 7.]
        # p_lims - { k_m, dk_m, v_max, v_min, a_max, a_min, ac_m } = (0.2,0.1,20.,0.,2.,-6.,10.)
        outs = []
        # if (self.v + goal.v)/2*5 < goal.r_s - self.r_s:
        # if random.random()<0.1:
        outs.append(goal)
        for n1 in accs:
            for n2 in v_offset:
                for n3 in times:
                    v = self.v + n1*n3
                    l = self.r_l + n2*n3
                    s = self.r_s + (self.v+v)/2*n3
                    # x = self.x + s*np.cos(self.theta) - l*np.sin(self.theta)
                    # y = self.y + s*np.sin(self.theta) + l*np.cos(self.theta)
                    # s, l = road.xy2sl(x,y)
                    i = int(round(s/road.grid_length))
                    j = int(round(l/road.grid_width))
                    k = int(round(v/2))
                    try:
                        state = state_dict[(i,j,k)]
                    except KeyError:
                        if self.r_i < i < goal.r_i and abs(j) < road.grid_num_lateral//2 and p_lims[3] < v < p_lims[2]:
                            state = State(road=road, r_i=i, r_j=j, v=v, acc=n1, heuristic_map=heuristic_map, vehicle=vehicle)
                        else:
                            state = None
                    finally:
                        if state is not None:
                            outs.append(state)
        return outs



def trajectory(start, goal, cursor):
    p, r = TG.calc_path(cursor, start.q, goal.q)
    if r is not None and p[4]>0:
        u = TG.calc_velocity(start.v, start.a, goal.v, p[4])
        if u[3] is not None and u[3]>0:
            path = TG.spiral3_calc(p,r,q=start.q,ref_delta_s=0.2)
            traj = TG.calc_trajectory(u,p,r,s=p[4],path=path,q0=start.q, ref_time=start.time, ref_length=start.length)
            return traj
    return None



def Astar(start, goal, road, cost_map, vehicle, heuristic_map, cursor, weights=np.array([5., 10., 0.05, 0.2, 0.2, 0.2, 10., 0.5, 10., -2.])):
    # pq - priority queue of states waiting to be extended, multiprocessing
    # node_dict - {(i,j,k):state}
    # edge_dict - store trajectory, {(state1, state2):trajectory}
    # pq, node_dict, edge_dict are better defined outside, for multiprocessing
    # count = 0
    pq = PriorityQueue()
    pq.put(start)
    node_dict = {(start.r_i, start.r_j, int(round(start.v/2))):start, (goal.r_i, goal.r_j, int(round(goal.v/2))):goal}
    edge_dict = {}
    while not goal.extend and not pq.empty():
        current = pq.get()
        successors = current.successors(state_dict=node_dict, road=road, goal=goal, vehicle=vehicle, heuristic_map=heuristic_map)
        current.extend = True
        for successor in successors:
            # count += 1
            traj = trajectory(current, successor, cursor)
            if traj is not None:
                if successor == goal:
                    cost = TG.eval_trajectory(traj, cost_map, vehicle=vehicle, road=road, truncate=False, weights=weights)
                    truncated = False
                else:
                    cost, traj, truncated = TG.eval_trajectory(traj, cost_map, vehicle=vehicle, road=road, weights=weights)
                if not np.isinf(cost) and traj is not None:
                    State.post_process(current, successor, goal, cost, traj, truncated, pq, node_dict, edge_dict, vehicle, road, cost_map, heuristic_map, cursor, weights=weights)
    if goal.extend:
        return True, node_dict, edge_dict
    else:
        return False, node_dict, edge_dict
    # return count + len(node_dict) + len(edge_dict)