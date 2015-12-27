from Environment import *
import TrajectoryGeneration as TG
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches
import sqlite3

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
    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, q=(5.,30.,np.pi/9))
    road = Road(center_line)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(center_line[:,1], center_line[:,2], color='red', linestyle='--', linewidth=2.)

    for i in range(road.grid_num_lateral+1):
        if (i % road.grid_num_per_lane) == 0:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='green', linewidth=1.)
        else:
            ax1.plot(road.longitudinal_lines[:,2*i], road.longitudinal_lines[:,2*i+1], color='black', linewidth=0.3)
    for i in range(road.grid_num_longitudinal+1):
        ax1.plot(road.lateral_lines[:,2*i], road.lateral_lines[:,2*i+1],color='black', linewidth=0.3)
    #
    plt.axis('equal')
    # plt.axis('off')
    plt.show()
    # sl = np.array([[10,1],[50,2],[100,-3]])
    # print(sl)
    # xy = np.zeros(sl.shape)
    # for i in range(len(xy)):
    #     t = road.sl2xy(sl[i,0], sl[i,1])
    #     xy[i] = np.array([t[0], t[1]])
    # print(xy)
    # sl2 = np.zeros(sl.shape)
    # for i in range(len(sl2)):
    #     t = road.xy2sl(xy[i,0],xy[i,1])
    #     sl2[i] = np.array([t[0], t[1]])
    # print(sl2)
    # print(sl-sl2)


def test_workspace():
    # veh = Vehicle(trajectory=np.array([[-1.,-1.,2.,30.,0.,0.,0.,0.,0.,0.]]))
    p = (0.01, 0.0070893847415232263, 0.0056488099243383414, -0.01, 109.61234595301809)
    center_line = TG.spiral3_calc(p, s=100., q=(5.,30.,np.pi/9))
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

    # lane_map = 0.2*ws.lane_grids[0] + 0.4*ws.lane_grids[1]
    # lane_map = np.where(lane_map > 0.4, 0.4, lane_map)
    # lane_map += 0.6*ws.lane_grids[2]
    # lane_map = np.where(lane_map > 0.6, 0.6, lane_map)


    ax1.imshow(ws.cost_filter, cmap=plt.cm.Greens, origin="lower",extent=(0.,ws.resolution*ws.cost_filter.shape[1],0.,ws.resolution*ws.cost_filter.shape[0]))
    np.savetxt('cost_filter.txt',ws.cost_filter,fmt='%i',delimiter=' ')
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


if __name__ == '__main__':
    # test_vehicle()
    # test_road()
    test_workspace()
    # test()
    # test_traj()
