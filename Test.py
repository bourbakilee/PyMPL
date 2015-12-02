from Environment import *
import TrajectoryGeneration as TG
import matplotlib.pyplot as plt 
from matplotlib.path import Path
import matplotlib.patches as patches

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
    center_line = TG.spiral_calc(road_profile, 110., q=(0.,0.,0.))
    print(center_line[-1,:])


def test_road():
    center_line = TG.spiral_calc(road_profile, 100., q=(5.,30.,np.pi/9))
    road = Road(center_line)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(center_line[:,2], center_line[:,3], color='red', linestyle='--', linewidth=2.)

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
    center_line = TG.spiral_calc(road_profile, 100., q=(5.,30.,np.pi/9))
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
    test_veh = ws.vehicle_filter(theta = np.pi/4)
    #
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # lane_map = 0.2*ws.lane_grids[0] + 0.4*ws.lane_grids[1]
    # lane_map = np.where(lane_map > 0.4, 0.4, lane_map)
    # lane_map += 0.6*ws.lane_grids[2]
    # lane_map = np.where(lane_map > 0.6, 0.6, lane_map)


    ax1.imshow(ws.lane_map,cmap=plt.cm.Greens, origin="lower",extent=(0.,100.125,0.,100.125))
    plt.show()



if __name__ == '__main__':
    # test_vehicle()
    # test_road()
    # test_workspace()
    # test()