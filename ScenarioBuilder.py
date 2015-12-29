from Environment import *
import TrajectoryGeneration as TG
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
    # road center line points
    p = (0.,0.,0.,0.,90.) # (p0~p3, sg)
    center_line = TG.spiral3_calc(p, q=(5.,50.,0.))
    print(center_line)
    # road
    road = Road(center_line)
    # vehicle
    veh = Vehicle()
    # workspace
    ws = Workspace(vehicle=veh, road=road)
    road_lane_bitmap0 = ws.lane_maps[0]
    road_lane_bitmap1 = ws.lane_maps[1]
    road_lane_bitmap2 = ws.lane_maps[2]
    # write the lane bitmaps into files
    np.savetxt('road_lane_bitmap0.txt', d_lane_bitmap0, fmt='%i',delimiter=' ')
    np.savetxt('road_lane_bitmap1.txt', d_lane_bitmap1, fmt='%i',delimiter=' ')
    np.savetxt('road_lane_bitmap2.txt', d_lane_bitmap2, fmt='%i',delimiter=' ')
    # road bitmap
    road_bitmap = road_lane_bitmap0 + road_lane_bitmap1 + road_lane_bitmap2
    road_bitmap = np.where(road_bitmap>1.e-6, 1., 0.)
    np.savetxt('road_bitmap.txt', road_bitmap, fmt='%i', delimiter=' ')