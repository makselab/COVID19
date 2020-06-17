'''
    Some general functions shared by some of the main notebooks.
'''
import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import geohash
from matplotlib import cm
import geopandas as geopd
import datetime as dt
import graph_tool.all as gt
import geopy.distance as gdist
from geopy.distance import geodesic
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from shapely.geometry import Point
from collections import defaultdict, Counter

def set_datelist(init, final, days=1, hours=0):
    '''
        'arange'-like function for datetime objects.
        'init' and 'final' should be date or datetime
        objects.
    '''
    lst = [init]
    while lst[-1]<final:
        lst.append(lst[-1]+timedelta(days=days, hours=hours))
    return lst

def read_pickle(pickle_path):
    '''
        'pickle_path' should includes the '.pickle' extension.
    '''
    with open(pickle_path, 'rb') as handle:
        idict = pickle.load(handle)
    return idict

def unpack_users_timeline(path, fname):
    '''
        'path' to the folder of the serielized objects containing
        the daily trajectories for all users.
        'fname' should be the specific file to open.
    '''
    with open(os.path.join(path, fname), 'rb') as handle:
        info = pickle.load(handle)
    return info

def get_trajectory(uid, date_init, date_final, path_to_info='../Data/gps/users/'):
    '''
        Given an ID and the range of days, get the trajectory
        of this ID over time.
    '''
    #path_to_info = os.path.join('..', 'Data', 'gps', 'users')
    dir_files = sorted(os.listdir(path_to_info))
    dir_files = dir_files[:]
    
    all_lat, all_lon, all_time = [], [], []
    for fname in tqdm(dir_files):
        name_str = fname.split('_')
        day_date = dt.date(2020, int(name_str[2]), int(name_str[3].split('.')[0]))
        # get data only inside the time window
        if day_date<date_init or day_date>date_final: continue
        info = unpack_users_timeline(path_to_info, fname)
        if len(info[uid])>0: 
            lat_user1, lon_user1, time_user1 = info[uid][0], info[uid][1], info[uid][2]
            all_lat += lat_user1
            all_lon += lon_user1
            all_time += time_user1
    return [all_lat, all_lon, all_time]

def close_to_hospital(user_points, hospital_lat, hospital_lon, d=20.0):
    '''
        Given a list of lat and lon coordinates of hospitals, return which
        hospitals the ID was close at some period. 'user_points' is the list
        of lat and the list of lon of this user.
    '''
    user_lat, user_lon = np.array(user_points[0]), np.array(user_points[1])
    user_time = user_points[2]
    
    hospital_indexes = []
    for ind, xy in enumerate(zip(hospital_lat, hospital_lon)):
        lat_min, lat_max, lon_min, lon_max = define_bounding(xy[0],xy[1], dist=(d+1.0))
        cond1 = np.logical_and(user_lat>=lat_min, user_lat<=lat_max)
        cond2 = np.logical_and(user_lon>=lon_min, user_lon<=lon_max)
        true_lat_arr = user_lat[np.logical_and(cond1, cond2)]
        #true_lon_arr = user_lon[np.logical_and(cond1, cond2)]
        if true_lat_arr.shape[0]>0:
            hospital_indexes.append(ind)
    return hospital_indexes

def define_bounding(src_lat, src_lon, dist=60.0):
    '''
        Given a (src_lat, src_lon) coordinate point, define the
        bounding box coordinates according to distance 'dist' provided.
        
        return: lat_min, lat_max, lon_min, lon_max
    '''
    earth_radius = gdist.EARTH_RADIUS*1000 # meters.
    center_rad = (np.radians(src_lat), np.radians(src_lon))
    
    r = dist/earth_radius
    lat_min = np.degrees(center_rad[0]-r)
    lat_max = np.degrees(center_rad[0]+r)
    
    dlon = np.arcsin(np.sin(r)/np.cos(np.radians(src_lat)))
    lon_min = np.degrees(center_rad[1]-dlon)
    lon_max = np.degrees(center_rad[1]+dlon)
    return lat_min, lat_max, lon_min, lon_max

def get_colorlist(ncolors, cmap_name='gist_rainbow'):
    cmap = cm.get_cmap(cmap_name, ncolors)
    values = np.linspace(0, 1, ncolors)
            
    cmap_list = [ cm.colors.to_hex([cmap(v)[0],cmap(v)[1],cmap(v)[2]]) for v in values ]
    return cmap_list
################################################################

###############################################################
# --------------- RMS Mobility Calculation ------------------ #
###############################################################

# -> Document.
def get_user_rms(user_lat, user_lon, user_time):
    num_points = len(user_time)
    if num_points<10:
        return (user_time[0], user_time[-1], np.nan)
    d = 0
    for k in range(0, num_points-1):
        dx = gdist.geodesic((user_lat[k], user_lon[k]), (user_lat[k+1], user_lon[k+1])).m
        d += dx*dx
    rms = np.sqrt((d/num_points))
    return (user_time[0], user_time[-1], rms)

def get_daily_rms_all_users(fname, mobility_folder, timeline_folder):
    users_table = []
    info = unpack_users_timeline(timeline_folder, fname)
    for uid in tqdm(info.keys()):
        if len(info[uid])>0:
            user_lat, user_lon, user_time = info[uid][0], info[uid][1], info[uid][2]
            rms_info = get_user_rms(user_lat, user_lon, user_time)
        else:
            rms_info = (np.nan, np.nan, np.nan)
        users_table.append({'mobileId': uid, 'start_time': rms_info[0], 'end_time': rms_info[1], 'rms': rms_info[2]})
    daily_table = pd.DataFrame.from_dict(users_table)
    table_name = fname.split('.')[0]
    daily_table.to_csv(os.path.join(mobility_folder, '{}_mob.csv'.format(table_name)))
    
def get_rms_daylist(fname_list, mobility_folder, timeline_folder):
    for fname in tqdm(fname_list):
        get_daily_rms_all_users(fname, mobility_folder, timeline_folder)

############################################################
def adjlist_fmt_CI(g, output_file=None):
    adj_list = []
    for v in g.get_vertices():
        neighbors = g.get_all_neighbors(v)
        if neighbors.shape[0]>0:
            line = [g.vertex_index[v]] + list(neighbors)
            adj_list.append(line)
        else:
            adj_list.append([g.vertex_index[v]])
    if output_file!=None:
        file = open(output_file, 'w')
        for line in adj_list:
            str_list = list(map(lambda x: str(x), line))
            line_to_write = ' '.join(str_list)
            file.write(line_to_write+'\n')
        file.close()
    else:
        return adj_list
    