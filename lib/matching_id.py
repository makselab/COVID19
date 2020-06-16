import os
import pytz
import pickle
import geohash
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as geopd
import geopy.distance as gdist
from geopy.distance import geodesic
import datetime as dt
from datetime import datetime
from datetime import timedelta
from shapely.geometry import Point
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

import lib.general_utils as utils
Forta = pytz.timezone("America/Fortaleza")

def generate_cross_table(infected_places_dict, valid_days, hours_per_day=7):
    table = []
    key_list = sorted(list(infected_places_dict.keys()))
    
    for inf_index in key_list:
        user_sum = defaultdict(lambda:0)
        for uid, time_interval in infected_places_dict[inf_index]: user_sum[uid]+=time_interval
            
        uid_list = sorted(list(user_sum.keys()))
        time_values = [ user_sum[uid] for uid in uid_list ]
        days_users = [ valid_days[uid] for uid in uid_list ]
        argmax = np.argmax(time_values)
        total_sec = hours_per_day*days_users[argmax]*(60*60)
        
        table.append({'ORDEM': inf_index, 'mobileId': uid_list[argmax], 'p': time_values[argmax]/total_sec})
            
    return pd.DataFrame.from_dict(table)
    

'''
    Here we need to handle GPS files, cross them with infected household
    address, obtain the trajectories for specific IDs, etc.
'''

class GPS_MATCHING:
    def __init__(self, path_to_folder, path_to_trajectories, prefix=''):
        '''
            Give the path to the folder of the CSV files containing 
            the daily gps datapoints. The filename format should be
            $prefix_MM_DD.csv where MM and DD refers, respectively,
            to the month and the day of the datapoints.
            
            'path_to_trajectories' is the path to the folder where the
            users' serialized objects are gonna be stored. These objects
            are pickle files storing dictionaries where each user ID is
            a key holding the sorted timeline (lat,lon) points.
        '''
        self.prefix = prefix
        self.data_folder = path_to_folder
        self.pickle_folder = path_to_trajectories
        
    def create_trajectories_file(self, init_date, final_date, year=2020):
        '''
            'init_date' and 'final_date' are lists of length two.
            Their format must be ['MM', 'DD'] referring to the string
            indexes of the month and the day of the initial and final
            gps files.  
            
            From the 'init_date' to the 'final_date' the CSV files are
            converted to pickle files that will hold the users sorted
            timeline datapoints.
            
            obs: pay attention at the format of the CSV files. Here, we
            consider 5 columns: index, UID, timestamp, lat and lon.
        '''
        
        first_day = dt.date(year, int(init_date[0]), int(init_date[1]))
        last_day = dt.date(year, int(final_date[0]), int(final_date[1]))
        # List of date from first day to the last day.
        date_list = utils.set_datelist(first_day, last_day)
        
        for current_date in tqdm(date_list):
            day = current_date.day
            month = current_date.month
            day_str = str(day).zfill(2)
            month_str = str(month).zfill(2)
            # dict for all users for the current date.
            user_dict = defaultdict(list)
            lat_user = defaultdict(list)
            lon_user = defaultdict(list)
            time_user = defaultdict(list)
            n = 0
            with open(os.path.join(self.data_folder, f'{self.prefix}_{month_str}_{day_str}.csv'), 'r') as f:
                # get data for each user.
                for line in f:
                    if n==0: # the header of the file.
                        n+=1
                        continue
                    info = line.split(',')
                    time_user[info[1]].append(int(info[2]))
                    lat_user[info[1]].append(float(info[3]))
                    lon_user[info[1]].append(float(info[4][:-1]))
                    
                # sort the datapoints of each user according the timestamp.
                for uid in time_user.keys():
                    lat = lat_user[uid]
                    lon = lon_user[uid]
                    timing = time_user[uid]
                
                    timesorted_lat = [x for _,x in sorted(zip(timing, lat))]
                    timesorted_lon = [x for _,x in sorted(zip(timing, lon))]
                    timesorted = sorted(timing)
                    user_dict[uid] = [timesorted_lat, timesorted_lon, timesorted]
                    
                # Then, save the day dict in a serielized pickle object.
                with open(os.path.join(self.pickle_folder, f'{self.prefix}_users_{month_str}_{day_str}.pickle'), 'wb') as f:
                    pickle.dump(user_dict, f)
                    
    def patient_matching(self, order_index, lat_inf, lon_inf, inf_places,
                         R=30.0, init_interval=22, final_interval=5, num_files=-1, 
                         timezone_obj=pytz.timezone("America/Fortaleza")):
        '''
            Given the latitude and longitude of the household addresses of the
            infected people, match the users' IDs from the database with these
            places.
        '''
        info_files = sorted(os.listdir(self.pickle_folder))
        if num_files>0:
            info_files = info_files[:num_files]
        
        # Define the dictionary to store the time intervals.
        infected_places = defaultdict(list)
        valid_days = defaultdict(lambda:0)
        
        # The real date does not matter for this case, only
        # the time of the day.
        interval_begin = datetime(2020, 1, 1, init_interval, 0, 0)
        interval_final = datetime(2020, 1, 2, final_interval, 0, 0)
        
        for fname in tqdm(info_files):
            info = utils.unpack_users_timeline(self.pickle_folder, fname)
            unique_users = sorted(list(info.keys()))
            
            for index, current_uid in enumerate(unique_users):
                user_dict_for_candidates = defaultdict(list)
                valid_days[current_uid] += 1
                user_info = info[current_uid]
                person_lat = np.array(user_info[0])
                person_lon = np.array(user_info[1])
                person_timeline = user_info[2]
                person_hours = np.array([ datetime.fromtimestamp(x, timezone_obj) for x in person_timeline ])
                
                bool_list = []
                candidates = []
                # Defines a bool variable for each datapoint. True values refers to the
                # datapoints that are between the desired time range AND that are close
                # to at least one infected place at the bounding retangle.
                for time_index, date_hour in enumerate(person_hours):
                    if date_hour.time()>=interval_begin.time() or date_hour.time()<=interval_final.time():
                        # Check which infected places are close to the current point.
                        # lat_inf in [current_lat-dlat, current_lat+dlat] and
                        # lon_inf in [current_lon-dlon, current_lon+dlon].
                        cur_lat = person_lat[time_index]
                        cur_lon = person_lon[time_index]
                        lat_min, lat_max, lon_min, lon_max = utils.define_bounding(cur_lat, cur_lon, dist=R+2.0)
                        
                        condition1 = np.logical_and(lat_inf>=lat_min, lat_inf<=lat_max)
                        condition2 = np.logical_and(lon_inf>=lon_min, lon_inf<=lon_max)
                        all_candidates = order_index[np.logical_and(condition1, condition2)]
                        # If there is close infected addresses.
                        if all_candidates.shape[0]>0:
                            # We store each unique candidate for the infected place.
                            for order in all_candidates: 
                                user_dict_for_candidates[order]
                            bool_list.append(True)
                            candidates.append(all_candidates)
                        else:
                            bool_list.append(False)
                            candidates.append([])
                    else:
                        bool_list.append(False)
                        candidates.append([])
                
                # Through each datapoint
                for time_index, bool_v in enumerate(bool_list):
                    # If there is some close candidate at this point.
                    if bool_v:
                        for cur_place in candidates[time_index]:
                            place_lat = inf_places['LAT'].loc[cur_place]
                            place_lon = inf_places['LON'].loc[cur_place]
                            d = gdist.geodesic((person_lat[time_index], person_lon[time_index]), (place_lat, place_lon)).m
                            if d<=R:
                                user_dict_for_candidates[cur_place].append(person_timeline[time_index])
                            else:
                                user_dict_for_candidates[cur_place].append(-1)
                    else:
                        for order_ind in sorted(list(user_dict_for_candidates.keys())):
                            user_dict_for_candidates[order_ind].append(-1)
            
                # For each nearby infected place, we check how much time the current user
                # spent at these places.
                candidates_for_user = sorted(list(user_dict_for_candidates.keys()))
                for order_ind in candidates_for_user:
                    cur_interval = []
                    points_arr = user_dict_for_candidates[order_ind]
                    for time_s in points_arr:
                        if time_s>-1:
                            cur_interval.append(time_s)
                        else:
                            if len(cur_interval)>1 and (cur_interval[-1]-cur_interval[0])>0:
                                interval_size = cur_interval[-1]-cur_interval[0]
                                infected_places[order_ind].append((current_uid, interval_size))
                            cur_interval = []
        
        interval_diff = ((interval_final-interval_begin).seconds)/(60*60)
        table = generate_cross_table(infected_places, valid_days, hours_per_day=interval_diff)
        return table, infected_places
        
        