import numpy as np
import pandas as pd
import glob
import geopy.distance as gdist
#from geopy.distance import geodesic
import pickle
import  pytz


from igraph import *
from datetime import timedelta,datetime,time

from tqdm import tqdm
from collections import defaultdict, Counter
from joblib import Parallel, delayed


#Local_time = pytz.timezone("America/Mexico_City")
Local_time = pytz.timezone("America/Fortaleza")



### Load daily Grandata ###
def load_day(i):
    dtypes = {'id': 'str', 'lat': 'float', 'long': 'float', 'datastamp': 'int64'}
    parse_dates = ['datastamp']
    data=pd.read_csv(i, dtype=dtypes, parse_dates=parse_dates,infer_datetime_format=True)
    ##Convert to local time
    date=[datetime.fromtimestamp(int(float(i)),Local_time).replace(tzinfo=None) for i in data.datastamp]
    data['datastamp']=date
    return data


### Write pickle files in order to compute te msrd ###
def write_pickle(_input,_output):
    month=_input.split('/')[-1].split('_')[0]
    day=_input.split('/')[-1].split('_')[-1].split('.')[0]
    user_dict = defaultdict(list)
    with open(_input, 'r') as f:
                n = 0
                lat_user,lon_user,time_user = defaultdict(list),defaultdict(list),defaultdict(list)
                for line in f:
                    if n==0:
                        n+=1
                        continue
                    info = line.split(',')
                    time_user[info[0]].append(int(float(info[3][:-1])))
                    lat_user[info[0]].append(float(info[1]))
                    lon_user[info[0]].append(float(info[2]))
                for uid in time_user.keys():
                    lat = lat_user[uid]
                    lon = lon_user[uid]
                    timing = time_user[uid]

                    new_lat = [x for _,x in sorted(zip(timing, lat))]
                    new_lon = [x for _,x in sorted(zip(timing, lon))]
                    new_timing = sorted(timing)

                    # The info stored for each key(ID).
                    user_dict[uid] = [new_lat, new_lon, new_timing]
                ###### SIGNAL HERE: PUT YOUR FOLDER.
                with open(_output+month+'_'+day, 'wb') as f:
                    pickle.dump(user_dict, f)


### Compute the msrd for a given users ###
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

### load pickle files ###
def unpack_users_timeline(path):
    with open(path, 'rb') as handle:
        info = pickle.load(handle)
    return info

### Filter risky contact according in order to build the tree ###
def filtering(minus,plus,final_list,infected_list):
    filtered_list=[]
    for i,j in zip(np.unique(final_list['sourceId']),np.unique(final_list['targetId'])):
        if i in np.unique(infected_list['Mobile_id']):
            _symp=list(infected_list[infected_list['Mobile_id']==i]['date'])[0]
            d=final_list[final_list['sourceId']==i]
            _cond=np.logical_and(d.sourceTime>=(_symp+timedelta(days=minus)),
                                 d.sourceTime<=(_symp+timedelta(days=plus)))
            filtered_list.append(d[_cond])
        if j in np.unique(infected_list['Mobile_id']):
            _symp=list(infected_list[infected_list['Mobile_id']==j]['date'])[0]
            d=final_list[final_list['targetId']==j]
            _cond=np.logical_and(d.sourceTime>=(_symp+timedelta(days=minus)),
                                 d.sourceTime<=(_symp+timedelta(days=plus)))
            filtered_list.append(d[_cond])
    filtered_list=pd.concat(filtered_list)
    return filtered_list

### For the kcore_betweeness percolation ###
def calculate_out_core(g, deg, start_core, kcore):
    core_nodes = [ v for v,_ in enumerate(g.vs['ids']) if kcore[v]>=start_core ]
    # Select only core nodes that have neighbor outside the core.
    valid_nodes = []
    for v in core_nodes:
        neighbors = g.neighbors(v)#g.get_all_neighbors(v)
        for neigh in neighbors:
            if kcore[neigh] < start_core:
                valid_nodes.append(v)
                break
    
    # Get the total degree of the valid nodes. ### CHOOSE 'total', 'in' or 'out'.
    deg_valid_nodes = [ deg[v] for v in valid_nodes ]
    order = np.argsort(deg_valid_nodes)[::-1]
    for ind in order:
        return valid_nodes[ind]

### define nodes with the Highet BC in a given layer(s) ###
def node_in_layer(g,layers=None):
    if layers==None:
        _name=g.vs.select(_betweenness = np.max(g.betweenness(directed=False)))['ids'][0]
        _ids=g.vs["ids"].index(_name)
        return _name,_ids
    else:
        btw=g.betweenness(directed=False)
        layer=g.vs['layer']
        btw_l,_ids=[],[]
        for i,ii in enumerate(layer):
            if ii <=layers:
                btw_l.append(btw[i])
                _ids.append(i)
            else:
                btw_l.append(0)
                _ids.append(i)
        name=np.transpose(g.vs['ids'])
        _ids=np.transpose(_ids)
        
        return name[btw_l==np.max(btw_l)][0],_ids[btw_l==np.max(btw_l)][0]









