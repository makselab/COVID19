import os
import pickle
import pytz
import folium
import numpy as np
from tqdm import tqdm
import pandas as pd
import geohash
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

from folium import FeatureGroup, LayerControl, Map, Marker

Forta = pytz.timezone("America/Fortaleza")

def coreComponents_on_map(folium_obj, eff_table, date, id_layer, nodecolor_dict, set_color, days_after=7, weight_filter=10, geo_precision=9):
    '''
        Given a map, or layer, folium object, it plots circle markers for the places
        where contacts happened. The colors of the circle is related to the components
        of the original network.
        
        'eff_table' is the table of contacts. It must have the time of the contacts, the
        source and targets IDs, and the average position of the contact.
        
        'folium_obj' is a folium map or layer where the circles are plotted.
        
        'date' is the date of obtention of the original network.
        
        'id_layer' is the dictionary returning the layer index of the ID, given as
        a key.
        
        'nodecolor_dict' is the dictionary with the relation 'ID' -> component of the
        ID. It is used for proper coloring.
        
        'set_color' is a list of HTML colors. The size of the list must be at least
        equal to the number of components of the original network.
    '''
    # For each component of the aggregated core, we create a dictionary
    # containing the contacts between the node inside the component.
    ncomponents = set()
    for key in nodecolor_dict.keys():
        ncomponents.add(nodecolor_dict[key])
    ncomponents = len(list(ncomponents))
    geodict_all = [ defaultdict(lambda:0) for x in range(ncomponents) ]
    ids_all = [ [] for x in range(ncomponents) ]
    
    ## ----- Calculate the points of the map ------ ##
    for index, row in eff_table.iterrows():
        src, target = row['sourceId'], row['targetId']
        lat, lon = row['targetLat_avg'], row['targetLong_avg']
        srclat, srclon = row['sourceLat_avg'], row['sourceLong_avg']
        dd = gdist.geodesic((lat,lon), (srclat, srclon)).m
        if dd<=0.02: continue  # grid points - anomalies
        src_time = datetime.fromisoformat(row['sourceTime'])
        
        begin_interval = date - timedelta(days=0)
        final_interval = date + timedelta(days=days_after)
        # check if the contact is inside time window.
        if src_time.date()<begin_interval or src_time.date()>final_interval: continue
        # Get contacts only between the IDs in the same component.
        if nodecolor_dict[src]!=nodecolor_dict[target]:
            continue
        else:
            src_color = nodecolor_dict[src]
            
        geocode = geohash.encode(lat,lon,precision=geo_precision)
        cur_geodict = geodict_all[src_color]
        cur_geodict[geocode] += 1
        ids_all[src_color].append(src)
        ids_all[src_color].append(target)
        
    ## -------- Plot the points on the map --------- ##
    for index, geodict in enumerate(geodict_all):
        for geocode in geodict.keys():
            weight = geodict[geocode]
            circle_color = set_color[index]
            if weight<weight_filter:
                continue
            if weight<150:
                weight = 150
            if weight>700: 
                weight = 700
            lat, lon = geohash.decode(geocode)
            folium.Circle(location=[lat, lon], radius=weight, color=circle_color, fill=True).add_to(folium_obj)
    return geodict_all, ids_all

############################################################
# ------------------ MOVIE FUNCTIONS --------------------- #
############################################################

def set_features(lines):
    features = [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': line['coordinates'],
            },
            'properties': {
                'icon': 'circle',
                'times': line['dates'],
                'style': {
                    'color': line['color'],
                    'weight': 2,
                },
                'iconstyle': {'iconColor': line['color']}
            }
        }
        for line in lines
    ]
    return features

def put_lines(uid_points, lines, color='cyan'):
    for k in range(len(uid_points[0])-1):
        lat_before = uid_points[0][k]
        lat_after = uid_points[0][k+1]
        lon_before = uid_points[1][k]
        lon_after = uid_points[1][k+1]
        date_before = datetime.fromtimestamp(uid_points[2][k], Forta)
        date_after = datetime.fromtimestamp(uid_points[2][k+1], Forta)
        date_before_str = datetime.strftime(date_before, "%Y-%m-%d %H:%M:%S")
        date_after_str = datetime.strftime(date_after, "%Y-%m-%d %H:%M:%S")
        
        lines.append({'coordinates': [[lon_before, lat_before], [lon_after, lat_after]],
                      'dates': [date_before_str, date_after_str],
                      'color': color})
    return lines
