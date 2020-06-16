import os
import sys
import folium
import numpy as np
import geohash
import pandas as pd
import lib.map_utils as map_utils
import lib.network_utils as net_utils
from collections import defaultdict
import graph_tool.all as gt
from matplotlib import cm

class MapCore:
    def __init__(self, net, date, contact_folder, layers=5, rho=0.9, memory_tol=200.0):
        self.net = net
        self.net_date = date
        self.out_folder = contact_folder
        self.number_layer = layers
        self.custom_color = None
        self.cmap_list = None
        self.map = None
        self.cmap_name='hsv'
        
        filtered_file = f'filtered_risky_contacts_{rho}.csv'
        self.contact_table = pd.read_csv(os.path.join(self.out_folder, filtered_file))
        table_memory = sys.getsizeof(self.contact_table)/(10**6)
        if table_memory>memory_tol:
            print(f'Memory warning: table of contacts consuming {table_memory} MB of memory')
        
        vertices_ids = net.vp.ids
        vertices_layer = net.vp.layer
        self.id_to_layer = defaultdict(lambda:-1)
        for v in self.net.get_vertices(): self.id_to_layer[vertices_ids[v]] = int(vertices_layer[v])
    
    def initialize_map(self, m):
        '''
            The folium map 'm' object should be defined outside this class.
        '''
        self.map = m
    
    def get_map(self):
        return self.map
    
    def clear_map(self):
        map_loc = self.map.location
        map_zoom_start = self.map.options['zoom']
        tile = list(self.map.to_dict()['children'].keys())[0]
        self.map = folium.Map(location=map_loc, zoom_start=map_zoom_start, max_zoom=17, min_zoom=7, tiles=tile)
        
    def change_start_core(self, new_core, upgrade_colormap=True, cmap_name='hsv'):
        self.start_core = new_core
        if upgrade_colormap:
            nodecolor, ncomponents = net_utils.label_core_components(self.net, core=self.start_core)
            cmap = cm.get_cmap(cmap_name, ncomponents)
            values = np.linspace(0, 1, ncomponents)
            
            cmap_list = [ cm.colors.to_hex([cmap(v)[0],cmap(v)[1],cmap(v)[2]]) for v in values ]
            self.cmap_list = cmap_list
            
    def set_colormap(self, cmap_name='hsv', size=-1):
        '''
            It is also possible to define a matplotlib colormap to define
            the color of the markers on the map. The size of the colormap
            can be defined with 'size' parameter, which allows 'size' possible
            colors for the markers.
        '''
        if size>-1:
            # Set manually the number of different color values.
            cmap = cm.get_cmap(cmap_name, size)
            values = np.linspace(0, 1, size)
        else:
            # Extract the number of components on the core and use this
            # number to determine the number of different color values.
            nodecolor, ncomponents = net_utils.label_core_components(self.net, core=self.start_core)
            cmap = cm.get_cmap(cmap_name, ncomponents)
            values = np.linspace(0, 1, ncomponents)
        
        cmap_list = [ cm.colors.to_hex([cmap(v)[0],cmap(v)[1],cmap(v)[2]]) for v in values ]
        self.cmap_list = cmap_list
        
    def set_custom_palette(self, color_list, repeat=0):
        '''
            'color_list': list of colors. Contains strings specifying all
            the colors to be used. Both color names as HTML codes are accepted.
            
            'repeat': The number of times the 'color_list' should be repeated
            to increase the color capacity of the map.
        '''
        if repeat>0:
            new_color_list = []
            for k in range(repeat):
                new_color_list += color_list
            self.custom_color = new_color_list
        else:
            self.custom_color = color_list
        
    def change_network(self, g, g_date):
        self.net = g.copy()
        self.net_date = g_date
        vertices_ids = g.vp.ids
        vertices_layer = g.vp.layer
        self.id_to_layer = defaultdict(lambda:-1)
        for v in self.net.get_vertices(): self.id_to_layer[vertices_ids[v]] = int(vertices_layer[v])
            
        map_loc = self.map.location
        map_zoom_start = self.map.options['zoom']
        tile = list(self.map.to_dict()['children'].keys())[0]
        self.map = folium.Map(location=map_loc, zoom_start=map_zoom_start, max_zoom=17, min_zoom=7, tiles=tile)
        
    def set_cores_on_map(self, start_core=5, weight_filter=0, geo_precision=9):
        # decompose the k-shells
        cores = gt.kcore_decomposition(self.net)
        core_info = np.unique(cores.a)
        number_cores = core_info.shape[0]
                  
        # Gets the IDs of all nodes inside core, or shell if 'shell_mode' is True.
        kcore_ids = net_utils.extract_ids_kshell(self.net, shell=start_core, shell_mode=False)
        nodecolor, ncomponents = net_utils.label_core_components(self.net, core=start_core)
        print('Number of components: {}'.format(ncomponents))
        
        filtered_contacts = self.contact_table
        color_list = self.custom_color
        if color_list==None:
            if self.cmap_list==None:
                cmap = cm.get_cmap(self.cmap_name, ncomponents)
                values = np.linspace(0, 1, ncomponents)
                self.cmap_list = [ cm.colors.to_hex([cmap(v)[0],cmap(v)[1],cmap(v)[2]]) for v in values ]
            color_list = self.cmap_list
        
        # using the IDs from the selected core, filter the contacts.
        cond1 = filtered_contacts['sourceId'].isin(kcore_ids)
        cond2 = filtered_contacts['targetId'].isin(kcore_ids)
        eff_table = filtered_contacts[cond1 & cond2]
        place_info = map_utils.coreComponents_on_map(self.map, eff_table, self.net_date, self.id_to_layer, nodecolor, color_list, weight_filter=weight_filter, geo_precision=geo_precision)
        
        # generate structured csv table
        component_dict = place_info[0]
        unique_ids= place_info[1]
        
        table = []
        for comp_index, current_dict in enumerate(component_dict):
            for geo_key in current_dict.keys():
                geoloc = geo_key
                lat, lon = geohash.decode(geoloc)
                number_hits = current_dict[geo_key]
                current_component = comp_index
                color = self.cmap_list[comp_index]
                
                table.append({'geohash': geoloc, 'lat': lat, 'lon': lon,
                              'number_contacts': number_hits, 'component_index': current_component,
                             'color_hex': color })
        
        table = pd.DataFrame.from_dict(table)
        
        return self.map, table
    
    def put_hospital_icons(self, hospital_coordinates, icon_color='red', icon_name='hospital-o'):
        '''
            Receives a list of tuples (lat, lon) containing the geo coordinates of the hospitals
            to put on the map.
        '''
        for xy in hospital_coordinates:
            folium.Marker(location=xy, icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')).add_to(self.map)
        
        
                
        
        
        
        