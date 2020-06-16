import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import geohash
import geopandas as geopd
import graph_tool.all as gt
from collections import defaultdict, Counter

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

###########################################################
## ------------- EXTRA FUNCTIONS FOR MAPS -------------- ##
###########################################################
def extract_ids_kshell(g, shell=1, shell_mode=True):
    '''
        Get the nodes' Ids inside the given shell or given core
        To select only the nodes in the shell, 'shell_mode' should
        be true. Otherwise, we select the nodes in the core.
    '''
    kc = gt.kcore_decomposition(g)
    nodeId = g.vp.ids
    ids = []
    for v in g.get_vertices():
        if shell_mode:
            if kc[v]==shell:
                ids.append(nodeId[v])
        elif kc[v]>=shell:
            ids.append(nodeId[v])
    return ids

def label_shell_components(g, shell=1):
    kc = gt.kcore_decomposition(g)
    nodeId = g.vp.ids
    v_shell = g.new_vertex_property('bool')
    for v in g.get_vertices():
        if kc[v]==shell:
            v_shell[v] = 1
        else:
            v_shell[v] = 0
            
    g.set_vertex_filter(v_shell)
    labels, val = gt.label_components(g)
    nodecolor_comp = defaultdict(lambda:-1)
    for v in g.get_vertices():
        if v_shell[v]==1:
            nodecolor_comp[nodeId[v]] = labels[v]
    # Recover the original network.
    g.set_vertex_filter(None)
    return nodecolor_comp

def label_core_components(g, core=1):
    kc = gt.kcore_decomposition(g)
    nodeId = g.vp.ids
    
    v_core = g.new_vertex_property('bool')
    for v in g.get_vertices():
        v_core[v] = 0
        if kc[v]>=core:
            v_core[v] = 1
            
    g.set_vertex_filter(v_core)
    labels, val = gt.label_components(g)
    nodecolor_comp = defaultdict(lambda:-1)
    for v in g.get_vertices():
        if v_core[v]==1:
            nodecolor_comp[nodeId[v]] = labels[v]
    # Recover the original network.
    g.set_vertex_filter(None)
    return nodecolor_comp, val.shape[0]
#########################################################

class VisitorExample(gt.BFSVisitor):
    def __init__(self, pred, dist):
        self.pred = pred
        self.dist = dist

    def tree_edge(self, e):
        self.pred[e.target()] = int(e.source())
        self.dist[e.target()] = self.dist[e.source()] + 1
        
######################################################
### ------------ ID_TREE_DESCRIPTION ------------- ###
######################################################

def plot_tree(g, ID, max_dist=8, mpl_colormap='viridis', kcore_base=4, return_tree=False):
    '''
        Automatic plotting of the tree of shortest path.
    '''
    id_info = ID_TREE(g)
    id_info.initialize_id_node_dict()
    node_index = id_info.get_node_from_id(ID)
    v_list, dist_list = id_info.rank_bfs(source=node_index, max_dist=max_dist)
    g_tree = id_info.build_tree(source=node_index, targets=v_list, dist_target=dist_list)
    
    #node_color, node_size = id_info.get_colorcore_map(g_tree)
    node_color, node_size = id_info.set_aesthestic(g_tree, kcore_base=kcore_base, mpl_colormap=mpl_colormap)
    if return_tree:
        return g_tree, node_color, node_size
    
    pos = gt.radial_tree_layout(g_tree, g_tree.vertex(0)) # zero is always the root.
    gt.graph_draw(g_tree, pos=pos, vertex_color=node_color, vertex_fill_color=node_color, vertex_size=node_size, edge_pen_width=1.0)

def get_ids_from_distance(g, sourceId, dist_shell=0):
    '''
        Given a source node, returns all IDs from the nodes at
        distance 'dist_shell' from the source.
    '''
    id_info = ID_TREE(g)
    id_info.initialize_id_node_dict()
    node_index = id_info.get_node_from_id(sourceId)
    v_list, dist_list, uid_list = id_info.rank_bfs(source=node_index, min_dist=dist_shell, max_dist=dist_shell, return_ids=True)
    return uid_list
    
# ------------------------------------------------------------- #

# ===> OBJECT
class ID_TREE:
    def __init__(self, g):
        self.g = g
        self.id_to_node = defaultdict(lambda:-1)
        
    def initialize_id_node_dict(self):
        ids = self.g.vp.ids
        for v in self.g.get_vertices():
            self.id_to_node[ids[v]] = v
            
    def get_node_from_id(self, ID):
        return self.id_to_node[ID]
    
    def rank_bfs(self, source=None, min_dist=0, max_dist=10, return_ids=False):
        if source==None:
            return None
        else:
            dist = self.g.new_vp("int")
            pred = self.g.new_vp("int64_t")
            dist.a = -1
            gt.bfs_search(self.g, source=source, visitor=VisitorExample(pred, dist))
            nodelist = [ v for v in self.g.get_vertices() if dist[v]!=-1 and dist[v]>=min_dist and dist[v]<=max_dist ]
            node_distlist = [ dist[v] for v in self.g.get_vertices() if dist[v]!=-1 and dist[v]>=min_dist and dist[v]<=max_dist ]
            
            if return_ids:
                ids = self.g.vp.ids
                uid_list = [ ids[v] for v in nodelist ]
                return nodelist, node_distlist, uid_list
            return nodelist, node_distlist
        
    def get_ID_firstlayer(self):
        pass
    
    def build_tree(self, source=None, targets=None, dist_target=None):
        '''
            Starting from a source node and a list of nodes reachable from
            'source', then we create a new network where we only include the
            edges of the shortest path from 'source' to 'targets'. This way,
            the final network is a tree.
        '''
        
        if source==None:
            return None
        else:
            source_tree = gt.Graph(directed=False)
            g_to_tree = defaultdict(lambda:-1)
            tree_to_g = defaultdict(lambda:-1)
            g_to_tree[source] = 0
            tree_to_g[0] = source
            last_index = 0
            
            edgelist = []
            for tgt in targets:
                vlist, elist = gt.shortest_path(self.g, self.g.vertex(source), self.g.vertex(tgt))
                for v in vlist:
                    if g_to_tree[v]==-1:
                        g_to_tree[v] = last_index+1
                        tree_to_g[last_index+1] = v
                        last_index += 1
                for e in elist:
                    edgelist.append([int(e.source()), int(e.target())])
                    
            new_edgelist = []
            for e in edgelist:
                new_edgelist.append([g_to_tree[e[0]], g_to_tree[e[1]]])
                    
            g_tree = gt.Graph(directed=False)
            fmt_edgelist = np.vstack(new_edgelist)
            fmt_edgelist = np.unique(fmt_edgelist, axis=0)
            
            g_tree.add_edge_list(fmt_edgelist)
            original_index = g_tree.new_vp('int')
            for v in g_tree.get_vertices():
                original_index[v] = tree_to_g[v]
            g_tree.vertex_properties['original_index'] = original_index
            return g_tree
    
    def set_aesthestic(self, g_tree, kcore_base=4, mpl_colormap='viridis'):
        
        kcore_map = gt.kcore_decomposition(self.g)
        kcore_max = np.unique(kcore_map.a)[-1]
        cmap = cm.get_cmap(mpl_colormap, kcore_max)
        
        normal_size = 4
        core_size = 6
        source_size = 15
        node_color = g_tree.new_vp('string')
        node_size = g_tree.new_vp('int')
        node_size.a = normal_size
        
        original_index = g_tree.vp.original_index
        for v in g_tree.get_vertices():
            core_v = kcore_map[original_index[v]]
            if core_v<kcore_base:
                rgba = cmap(0.0)
                cur_color = cm.colors.to_hex([ rgba[0], rgba[1], rgba[2] ])
                node_color[v] = cur_color
                node_size[v] = normal_size
            else:
                rgba = cmap((core_v)/(kcore_max))
                cur_color = cm.colors.to_hex([ rgba[0], rgba[1], rgba[2] ])
                node_color[v] = cur_color
                node_size[v] = core_size
        
        g_tree.vertex_properties['node_color'] = node_color
        g_tree.vertex_properties['node_size'] = node_size
        node_size[g_tree.vertex(0)] = source_size
        return node_color, node_size
    
    def get_colorcore_map(self, g_tree, kcore_base=4):
        '''
            defines property maps to hold the size and color of
            the vertices of the given 'g_tree'. The color is defined
            using the k-core value of each node. 
        '''
        colors = ['#C02F1D', '#107896']
        kcore_map = gt.kcore_decomposition(self.g)
        
        normal_size = 5
        core_size = 6
        source_size = 15
        
        node_color = g_tree.new_vp('string')
        node_size = g_tree.new_vp('int')
        node_size.a = normal_size
        
        original_index = g_tree.vp.original_index
        for v in g_tree.get_vertices():
            core_v = kcore_map[original_index[v]]
            if core_v>=kcore_base:
                node_color[v] = colors[0]
                node_size[v] = core_size
            else:
                node_color[v] = colors[1]
                
        g_tree.vertex_properties['node_color'] = node_color
        g_tree.vertex_properties['node_size'] = node_size
        node_size[g_tree.vertex(0)] = source_size
        return node_color, node_size
################################################################

