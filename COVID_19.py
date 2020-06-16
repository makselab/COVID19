# **************************************************************************** #
#                                                                              #
#                                                        		       #
#    COVID_19.py                                  			       #				   
#                                                    			       #	       
#    By: Matteo Serafino <matteo.serafino@imtlucca.it>    		       #			   
#        Higor Monteiro  <higor.monteiro@fisica.ufc.b                          #
#    									       #									   
#    Created: 2020/06/13 11:42    by Matteo Serafino    		       #	           
#    Updated:                                       			       #	           
# *****************************************************************************#   
#
#
#                                                                          
#COVID_19.py can be used to reproduce the results presented in 
#"Superspreading k-cores at the center of pandemic persistence" by Serafino et al.
# 
#The package comes along whis a documetation and a chunck of aggreated fata, which 
#can be useful in order to test the packge itself and to undestand of herarchy of the
#inputs/outputs. 
#
#For more complete data you should contact the reference author.
#
#COVID_19.py comes with ABSOLUTELY NO WARRANTY
#
#
#

###Import lybraries...###
import numpy as np
import pandas as pd

import random
import glob
import collections

from collections import deque
from datetime import date,timedelta,datetime
from igraph import *
from geopy.distance import geodesic
from joblib import Parallel, delayed

from utilities import *




class COVID_19:
    class contact_network:
        def __init__(self,folder,cores=1,chunks=4,rho=0.9,layers=5):
            self.rho=0.9
            self.folder=folder
            self.chunks=chunks
            self.num_cores=cores
            self.layers=layers
        def distance(self,contacts):
            distance=[]
            for _,i in contacts.iterrows():
                distance.append(geodesic((i['sourceLat_avg'],i['sourceLong_avg']), 
                                    (i['targetLat_avg'],i['targetLong_avg'])).km*1000)
            filetered_contacts=contacts[np.asarray(distance)>0.1]
            return filetered_contacts
        def threshold_pc(self,risky_contacts):
            final_edges= self.final_edges[self.final_edges['p']>=self.rho]
            unique_ids_to_consider=np.concatenate([np.unique(final_edges['sourceId']),np.unique(final_edges['targetId'])])
            final_list=risky_contacts[np.logical_or(np.isin(risky_contacts['sourceId'],unique_ids_to_consider),
                                     np.isin(risky_contacts['targetId'],unique_ids_to_consider))]
            return final_list
        def remove_grid(self):
            #upload all the chunks
            risky_contacts=[]
            for i in glob.glob(self.folder+'raw_data/chunk_*'):
                if int(i.split('_')[-1])<=self.chunks:
                    risky_contacts.append(pd.read_csv(glob.glob(i+'/*')[0]))
            risky_contacts=pd.concat(risky_contacts)
            #remove grid: distance between source and target <=0.1
            _contacts=np.array_split(risky_contacts, self.num_cores)
            results=Parallel(n_jobs=self.num_cores)(delayed(self.distance)(i)  for i in _contacts)
            results=pd.concat(results) 
            try:
                results.to_csv(self.folder+'raw_data/filtered_risky_contacts.csv') 
                print('filtered_risky_contacts saved in ',self.folder)
            except:
                print('Error during the saving of the filtered_risky_contact...')  
            return -1
        def filtering(self):
            print('loading risky contacts...')
            risky_contacts=pd.read_csv(self.folder+'raw_data/filtered_risky_contacts.csv')
            risky_contacts['sourceTime']=[pd.to_datetime(datetime.strptime(i[:-15], '%Y-%m-%d'))
                                          for i in risky_contacts['sourceTime']]
            self.final_edges=pd.read_csv(self.folder+'raw_data/final_edges.csv')
            _contacts=np.array_split(risky_contacts,100)
            print('selecting users with critical probability > ',self.rho)
            results=Parallel(n_jobs=self.num_cores)(delayed(self.threshold_pc)(i)  for i in _contacts)
            results=pd.concat(results)
            try: 
                results.to_csv(self.folder+f'raw_data/filtered_risky_contacts_{self.rho}.csv')
                print(f'filtered_risky_contacts__{self.rho} correctly saved in: ',self.folder)
            except:
                print(f'problem during the saving of the filtered_risky_contacts_{self.rho}')

            return -1
        def datapoints(self,start,final,days):
            week=[]
            dt=start#date(2020,3,1)
            while dt <= final:
                week.append(pd.Timestamp(dt))
                dt=dt+timedelta(days=2)
            return week
        def build_contact_network(self,ids,layers,i):
            #create node dictionary
            all_nodes=np.concatenate([i[0] for i in ids if i!=[]])
            ID_dict={j:i for i,j in enumerate(all_nodes)}
            #empt graph
            g = Graph()
            #Add vertices and properties as ids and layer
            g.add_vertices(len(all_nodes))
            g.vs['ids']=all_nodes
            g.vs['layer']=np.concatenate([ii*np.ones(len(i[0])) for ii,i in enumerate(ids) if i!=[]])
            # create edgelist   
            edges,un,unique_connections_clean=[],[],[]
            for f in layers:
                edges+=[(ID_dict[i],ID_dict[j]) for i,j in zip(f[0]['sourceId'],f[0]['targetId']) if i!=j]
            for j in edges:
                if ((j[0],j[1]) not in un) and ((j[1],j[0]) not in un):
                    un.append((j[0],j[1]))
                    un.append((j[1],j[0]))
                    unique_connections_clean.append((j[0],j[1]))  
            #add edges
            g.add_edges(unique_connections_clean)
            #giant component
            g_c=g.components().giant()
            #count infected in the gcc
            ifc=[]
            _=[ifc.append(j) for j in g_c.vs['ids'] if j in ids[0][0]]
            #kcore
            kcore,count=np.unique(g_c.coreness(),return_counts=True)
            ### info to save ###
            table=[len(all_nodes),len(unique_connections_clean),len(ids[0][0]),
                  len([i for i in g_c.degree()]),len([i for i in g_c.es()]),len(np.unique(ifc)),
                  kcore,count]
            #save graph
            g_c.write_gml(self.folder+f'contact_networks/gc_{i}.gml')
            g.write_gml(self.folder+f'contact_networks/g_{i}.gml')
            return table
        def contacts_network(self,start,final,days,window=7):
            infected_list=pd.read_csv(self.folder+'infected_list/Infected_list.csv')#self.in_folder+'Filtered_Matching.csv')
            infected_list['date']=[pd.to_datetime(datetime.strptime(i, '%Y-%m-%d')) 
                                       for i in infected_list['date']]
            final_list=pd.read_csv(self.folder+f'raw_data/filtered_risky_contacts_{self.rho}.csv')
            final_list['sourceTime']=pd.to_datetime(final_list['sourceTime'])
            week=self.datapoints(start,final,days)
            table=[]
            for i,start in enumerate(week):
                #select cotacts week
                interactions=final_list[np.logical_and(final_list['sourceTime']>=start,
                                                     final_list['sourceTime']<start+timedelta(days=window))]
                print(start,start+timedelta(days=window))
                #0 layer contacts
                layer_0=interactions[np.logical_or(np.isin(interactions['sourceId'],infected_list['Mobile_id']),
                              np.isin(interactions['targetId'],infected_list['Mobile_id']))]
                all_ids=np.concatenate([np.unique(layer_0['sourceId']),np.unique(layer_0['targetId'])])
                zero_layer_ids=np.unique(all_ids[np.isin(all_ids,infected_list['Mobile_id'])])
                first_layer_ids=np.unique(all_ids[np.isin(all_ids,infected_list['Mobile_id'],invert=True)])
                #j layer contacts
                layers,ids=[],[]
                for j in np.arange(0,self.layers,1):
                    layers.append([])
                    ids.append([])
                    if j==0:
                        ids.append([])
                        layers[0].append(layer_0)
                        ids[0].append(zero_layer_ids)
                        ids[1].append(first_layer_ids)
                    else:
                        layers[j].append(interactions[np.logical_or(np.isin(interactions['sourceId'],ids[j]),
                                          np.isin(interactions['targetId'],ids[j]))])
                        all_ids=np.concatenate([np.unique(layers[j][0]['sourceId']),np.unique(layers[j][0]['targetId'])])
                        ids[j+1].append(np.unique(all_ids[np.isin(all_ids,np.concatenate([i[0] for i in ids if i!=[]]),
                                                      invert=True)]))
                _table=self.build_contact_network(ids,layers,i)
                table.append(_table)
            np.save(self.folder+'/simulations_results/G_Statistic',table)
            return -1
    class Percolation:
        def __init__(self, g, radius,layers=None):
            self.g = g
            self.cut_off=0.0005
            self.nodes_giant=len(g.degree())
            self.layers=layers
            self.rad=radius
        def get_ball_boundary(self,g,v_id,rad,num_vertices):
            """ 
                Returns an array of nodes ids of the graph `G`
                at the boundary of the ball of radius `rad` 
                centered on 'v_id'.
               `num_vertices` is the number of vertices of G.

            """
            color = np.zeros(num_vertices, dtype=np.int32) #zero=white
            dist = np.zeros(num_vertices, dtype=np.int32)

            l = 0 # explored radius
            color[v_id] = 1 # 1=gray (discovered)

            q = deque()

            boundary_nodes = []

            q.append(v_id)
            while l <= rad and q:
                s_id = q.popleft()
                out_neighbours = g.neighbors(s_id)
                imax = len(out_neighbours)
                for i in range(imax):
                    t_id = out_neighbours[i]
                    if color[t_id] == 0: #new node
                        color[t_id] = 1
                        l = dist[s_id] + 1
                        dist[t_id] = l

                        if l < rad:
                            q.append(t_id)
                        elif l == rad:
                            boundary_nodes.append(t_id)
            return np.array(boundary_nodes, np.uint64)

        def compute_node_CI_numpy(self,v_id,g,rad, num_vertices):
            """ 
                Computes and returns the CI value of vertex `v_id` 
                using a ball radius `rad` in the graph `G`.
                `k_map` is an array with the degree map to be used
                and `num_vertices` is the total number of vertices 
                of the graph.

            """
            k_v = g.degree(v_id)

            if k_v == 0 or k_v == 1:
                return 0

            boundary_nodes = self.get_ball_boundary(g, 
                                              v_id, 
                                              rad, 
                                              num_vertices)

            boundary_degrees = np.asarray(g.degree(boundary_nodes))

            return np.sum(boundary_degrees -1)*(k_v - 1)

        def CI_graph(self,g):
            """ 
                Returns an array containing the CI
                for each node in the network
            """
            CI=[]
            for i in g.vs["ids"]:
                _ids=g.vs["ids"].index(i)
                ci=self.compute_node_CI_numpy(_ids, g.copy(),  self.rad, len(g.degree()))
                CI.append(ci)
            return CI
        def delete_node(self,g):
            """ 
                Returns to the ids of the node in the higest kcore
                with highest value of BC       
            """
            dicti_idsbtw,dicti_idsshell={},{}
            for i,ii,jj in zip(g.vs['ids'],g.coreness(),g.betweenness(directed=False)):
                dicti_idsshell[i]=ii
                dicti_idsbtw[i]=jj
            maximum=0
            for i,j in zip(dicti_idsshell.values(),dicti_idsshell.keys()):
                if i==np.max(list(dicti_idsshell.values())):
                    if dicti_idsbtw[j] >maximum:
                        ids=j 
                        maximum=dicti_idsbtw[j]
            return ids

        def percolation(self,mode='random',g=None):
            """ 
                Main: returns to the vector
                G(q) and q. The default mode=='random'
            """
            if g==None:
                g = self.g.copy()
            else:
                g = g.copy()
            Giant_dimension,q=[],[]
            removed=0
            gc_fraq= 1
            Giant_dimension.append(gc_fraq)
            q.append(removed)
            try:
                while gc_fraq > self.cut_off:
                    #select name of the node with higest degree and get ids
                    if mode=='random':
                        _name=random.choice([i for i in g.vs['ids']])
                        _ids=g.vs["ids"].index(_name)
                    elif mode=='degree':
                        _name=g.vs.select(_degree = g.maxdegree())['ids'][0]
                        _ids=g.vs["ids"].index(_name)
                    elif mode=='betweeness':
                        _name,_ids=node_in_layer(g,layers=self.layers)
                        #_name=g.vs.select(_betweenness = np.max(g.betweenness(directed=False)))['ids'][0]
                        #_ids=g.vs["ids"].index(_name)  
                    elif mode=='closeness':
                        _name=g.vs.select(_closeness = np.max(g.closeness()))['ids'][0]
                        _ids=g.vs["ids"].index(_name)  
                    elif mode=='CI':
                        CI=self.CI_graph(g)
                        _ids=np.where(CI==np.max(CI))[0][0]
                    elif mode=='KC+BC':
                        _name= self.delete_node(g)
                        _ids=g.vs["ids"].index(_name) 
                    elif mode=='KC+HB':
                        g.vs["shell"]=g.coreness()
                        _names=g.vs.select(shell = np.max(g.coreness()))['ids']
                        _max=0
                        for s,ss in zip(g.degree(),g.vs['ids']):
                            if ss in _names:
                                if s>_max:
                                    _max=s
                                    _name=ss
                        _ids=g.vs['ids'].index(_name) 
                    #delete node
                    g.delete_vertices(_ids)
                    #recomupute giant component
                    g=g.components().giant()
                    gc_fraq=len(g.degree())/self.nodes_giant
                    removed+=1
                    Giant_dimension.append(gc_fraq)
                    q.append(removed)
            except:
                   print('No GCC')
            return Giant_dimension,q

        def by_kcore_betweenness(self,core_frac=0.55):
            g = self.g.copy()
            Giant_dimension,q=[],[]
            removed=0
            gc_fraq= 1
            Giant_dimension.append(gc_fraq)
            q.append(removed)  
            kcore = g.coreness()
            core_indexes = np.unique(kcore)
            normed_cores = core_indexes/core_indexes[-1]
            for ind, k in enumerate(normed_cores):
                if k>=core_frac: 
                    start_core = core_indexes[ind]
                    break
            # First part: Isolate k-core
            max_core = core_indexes[-1]
            while max_core>=start_core:
                deg =[i for i in g.degree()]
                # Select the right node in the core to remove.
                node_to_remove = calculate_out_core(g, deg, start_core, kcore)
                if node_to_remove>-1:
                    g.delete_vertices(node_to_remove)
                else:
                    break
                g= g.components().giant()
                removed+=1
                Giant_dimension.append(len(g.degree())/self.nodes_giant)
                q.append(removed)
                kcore = g.coreness()
                core_indexes = np.unique(kcore)
                max_core = core_indexes[-1]

            # Second part: Remove by betweenness.
            _sum=q[-1]
            _GCC,_q=self.percolation(mode='betweeness',g=g)
            for i,j in zip(_GCC[1:],_q[1:]):
                Giant_dimension.append(i)
                q.append(j+_sum)
            return  Giant_dimension,q   
		
    class SIR:
        def __init__(self, beta_max,folder,name,sampling=100,layer=0.0,cores=1):
            self.samplings=sampling
            self.beta = beta_max
            self.folder=folder
            self.layer=layer
            self.num_cores=cores
            self.name=name
        def col(self,obj,i):
            if obj[i]==0:
                obj[i]=1
        def SIR_model(self,node,_beta):
            #zero stays for S
            M=[]
            for _ in range(self.samplings):
                #color = np.zeros(len(self.g.get_out_degrees(self.g.get_vertices())), dtype=np.int32)
                color = np.zeros(len([i for i in self.g.degree()]), dtype=np.int32)
                #activate the first node
                _colors=[self.col(color,i) for i in node]
                I=np.where(color==1)[0]
                #continue until the percentage of infected (I+R) is below a given threshold
                while len(I)>0:
                    for j in I:
                        #ramification of each neighbour
                        ##this nhbs is here because the function g.get_all_neighbors(j) does not work##
                        ##in principle it would be enough to use g.get_all_neighbors(j)##
                        nhbs=self.g.neighbors(j)#self.g.get_out_neighbors(j)
                        _infected_nodes=[i for i in nhbs if (random.random() < _beta and color[j]==1)]
                        _colors=[self.col(color,i) for i in _infected_nodes]
                        color[j]=2
                    I=np.where(color==1)[0]
                    R=np.where(color==2)[0]
                M.append(len(I)+len(R))
            return np.mean(M)
        def sampling(self,beta_range):
            results=[]
            for _beta in beta_range:
                for _shells in np.unique(self.infected_net.shell):
                    _shell=self.infected_net[np.isin(self.infected_net.shell,_shells)]
                    #SIR starting from each of this ids
                    for _idx in _shell.idx:
                        results.append([_beta,_shells,self.SIR_model([int(_idx)],_beta)])
            return results
        def average_shell(self,results):
            #This function compute the average infected population
            #for each shell. It return to a numpy vector.
            res=pd.DataFrame(results)
            _all=[]
            for k in np.unique(res[1]):
                _res=res[res[1]==k]
                y=[]
                for i in np.unique(res[0]):
                    _res_1=_res[_res[0]==i]
                    y.append(np.mean(_res_1[2]))
                _all.append(np.divide(y,len(self.g.degree())))
            _all.append(np.unique(res[0]))
            return _all
        def run_SIR(self,g):
            self.g=g.copy()#Graph.Read_GML(in_folder)
            ##get_total_degree does not work. To obtain it i sum over in and out ##
            degree=[i for i in self.g.degree()]#self.g.get_out_degrees(self.g.get_vertices())
            table=[]
            #gt.kcore_decomposition(self.g)
            for ii,i,j,k,l in zip(np.arange(0,len(degree)+1),self.g.vs['ids'],self.g.coreness(),degree, self.g.vs['layer']):
                table.append([ii,i,j,k,l])
            net=pd.DataFrame(table,columns=['idx','ids','shell','k','layer'])
            net[['idx','shell', 'k','layer']] = net[['idx','shell', 'k','layer']].astype('float64')
            self.infected_net=net[np.isin(net['layer'],np.arange(0,self.layer+1,1))]
            if self.num_cores==1:
                results=self.sampling(np.arange(0.0,self.beta,0.02))
            else:    
                betas_per_core=[np.arange(0.0,self.beta,0.02)[i::self.num_cores] for i in range(self.num_cores)]
                results=Parallel(n_jobs=self.num_cores)(delayed(self.sampling)(i)  for i in betas_per_core)
                results=np.concatenate(results)
            result=self.average_shell(results)
            try:
                np.save(self.folder+f'simulations_results/SIR_{int(self.layer)}_'+self.name,result)
                print('file succefully saved in:',self.folder+'/simulation_results')
            except:
                print('Error during the saving:',self.folder+'/simulation_results')
            return -1

    class msrd:
        def __init__(self, directory,out_directory):
                self.input_dir = directory
                self.out_directory=out_directory
        def generate_pickle(self):
            for fname in tqdm(glob.glob(self.input_dir+'*')):
                write_pickle(fname,self.out_directory+'/msd_pickle/') 
            return -1
        def daily_msrd(self):
            daily=[]
            for fname in tqdm(glob.glob(self.out_directory+'/msd_pickle/'+'*')):
                month=fname.split('/')[-1].split('_')[0]
                day=int(fname.split('/')[-1].split('_')[1])
                users_table= []
                info = unpack_users_timeline(fname)
                for uid in info.keys():
                    if len(info[uid])>0:
                        user_lat, user_lon, user_time = info[uid][0], info[uid][1], info[uid][2]
                        rms_info = get_user_rms(user_lat, user_lon, user_time)
                    else:
                        rms_info = (np.nan, np.nan, np.nan)
                    users_table.append(rms_info[2])
                daily.append([np.mean([x for x in users_table if str(x) != 'nan']),date(2020,int(datetime.strptime(month, "%B").month),day)])
            daily=pd.DataFrame(daily,columns=['rmsd','date'])
            daily.to_csv(self.out_directory+'simulation_results/daily_rmsd.csv')
            return -1

		
		
		
		
		
			
		
		

if __name__ == "__main__":
    ##13/06/2020
    #folder='/home/alex/Matteo_S/CoronaVirus/Test_library/'
    #trial=COVID_19.contact_network(folder,cores=6)
    #_=trial.remove_grid()
    #_=trial.filtering()
    #_=trial.contacts_network(date(2020,3,1),date(2020,5,1),2,window=7)

    ##13/06/2020
    #g=Graph.Read_GML('contact_networks/gc_4.gml')
    #_output='/home/alex/Matteo_S/CoronaVirus/Test_library/'
    #sir_analyses=COVID_19.SIR(0.5,_output,'_2_',sampling=200,cores=6)
    #res=sir_analyses.run_SIR(g)

    ##14/06/2020
    directory='/media/wangjiannan/raw_data_FL/raw_data/'
    out_directory='/home/alex/Matteo_S/CoronaVirus/Test_library/'
    mrsd=COVID_19.msrd(directory,out_directory)
    #_=mrsd.generate_pickle()
    _=mrsd.daily_msrd()
