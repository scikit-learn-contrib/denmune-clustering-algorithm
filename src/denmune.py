# ====================================================================================================================
# About the source code and the associated published paper               
# ====================================================================================================================
# This is the source code of DenMune Clustering Algorithm accompanied with the experimental work
# which is published in Elsevier Pattern Recognition, Volume 109, January 2021
# paper can be accessed from 107589 https://doi.org/10.1016/j.patcog.2020.107589
# source code and several examples on using it, can be accessed from 
# Gitbub's repository at https://github.com/egy1st/denmune-clustering-algorithm
# Authors: Mohamed Abbas, Adel El-Zoghabi, and Amin Shoukry
# Edition 0.0.2.3 Released 29-12-2021
# PyPi package installation from  https://pypi.org/project/denmune/
# ====================================================================================================================


# ====================================================================================================================
# About the DenMune Algorithm
# ====================================================================================================================
# DenMune Clustering Algorithm's Highlights
# DenMune is a clustering algorithm that can find clusters of arbitrary size, shapes and densities in two-dimensions.
# Higher dimensions are first reduced to 2-D using the t-sne.
# The algorithm relies on a single parameter K (the number of nearest neighbors).
# The results show the superiority of DenMune.
# =====================================================================================================================


# =====================================================================================================================
# About me
# =====================================================================================================================
# Name: Mohamed Ali Abbas
# Egypt - Alexandria - Smouha
# Cell-phone: +20-01007500290
# Personal E-mail: mohamed.alyabbas@outlook.com
# Business E-meal: 01@zerobytes.one
# website: https://zerobytes.one
# LinkedIn: https://www.linkedin.com/in/mohabbas/
# Github: https://github.com/egy1st
# Kaggle: https://www.kaggle.com/egyfirst
# Udemy: https://www.udemy.com/user/mohammad-ali-abbas/
# Facebook: https://www.facebook.com/ZeroBytes.One
# =====================================================================================================================

import numpy as np
import ngtpy
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import genfromtxt
from anytree import Node, RenderTree
from sklearn.manifold import TSNE
import operator
import os.path
import time
from treelib import Node as nd, Tree as tr

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.99, 's' : 80, 'linewidths':0}

# import for possible needs
#from sklearn.metrics import confusion_matrix
#from sklearn import metrics
#import sklearn.cluster as cluster


class DataPoint():
    
    def __init__ (self, id):
              
        self.point_id = id
        self.class_id = 0 # 0 not clustered but -1 means a noise
        self.refer_to = []
        self.referred_by = []
        self.reference = []
        self.visited = False
        self.homogeneity = 0
        
         
class DenMune():
    
    
    #def __init__ (self, dataset, k_nearest=1, data_path='', verpose=True, show_plot=True, show_noise=False):
    def __init__ (self, data, file_2d ='', k_nearest=1, verpose=True, show_noise=False, rgn_tsne=False, prop_step=0):     
         
        self.analyzer = {}
        self.analyzer['exec_time'] = {}
        self.analyzer['n_points'] = {}
        self.analyzer['n_points']["noise"] = {}
        self.analyzer['n_points']["weak"] = {}
        self.analyzer["n_points"]["weak"]["all"] = 0
        self.analyzer["n_points"]["size"] = data.shape[0]
        self.analyzer["n_points"]["dim"] = data.shape[1]
        self.analyzer["validity"] = {}
        self.analyzer["n_clusters"] = {}
        self.analyzer["n_clusters"]["actual"] = 0
        self.analyzer["n_clusters"]["detected"] = 0
        
        if data.shape[1] != 2:
            if  not os.path.isfile(file_2d) or rgn_tsne == True:
                start = time.time()
                self.generate_tsne(data, file_2d, 2)
                end = time.time()
                self.analyzer["exec_time"]["t_SNE"] = end-start
                #if verpose:
                #    print('using t-SNE dataset has been reduced to 2-d in ', self.analyzer["exec_time"]["t_SNE"], ' seconds')

            data = genfromtxt(file_2d , delimiter='\t')


        start_time = time.time()
        
        self.alg_name = 'denmune'
        self.show_noise = show_noise
        self.prop_step = prop_step
        self.verpose = verpose
        self.data = data
        self.dp_count = self.data.shape[0] 
        self.dp_dim = self.data.shape[1]
        self.k_nearest = k_nearest
        self.dp_dis = []
        self.labels = []
        #self.n_noise = {}
               
        self.DataPoints = []
        self.ClassPoints = {}
        self.KernelPoints = []
       
        self.init_DataPoints()
        self.kd_NGT()
        self.load_DataPoints() # load_DataPoints must come after kd_NGT()
        self.compute_Links()
        #self.semi_init_DataPoints #it is usuful with csharp and cnune algorithms only
        self.find_Noise()
        self.sort_DataPoints()
        self.prepare_Clusters()
        self.attach_Points()
        #self.fit_predict()
       
        end_time = time.time() 
        self.analyzer["exec_time"]["DenMune"] = end_time-start_time
        #if verpose:
        #    print('DenMune took',  self.analyzer["exec_time"]["DenMune"] , ' seconds', 'to cluster the dataset' )
            
        return None # __init__ should return Nine
    
      
    
    def kd_NGT(self):
        
        if len(self.dp_dis) == 0:
                    
            ngtpy.create(b"tmp", self.dp_dim)
            index = ngtpy.Index(b"tmp")
            index.batch_insert(self.data)
            index.save()

            k= self.k_nearest

            start = time.time()
            self.dp_dis = []
            for i in range(self.dp_count):
                query = self.data[i]
                result = index.search(query, k+1) [1:] #we skip first distance from a point to itself 
                self.dp_dis.append(result)

            end =  time.time()
            self.analyzer["exec_time"]["NGT"] = end-start
          
            #if self.verpose:
            #    print ('using NGT, Proximity matrix has been calculated  in: ', self.analyzer["exec_time"]["NGT"], ' seconds')
            #self.data = None
                       
        
    def getValue(self, dic, what, who, other=False):
    
        if what == 'max' and who == 'key' and other==False:
            val = max(dic.items(), key=operator.itemgetter(0))[0] #max key
        elif what == 'max' and who == 'key' and other==True: 
            val = max(dic.items(), key=operator.itemgetter(0))[1] #max key==>Value
        elif what == 'max' and who == 'value' and other==True:  
            val = max(dic.items(), key=operator.itemgetter(1))[0] #max value==>key
        elif what == 'max' and who == 'value'and other == False: 
            val = max(dic.items(), key=operator.itemgetter(1))[1] #max value

        return val   
        
        

    def init_DataPoints(self):
        
        self.DataPoints = []
        self.KernelPoints = []
            
        for i  in range (self.dp_count):
            dp = DataPoint(i)
        #no need since datapoint is initialised with these values   
            """ 
            dp.refer_to = []
            dp.referred_by = []
            dp.reference = []
            dp.class_id = 0
            dp.visited = False
            dp.homogeneity = 0.0
            """
            self.DataPoints.append(dp)
        return 0
    
    def semi_init_DataPoints(self):
        
        for dp in self.DataPoints:
            dp.visited= False
            dp.class_id = 0
            dp.homogeneity = 0
            
        return 0    
    
    def find_Noise(self):
        
        self.ClassPoints[-1] = Node(-1, parent=None)
        self.ClassPoints[0] = Node(0, parent=None)
        
        for i  in range (self.dp_count):
            dp = self.DataPoints[i]
            if len(dp.reference) == 0:
                dp.class_id = -1
                self.ClassPoints[i] = self.ClassPoints[-1] # Node(-1, parent=None) # this it is a noise
            else : # at least one point
                dp.class_id = 0 #this is allready set initally
                self.ClassPoints[i] = self.ClassPoints[0] #Node(0, parent=None) # this it is a non-clustered point
                # where -1 is noise and 0 is non-clustered
        return 0
        
   
    def sort_DataPoints(self):
        
        for dp in self.DataPoints:
            if len(dp.reference) != 0:
                self.KernelPoints.append([dp.point_id, dp.homogeneity])
            
        self.KernelPoints = self.sort_Tuple(self.KernelPoints, reverse=True)    
      
        return 0
    
    def compute_Links(self):
        start = time.time()
        
        for i  in range (self.dp_count) :
            for pos in self.DataPoints[i].refer_to:
                              
                for pos2 in self.DataPoints[i].referred_by:
                    if pos[0] == pos2[0]:
                        self.DataPoints[i].reference.append(pos)
                        break
                                                
        self.analyzer["n_points"]["strong"] =   0
        for i  in range (self.dp_count) :
            self.DataPoints[i].referred_by = self.sort_Tuple(self.DataPoints[i].referred_by, reverse=False)
            if len(self.DataPoints[i].referred_by) >= self.k_nearest:
                 self.analyzer["n_points"]["strong"]  += 1
            else:
                self.analyzer["n_points"]["weak"]["all"] += 1
                
            self.DataPoints[i].reference = self.sort_Tuple(self.DataPoints[i].reference, reverse=False)
            homogeneity = (100 * len(self.DataPoints[i].referred_by)) + len(self.DataPoints[i].reference)
            self.DataPoints[i].homogeneity = homogeneity
            
        end = time.time()  
        #print ('Compute links tools:', end-start)
            
        return 0    

    
    def sort_Tuple(self, li, reverse=False): 
         
        # reverse = None (Sorts in Ascending order) 
        # key is set to sort using second element of  
        # sublist lambda has been used 
        li.sort(key = lambda x: x[1], reverse=reverse) 
        return li 
    
    def load_DataPoints(self):
   
        #initialize datapoints to its default values
        self.init_DataPoints()

        for i  in range (self.dp_count) :
            result = self.dp_dis[i]
            for k, o in enumerate(result):
                if k >=  self.k_nearest:
                    break
                
               
                #if k != 0:
                _dis = round(o[1], 6)
                _point = o[0]

                self.DataPoints[i].refer_to.append([_point, _dis])
                self.DataPoints[_point].referred_by.append([i, _dis]) 
                #print (i, k, _dis, _point)
                     
        return 0
    
    
    def prepare_Clusters(self):
        start = time.time()
        class_id = 0
        
        itr = 0
        for dp_kern in self.KernelPoints:
            itr+=1
            if self.prop_step <= itr :
                continue
                
            dp_core = self.DataPoints[dp_kern[0]]
           
            #remember no strong points & weak points in Tirann
            #all points with at least one refernce are considered  (ignore noises)
            if len(dp_core.reference) > 0 and len(dp_core.referred_by) >= len(dp_core.refer_to):
                
                class_id += 1
                dp_core.visited = True
                dp_core.class_id = class_id
                self.ClassPoints[class_id] = Node(class_id, parent=None)
                max_class = -1
                weight_map = {}
                #Class_Points[class_id] = new TreeCls::Node(class_id)

                for pos2 in dp_core.reference :
                    # if DataPoints[*pos2].visited &&  visited was tested not to affect on results, so you can ommit it
                    if self.DataPoints[pos2[0]].class_id > 0  and len(self.DataPoints[pos2[0]].referred_by) >= len(self.DataPoints[pos2[0]].refer_to):
                   
                        
                        # this condition is a must, as some points may be visited but not classified yet
                        # maa we may neglect is noise as long as it is in our refernce points
                                  
                        _cls = self.DataPoints[pos2[0]].class_id
                        _class_id = self.ClassPoints[_cls].root.name
                        #_class_id = _cls
                                                                      
                        
                        if _class_id not in weight_map.keys():
                             weight_map[_class_id] = 1
                        else:
                            weight_map[_class_id] += 1
                        
                        
                    elif self.DataPoints[pos2[0]].visited == False:
                        self.DataPoints[pos2[0]].visited = True  # this point is visited but not classified yet
                        
               
               
                while len(weight_map) > 0 :
                    #weight_no = self.getValue(dic=weight_map, what='max', who='value') # no need to it in DenMune
                    max_class = self.getValue(dic=weight_map, what='max', who='value', other=True)
                 
                    if max_class != -1 and  max_class != class_id:
                        self.ClassPoints[max_class].parent = self.ClassPoints[class_id]

                    del weight_map[max_class]
                   
                           
        for i  in range (self.dp_count):
            clsid = self.DataPoints[i].class_id
            clsroot = self.ClassPoints[clsid].root.name
            self.DataPoints[i].class_id = clsroot 
            
        end = time.time() 
        #print ('prepare Clusters took:', end-start)
         
        return 0
    
    
    def attach_Points(self):
        
        start = time.time()
        olditr = 0
        newitr = -1
        while olditr != newitr:
            newitr = olditr
            olditr = 0
            
            for pos in self.KernelPoints:
                if self.DataPoints[pos[0]].class_id == 0 :
                    self.DataPoints[pos[0]].class_id = self.attach_StrongPoint(pos[0])
                    olditr += 1

        olditr = 0
        newitr = -1
        while olditr != newitr:
            newitr = olditr
            olditr = 0
            
            for pos in self.KernelPoints:
                if self.DataPoints[pos[0]].class_id == 0: 
                    self.DataPoints[pos[0]].class_id = self.attach_WeakPoint(pos[0])
                    olditr += 1
       
        end = time.time()      
        #print ('Attach points tooks:', end-start)
    
    def attach_StrongPoint(self, point_id):
        weight_map = {}
        max_class = 0 # max_class in attach point = 0 , thus if a point faild to merge with any cluster, it has one more time 
            #to merge in attach weak point
        dp_core = self.DataPoints[point_id]
        if len(dp_core.reference) != 0:
            dp_core.visited = True
        
            for pos2 in dp_core.reference:

                
                if self.DataPoints[pos2[0]].visited == True and len(self.DataPoints[pos2[0]].referred_by) >= len(self.DataPoints[pos2[0]].refer_to):
                    
                    clsid = self.DataPoints[pos2[0]].class_id
                    clsroot = self.ClassPoints[clsid].root.name
                    self.DataPoints[pos2[0]].class_id = clsroot 


                    if clsroot not in weight_map.keys():
                         weight_map[clsroot] = 1
                    else:
                        weight_map[clsroot] += 1


            if len (weight_map) != 0:
                weight_map = dict(sorted(weight_map.items()))
                max_class = self.getValue(dic=weight_map, what='max', who='value', other=True)
                
        return max_class # this will return get_Root(max_class) as we computed earlier _class_id = get_Root(_cls) 

    
    def attach_WeakPoint(self, point_id):
              
        weight_map = {}
        max_class = -1 # max_class in attach weak point = -1 , thus if a point faild to merge with any cluster it is a noise
        
        dp_core = self.DataPoints[point_id]
        if len(dp_core.reference) != 0:
            dp_core.visited = True
        
            for pos2 in dp_core.reference:

                
                if self.DataPoints[pos2[0]].visited == True :
                    
                    clsid = self.DataPoints[pos2[0]].class_id
                    clsroot = self.ClassPoints[clsid].root.name
                    self.DataPoints[pos2[0]].class_id = clsroot 

                    if clsroot not in weight_map.keys():
                         weight_map[clsroot] = 1
                    else:
                        weight_map[clsroot] += 1

            if len (weight_map) != 0:
                weight_map = dict(sorted(weight_map.items()))
                max_class = self.getValue(dic=weight_map, what='max', who='value', other=True)

        return max_class # this will return get_Root(max_class) as we computed earlier _class_id = get_Root(_cls)

                   

    def fit_predict(self, data_labels=None):
        solution_file = 'solution.txt'
        GroundTruth = False
        
        if  os.path.isfile(solution_file):
            os.remove(solution_file)
                
        pred_list = []
        for dp in self.DataPoints:
            pred_list.append(dp.class_id)
            
        with open(solution_file, 'w') as f:
            f.writelines("%s\n" % pred for pred in pred_list)
            
        file_labels = data_labels
        if file_labels is not None:
            # Avoid the Case of Cham 01==>Cham_04 which have no groundtruth
            GroundTruth = True
            labels_true = data_labels.astype(np.int64)
            
        
        labels_pred = []
        labels_pred = genfromtxt(solution_file)
        
        if self.prop_step: # any step greater than zero will be evaluated to true
            #labels_pred = np.array(labels_pred)
            #labels_pred = labels_pred.astype(np.int64)
            dummy = 0 # nothing to do
        elif  GroundTruth:
            labels_pred = self.match_Labels(labels_pred, labels_true)
        else: # Case of Cham 01==>Cham_04 which have no groundtruth
            #labels_pred = np.array(labels_pred)
            #labels_pred = labels_pred.astype(np.int64)
            dummy = 0 # nothing to do
            
        
        #if self.show__:
        #    self.plot_clusters(self.data, labels_pred, self.show_noise)
        
        # self.preplot_Clusters(labels_pred, ground=False)
        
        #if self.verpose == 1 or self.verpose == 3 :
        #    self.show_Analyzer(root="DenMune Clustering")
       
        return  labels_pred 
       
        
            
    def match_Labels(self, labels_pred, labels_true):
       
        if 0 in labels_true: # let us start with class number 1 since class 0 in denmune is noise
            labels_true += 1
                       
        list_pred = labels_pred.tolist()
        pred_set = set(list_pred) 

        index = []
        x = 1
        old_item = labels_true[0]
        old_x = 0

        for item in labels_true:

            if item != old_item:
                count = x - old_x
                index.append([old_x, old_item, count])
                old_item = item
                old_x = x
            x+= 1    

        ln = len(labels_true)
        count = x - old_x
        index.append([old_x, old_item, count])
        index[0][2] = index[0][2] -1

        index.sort(key=lambda x: x[2], reverse=True)

        lebeled = []
        for n in range (len(index)):
            newval = index[n][1]
            max_class = max(set(list_pred), key = list_pred[index[n][0]:index[n][0]+index[n][2]-1].count)
            if max_class not in lebeled:
                list_pred = [newval if x==max_class else x for x in list_pred]
                lebeled.append(newval)

        list_pred = np.array(list_pred)
        list_pred = list_pred.astype(np.int64)
        
        #with open('solumatch', 'w') as f:
        #    f.writelines("%s\n" % pred for pred in list_pred)
            
        return list_pred
    
    
    def validate_Clusters(self, labels_true, labels_pred):
       
        if 0 in labels_true: # let us start with class number 1 since, in denmune class 0 is noise
            labels_true+=1
        self.analyzer["n_clusters"]["actual"] = len(np.unique(labels_true))
        
        # Score the clustering
        from sklearn.metrics.cluster import adjusted_mutual_info_score #2010
        from sklearn.metrics.cluster import adjusted_rand_score # 1985
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score

        #from sklearn.metrics import davies_bouldin_score 
        # #1975 - 2001    ## no ground truth   ##Values closer to zero indicate a better partition.

        from sklearn.metrics import pairwise_distances # for calinski_harabasz_score
        ## also known as the Variance Ratio Criterion - can be used to evaluate the model, 
        ## where a higher Calinski-Harabasz score relates to a model with better defined clusters.

        from sklearn import metrics # for homogeneity, completeness, fowlkes
        ##  homogeneity: each cluster contains only members of a single class.
        ## completeness: all members of a given class are assigned to the same cluster.
        #v-measure the harmonic mean of homogeneity and completeness called V-measure 2007

        acc = metrics.accuracy_score(labels_true, labels_pred, normalize=False)

        #mi = metrics.mutual_info_score(labels_true, labels_pred)
        #print("mutual_info_score: %f." %  mi)

        nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
        #print("normalized_mutual_info_score: %f." % nmi)

        ami = adjusted_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
            #print("Adjusted_mutual_info_score: %f." %  adj_nmi)

        homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
        #print("homogeneity_score: %f." % homogeneity_score)

        completeness = metrics.completeness_score(labels_true, labels_pred)
        #print("completeness_score: %f." % completeness_score)

        f1_weight = metrics.f1_score(labels_true, labels_pred, average='weighted')
        #f1_micro = metrics.f1_score(labels_true, labels_pred, average='micro')
        #f1_macro = metrics.f1_score(labels_true, labels_pred, average='macro')
        #print("f1_score: %f." % f1_score)
        
        
        ari = adjusted_rand_score(labels_true, labels_pred)
        #print("adjusted_rand_score: %f." % adj_rand)

        f1 =  f1_weight
        
        validity = {"ACC" : acc,
                    "F1"  : f1,
                    "NMI" : nmi,
                    "AMI" : ami,
                    "ARI" : ari,
                    "homogeneity"  : homogeneity,
                    "completeness" : completeness
                   }
        

       # val = [acc, f1, nmi, ami, ari, homogeneity, completeness]
        self.analyzer["validity"]  =  validity
        self.analyzer["n_points"]["weak"]["succeeded to merge"] = self.analyzer["n_points"]["weak"]["all"] - self.analyzer["n_points"]["noise"]["type-2"]
        self.analyzer["n_points"]["weak"]["failed to merge"] = self.analyzer["n_points"]["noise"]["type-2"]   
                                    
        if self.verpose:
            self.show_Analyzer(self.analyzer, root="DenMune Analyzer")
         

        return validity  
    
        
        
    def preplot_Clusters(self, labels, ground=False):
        labels = np.array(labels, dtype=np.int64)
             
        noise_1 = list(labels).count(-1)
        self.analyzer["n_points"]["noise"]["type-1"] = noise_1
        
        noise_2 = list(labels).count(0)
        self.analyzer["n_points"]["noise"]["type-2"]  = noise_2
        
        #if self.show_noise and self.verpose and not ground:
        #    print('There are', noise_1 , 'outlier point(s) in black (noise of type-1)', 'represent',
        #          f"{ noise_1 /len(labels):.0%}", 'of total points')
        #    print('There are', noise_2 , 'weak point(s) in light grey (noise of type-2)', 'represent', f"{noise_2/len(labels):.0%}", 'of total points')
        unique_labels = np.unique(labels)
        num_of_clusters = len(unique_labels)
                  
        fake_clusters = 0 #otlier = -1 and weak points that fail to merge (noise) = 0
        
        i= 1
        for n in (unique_labels):
            if n > 0:
                labels = np.where(labels==n, i, labels) 
                i +=1
            else:
                fake_clusters+=1
               
        self.analyzer["n_clusters"]["detected"]=   num_of_clusters - fake_clusters  
        
        #if self.verpose and not ground:
        #    print('DenMune detected', self.analyzer["n_clusters"], 'clusters', '\n')
        palette = sns.color_palette( 'deep', np.unique(labels).max()+2) #deep, dark, bright, muted, pastel, colorblind
        
        if ground:
            if 0 in labels: # let us start with class number 1 to match with results in denmune
                labels +=1
                
        return  labels       
                
      
    def plot_clusters(self, labels, show_noise=False, ground=False):
        
        #labels = np.array(labels, dtype=np.int64)
        labels = self.preplot_Clusters(labels, ground=ground)
        palette = sns.color_palette( 'deep', np.unique(labels).max()+2) #deep, dark, bright, muted, pastel, colorblind
        
      
        if self.prop_step:
            data2 = []
            colors2 = []
            colors = [palette[x] if x > 0 else ( (0.0, 0.0, 0.0) if x == -1 else (0.0, 0.0, 0.0)) for x in labels]
            v = 0
            for c in colors:
                if (c[0]+c[1] + c[2]) > 0.0:  # outlier :: keep it in black 
                    colors2.append((c[0], c[1] , c[2], 1.0))
                    data2.append((self.data[v][0],self.data[v][1]))
                v+=1 
            data2 = np.array(data2) 
     
        elif ground:
            if 0 in labels: # let us start with class number 1 to match with results in denmune
                labels +=1
            colors = [palette[x]  for x in labels]
        else:
            if show_noise == False:
                colors = [palette[x] if x > 0 else (1.0, 1.0, 1.0) for x in labels] # noise points wont be printed due to x > 0 , else (1.0, 1.0, 1.0)
            else :
                colors = [palette[x] if x > 0 else ( (0.0, 0.0, 0.0) if x == -1 else (0.9, 0.9, 0.9)) for x in labels] # noise points wont be printed due to x > 0 , else (1.0, 1.0, 1.0)

        plt.figure(figsize=(12, 8))
        
        if self.prop_step:
            plt.scatter(data2.T[0], data2.T[1], c=colors2, **plot_kwds, marker='o')
        else:
             plt.scatter(self.data.T[0], self.data.T[1], c=colors2, **plot_kwds, marker='o')
       
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
       
        #plt.savefig('mydata/foo.png')
        plt.show()
        #plt.clf()    # this is a must to clear figures if you plot continously

        return 0
    
    
    def generate_tsne(self, data, file_2d, d):
        
        
        dim_two = TSNE(n_components=d, random_state=1971, init='random').fit_transform(data)

        mystr = ""
        data_len = len (dim_two)
        for i in range(data_len):
            for n in range(d):
                mystr += str(round(dim_two[i][n],6))
                if (n  < d-1):mystr += '\t'
                if (n  == d-1): mystr += '\n'

        text_file = open(file_2d, "w")
        text_file.write(mystr)
        text_file.close()

        return 0
    
    def show_Analyzer(self, mydic=None, root="DenMune"):
        
        if mydic is None:
            mydic=self.analyzer
        
        
        tree = tr()
        tree.create_node(root, "root")

        def creat_TreefromDict(self, tree, mydict, key, parent):
            if type(mydict[key]) is not dict:
                val = key + ': ' + str(round(mydict[key],3))
                tree.create_node(val, key,  parent=parent)

        for d in mydic:
            if type(mydic[d]) is not dict:
                creat_TreefromDict(self, tree, mydic, d, parent='root')
            else:
                tree.create_node(d, d,  parent="root")
                subdic =mydic[d]
                for v in subdic:
                    if type(subdic[v]) is not dict:
                        creat_TreefromDict(self, tree, subdic, v, parent=d)
                    else:
                        tree.create_node(v, v,  parent=d)
                        subsubdic =subdic[v]
                        for z in subsubdic:
                            creat_TreefromDict(self, tree, subsubdic, z, parent=v)

        tree.show()    
    
    
    def flattenPred(self, labels_pred, labels_true): #this function for future use
    
        list_pred = labels_pred.tolist()
        pred_set = set(list_pred) 

        index = []
        x = 0
        old_item = labels_true[0]
        for item in labels_true:
            x+= 1
            if item != old_item:
                old_item = item
                index.append(x)

        ln = len(labels_true)
        res = []
        old_idx = 0
        for idx in index:
            mx = max(set(list_pred), key = list_pred[old_idx:idx].count) 
            old_idx = idx
            res.append(mx)
        mx = max(set(list_pred), key = list_pred[idx:ln].count) 
        res.append(mx)

        index.insert(0, 0)
        n = 0
        for item in res:
            newval = labels_true[index[n]]
            list_pred = [newval if x==item else x for x in list_pred]
            n+=1

        list_pred = np.array(list_pred)
        list_pred = list_pred.astype(np.int64)
        return list_pred
        
   

    def labels_Patterns (self, mylist): # this function for future use
        #mylist = [1, 26, 27, 51, 52, 77, 78, 103, 105]
        mylist = [str(i) for i in mylist]
        if  int(max (mylist)) <= 103:

            x = 0
            for item  in mylist:
                if item <= '25':
                    mylist[x] = chr(int(item)+66)
                elif item <= '51':
                    mylist[x] = 'A' + chr(int(item)+39)
                elif item <= '77':
                    mylist[x] = 'B' + chr(int(item)+13)
                else:
                    mylist[x] = 'C' + chr(int(item)-13)

                x += 1 
            return mylist    

        else:
            return 'Max classes numbers are 103 classes'
