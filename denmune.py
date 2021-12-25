import numpy as np
import ngtpy
from numpy import genfromtxt
from anytree import Node, RenderTree
import operator
import os.path
import time
from functions import Functions as func


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
    
         
    def __init__ (self, dataset, k_nearest=1, data_path='', verpose=True, show_plot=False):
        
                     
        file_2d = data_path + dataset + '-2d.txt'
        #if os.path.isfile(file_2d):
        #print('dataset has been allready reduced to 2-d' )
        if  not os.path.isfile(file_2d):
            start = time.time()
            func.generate_tsne(dataset, 2)
            end = time.time()
            if verpose:
                print('using t-SNE', dataset, ' dataset has been reduced to 2-d in ', end-start, ' seconds')
                
            
                
        #if verpose:
        #    print('k= ', k_nearest)  
        
        data = genfromtxt(file_2d , delimiter='\t')
        start_time = time.time()
        
        self.alg_name = 'denmune'
        self.show_plot = show_plot
        self.verpose = verpose
        self.data_path = data_path     
        self.dataset = dataset
        self.data = data
        self.dp_count = self.data.shape[0] 
        self.dp_dim = self.data.shape[1]
        self.k_nearest = k_nearest
        self.dp_dis = []
               
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
        #self.output_Clusters()
        
        end_time = time.time()  
        #if verpose:
        #    print('DenMune took' , end_time-start_time, ' seconds' )
            
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
            if self.verpose:
                print ('using NGT, Proximity matrix has been calculated  in: ', end-start, ' seconds')
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
                                                
                        
        for i  in range (self.dp_count) :
            self.DataPoints[i].referred_by = self.sort_Tuple(self.DataPoints[i].referred_by, reverse=False)
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
        
        for dp_kern in self.KernelPoints:
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

                   

    def output_Clusters(self):
        solution_file = 'solution.txt'
        #validity_idx = 2   ==> so we are measuring best F1-score
        
        if  os.path.isfile(solution_file):
            os.remove(solution_file)
                
        pred_list = []
        for dp in self.DataPoints:
            pred_list.append(dp.class_id)
            
        with open(solution_file, 'w') as f:
            f.writelines("%s\n" % pred for pred in pred_list)
            
        file_labels = self.dataset +'-gt.txt' #dataset + '-gt.txt'
        labels_true = genfromtxt(self.data_path+file_labels)
        labels_true = labels_true.astype(np.int64)
        
        labels_pred = []
        labels_pred = genfromtxt(solution_file)
        labels_pred = func.match_Labels(labels_pred, labels_true) 
        
        if self.show_plot or self.verpose:
            func.plot_clusters(data=self.data, labels=labels_pred, alg_name=self.alg_name, dp_name=self.dataset)
        
           
        return  labels_true, labels_pred 
    
    
    def validate_Clusters(self, labels_true, labels_pred):
       
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
        val = ['0', acc, f1, nmi, ami, ari, homogeneity, completeness, '0' ]
         
        return val    
    