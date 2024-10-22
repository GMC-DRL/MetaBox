import numpy as np
from scipy.spatial.distance import cdist
import copy

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2,axis=-1))
    

class Population(object):
    def __init__(self,dim,pop_size,min_x,max_x,max_fes,problem):
        self.dim=dim
        self.pop_size=pop_size
        self.min_x=min_x
        self.max_x=max_x
        self.max_fes=max_fes
        self.problem=problem
        self.max_dist=((max_x-min_x)**2*dim)**0.5
        self.cur_fes=0

    # calculate costs of solutions
    def get_costs(self,position):
        ps=position.shape[0]
        self.cur_fes+=ps
        if self.problem.optimum is None:
            cost=self.problem.eval(position)
        else:
            cost=self.problem.eval(position) - self.problem.optimum
        
        return cost
    
    def reset(self,init_pop=None,init_y=None,need_his=True):
        np.random.seed()
        # init fes and stag_count
        if init_y is not None:
            self.cur_fes+=init_y.shape[0]
        else:
            self.cur_fes=0
        self.stag_count=0
        
        # init population
        if init_pop is None:
        # randomly generate the position and velocity
            rand_pos=np.random.uniform(low=-self.max_x,high=self.max_x,size=(self.pop_size,self.dim))
        else:
            rand_pos=init_pop

        self.current_position=rand_pos.copy()
        self.dx=np.zeros_like(rand_pos)
        self.delta_x=np.zeros_like(rand_pos)
        
        # get the initial cost
        if init_y is None:
            self.c_cost = self.get_costs(self.current_position) # ps
        else:
            self.c_cost = init_y

        # init pbest related
        self.pbest_position=rand_pos.copy()
        self.pbest_cost=self.c_cost.copy()

        # find out the gbest_val
        self.gbest_cost = np.min(self.c_cost)
        gbest_index = np.argmin(self.c_cost)
        self.gbest_position=rand_pos[gbest_index]
        

        # init cbest related
        self.cbest_cost=self.gbest_cost
        self.cbest_position=self.gbest_position
        self.cbest_index=gbest_index
        
        # init gworst related
        self.gworst_cost=np.max(self.c_cost)
        gworst_index=np.argmax(self.c_cost)
        self.gworst_position=rand_pos[gworst_index]

        # record
        self.init_cost=np.min(self.c_cost)
        self.pre_position=self.current_position
        self.pre_cost=self.c_cost
        self.pre_gbest=self.gbest_cost
    
        
    def update(self,next_position, filter_survive=False):
        self.pre_cost=self.c_cost
        self.pre_position=copy.deepcopy(self.current_position)
        # self.pre_gbest=self.gbest_cost

        self.before_select_pos=next_position

        new_cost=self.get_costs(next_position)
        
        if filter_survive:
            surv_filter=new_cost<=self.c_cost
            next_position=np.where(surv_filter[:,None],next_position,self.current_position)
            new_cost=np.where(surv_filter,new_cost,self.c_cost)
       
        filters = new_cost < self.pbest_cost
        
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        
        self.current_position=next_position
        self.c_cost=new_cost
        self.pbest_position=np.where(np.expand_dims(filters,axis=-1),
                                                        next_position,
                                                        self.pbest_position)
        self.pbest_cost=np.where(filters,
                                new_cost,
                                self.pbest_cost)
        if new_cbest_val<self.gbest_cost:
            self.gbest_cost=new_cbest_val
            self.gbest_position=self.current_position[new_cbest_index]
            # gbest_index=new_cbest_index
            self.stag_count=0
        else:
            self.stag_count+=1

        self.cbest_cost=new_cbest_val
        self.cbest_position=next_position[new_cbest_index]
        self.cbest_index=new_cbest_index

        new_cworst_val=np.max(new_cost)
        if new_cworst_val>self.gworst_cost:
            self.gworst_cost=new_cworst_val
            gworst_index=np.argmax(new_cost)
            self.gworst_position=next_position[gworst_index]
        
        self.dx=(self.c_cost-self.pre_cost)[:,None]/(self.current_position-self.pre_position+1e-5)
        self.dx=np.where(np.isnan(self.dx),np.zeros_like(self.current_position),self.dx)

        self.delta_x=self.current_position-self.pre_position

    
    def update_cmaes(self,next_position,next_y):
        self.pre_cost=self.c_cost
        self.pre_position=self.current_position
        # self.pre_gbest=self.c_cost

        new_cost=next_y

        # update particles
        filters = new_cost < self.pbest_cost
        
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        

        self.current_position=next_position
        self.c_cost=new_cost
        self.pbest_position=np.where(np.expand_dims(filters,axis=-1),
                                                        next_position,
                                                        self.pbest_position)
        self.pbest_cost=np.where(filters,
                                new_cost,
                                self.pbest_cost)
        if new_cbest_val<self.gbest_cost:
            self.gbest_cost=new_cbest_val
            self.gbest_position=self.current_position[new_cbest_index]
            self.stag_count=0
        else:
            self.stag_count+=1

        self.cbest_cost=new_cbest_val
        self.cbest_position=next_position[new_cbest_index]
        self.cbest_index=new_cbest_index

        new_cworst_val=np.max(new_cost)
        if new_cworst_val>self.gworst_cost:
            self.gworst_cost=new_cworst_val
            gworst_index=np.argmax(new_cost)
            self.gworst_position=next_position[gworst_index]
        

    def feature_encoding(self):
        assert self.gbest_cost!=self.gworst_cost,f'gbest == gworst!!,{self.gbest_cost}'
        fea_1=(self.c_cost-self.gbest_cost)/(self.gworst_cost-self.gbest_cost+1e-8)
        fea_1=np.mean(fea_1)
        
        fea_2=calculate_mean_distance(self.current_position)/self.max_dist

        fit=np.zeros_like(self.c_cost)
        fit[:self.pop_size//2]=self.gworst_cost
        fit[self.pop_size//2:]=self.gbest_cost
        maxstd=np.std(fit)
        fea_3=np.std(self.c_cost)/(maxstd+1e-8)

        fea_4=(self.max_fes-self.cur_fes)/self.max_fes

        fea_5=self.stag_count/(self.max_fes//self.pop_size)
        
        fea_6=dist(self.current_position,self.cbest_position[None,:])/self.max_dist
        fea_6=np.mean(fea_6)

        fea_7=(self.c_cost-self.cbest_cost)/(self.gworst_cost-self.gbest_cost+1e-8)
        fea_7=np.mean(fea_7)

        fea_8=dist(self.current_position,self.gbest_position[None,:])/self.max_dist
        fea_8=np.mean(fea_8)

        fea_9=0
        if self.gbest_cost<self.pre_gbest:
            fea_9=1
        
        feature=np.array([fea_1,fea_2,fea_3,fea_4,fea_5,fea_6,fea_7,fea_8,fea_9])
        

        assert not np.any(np.isnan(feature)),f'feature has nan!!,{feature}'
        return feature
    
def calculate_mean_distance(population):
    distances = cdist(population, population, metric='euclidean')
    
    np.fill_diagonal(distances, 0)
    
    mean_distance = np.mean(distances)
    
    return mean_distance
