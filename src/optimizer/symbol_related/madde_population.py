from .population import Population
import numpy as np

'''population only for MadDE'''
class MadDE_Population(Population):
    def __init__(self, dim, pop_size, min_x, max_x, max_fes, problem):
        super().__init__(dim, pop_size, min_x, max_x, max_fes, problem)

    def reset(self):
        self.index=np.arange(self.pop_size)
        return super().reset()

    def update(self,next_position, new_cost,filter_survive=False):
        self.pre_cost=self.c_cost
        self.pre_position=self.current_position
        self.pre_gbest=self.gbest_cost

        self.before_select_pos=next_position

        # new_cost=self.get_costs(next_position)
        if filter_survive:
            surv_filter=new_cost<=self.c_cost
            next_position=np.where(surv_filter[:,None],next_position,self.current_position)
            new_cost=np.where(surv_filter,new_cost,self.c_cost)

       
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
            # self.gbest_index=new_cbest_index
            self.stag_count=0
        else:
            self.stag_count+=1

        self.cbest_cost=new_cbest_val
        self.cbest_position=next_position[new_cbest_index]

        new_cworst_val=np.max(new_cost)
        if new_cworst_val>self.gworst_cost:
            self.gworst_cost=new_cworst_val
            gworst_index=np.argmax(new_cost)
            self.gworst_position=next_position[gworst_index]
        
        # deprecated
        self.dx=(self.c_cost-self.pre_cost)[:,None]/(self.current_position-self.pre_position+1e-5)
        self.dx=np.where(np.isnan(self.dx),np.zeros_like(self.current_position),self.dx)


    def update2(self,next_position,new_cost):
        self.pre_cost=self.c_cost
        self.pre_position=self.current_position
        self.pre_gbest=self.gbest_cost

        self.before_select_pos=next_position
        

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
            # self.gbest_index=new_cbest_index
            self.stag_count=0
        else:
            self.stag_count+=1

        self.cbest_cost=new_cbest_val
        self.cbest_position=next_position[new_cbest_index]

        new_cworst_val=np.max(new_cost)
        if new_cworst_val>self.gworst_cost:
            self.gworst_cost=new_cworst_val
            gworst_index=np.argmax(new_cost)
            self.gworst_position=next_position[gworst_index]
        
        self.dx=(self.c_cost-self.pre_cost)[:,None]/(self.current_position-self.pre_position+1e-5)
        self.dx=np.where(np.isnan(self.dx),np.zeros_like(self.current_position),self.dx)

        

    def reset_order(self,index):
        self.current_position=self.current_position[index]
        self.c_cost=self.c_cost[index]
        self.dx=self.dx[index]
        self.pbest_cost=self.pbest_cost[index]
        self.pbest_position=self.pbest_position[index]
        self.index=self.index[index]
    
    def reset_popsize(self,NP):
        self.current_position=self.current_position[:NP]
        self.c_cost=self.c_cost[:NP]
        self.pbest_cost=self.pbest_cost[:NP]
        self.pbest_position=self.pbest_position[:NP]
        self.index=self.index[:NP]
        self.pop_size=NP