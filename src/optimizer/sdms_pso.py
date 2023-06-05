import torch
import numpy as np
from scipy.optimize import minimize
from optimizer.basic_optimizer import Basic_Optimizer


class sDMS_PSO(Basic_Optimizer):
    def __init__(self,config):
        super().__init__(config)
        self.__w,self.__c1,self.__c2=0.729,1.49445,1.49445
        self.__m,self.__R,self.__LP,self.__LA,self.__L,self.__L_FEs=3,10,10,8,100,200
        self.__NP=99
        self.__w_decay=True
        self.config=config

        self.__dim,self.__max_fes=config.dim,config.maxFEs

        self.__n_swarm=self.__NP//self.__m
        self.__fes=0
        
        self.__group_index=np.zeros(self.__NP,dtype=np.int8)
        self.__per_no_improve=np.zeros(self.__NP)
        self.__lbest_no_improve=np.zeros(self.__n_swarm)
        
        assert self.__NP%self.__m==0, 'population cannot be update averagely'
        for sub_swarm in range(self.__n_swarm):
            if sub_swarm!=self.__n_swarm-1:
                self.__group_index[sub_swarm*self.__m:(sub_swarm+1)*self.__m]=sub_swarm
            else:
                self.__group_index[sub_swarm*self.__m:]=sub_swarm
        
        self.__parameter_set=[]
        self.__success_num=np.zeros((self.__n_swarm))
        self.log_interval = config.log_interval

    def __get_costs(self,problem,position):
        ps=position.shape[0]
        self.__fes+=ps
        if problem.optimum is None:
            cost=problem.eval(position)
        else:
            cost=problem.eval(position) - problem.optimum
        return cost

    def __initilize(self,problem):
        rand_pos=np.random.uniform(low=problem.lb,high=problem.ub,size=(self.__NP,self.__dim))
        self.__max_velocity=0.1*(problem.ub-problem.lb)
        rand_vel = np.random.uniform(low=-self.__max_velocity,high=self.__max_velocity,size=(self.__NP,self.__dim))
        # rand_vel = torch.zeros(size=(self.__batch_size, self.__ps, self.__dim),dtype=torch.float32).to(self.__cuda)

        c_cost = self.__get_costs(problem,rand_pos) # ps

        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position=rand_pos[gbest_index]
        self.__max_cost=np.min(c_cost)
        # print("rand_pos.shape:{}".format(rand_pos.shape))

        self.__particles={'current_position': rand_pos.copy(), #  ps, dim
                          'c_cost': c_cost.copy(), #  ps
                          'pbest_position': rand_pos.copy(), # ps, dim
                          'pbest': c_cost.copy(), #  ps
                          'gbest_position':gbest_position.copy(), # dim
                          'gbest_val':gbest_val,  # 1
                          'velocity': rand_vel.copy(), # ps,dim
                          }
        self.__particles['lbest_cost']=np.zeros(self.__n_swarm)
        self.__particles['lbest_position']=np.zeros((self.__n_swarm,self.__dim))
        
    def __reset(self,problem):
        if self.__w_decay:
            self.__w=0.9
        self.__gen=0
        self.__fes-=self.__fes
        self.__per_no_improve-=self.__per_no_improve
        self.__lbest_no_improve-=self.__lbest_no_improve
        self.__learning_period=True
        self.__parameter_set=[]
        self.__success_num=np.zeros((self.__n_swarm))
        self.__cur_mode='ls'
        self.__initilize(problem)
        self.__random_regroup()
        self.__update_lbest(init=True)
        self.__fes_eval=np.zeros_like(self.__fes)
        self.log_index = 1
        self.cost = [self.__particles['gbest_val']]
        return None

    def __random_regroup(self):
        regroup_index=torch.randperm(n=self.__NP)
        self.__lbest_no_improve-=self.__lbest_no_improve
        self.__regroup_index=regroup_index
        self.__particles['current_position']=self.__particles['current_position'][regroup_index] # bs, ps, dim
        self.__particles['c_cost']= self.__particles['c_cost'][regroup_index] # bs, ps
        self.__particles['pbest_position']=self.__particles['pbest_position'][regroup_index] # bs, ps, dim
        self.__particles['pbest']= self.__particles['pbest'][regroup_index] # bs, ps
        self.__particles['velocity']=self.__particles['velocity'][regroup_index]
        self.__per_no_improve=self.__per_no_improve[regroup_index]
        
    def __update_lbest(self,init=False):
        if init:
            grouped_pbest=self.__particles['pbest'].reshape(self.__n_swarm,self.__m)
            grouped_pbest_pos=self.__particles['pbest_position'].reshape(self.__n_swarm,self.__m,self.__dim)
            # print(f'pbest_feature.shape:{self.__pbest_feature.shape}')
            
            self.__particles['lbest_cost']=np.min(grouped_pbest,axis=-1)
            index=np.argmin(grouped_pbest,axis=-1)
            self.__lbest_index=index+np.arange(self.__n_swarm)*self.__m   # n_swarm,
            self.__particles['lbest_position']=grouped_pbest_pos[range(self.__n_swarm),index]
            
        else:
            # self.__lbest_feature[:,:,4]+=1./self.__max_step
            # self.__lbest_feature[:,:,3]=((self.__max_fes-self.fes)/self.__max_fes).unsqueeze(dim=1)
            # self.__lbest_no_improve+=1
            grouped_pbest=self.__particles['pbest'].reshape(self.__n_swarm,self.__m)
            grouped_pbest_pos=self.__particles['pbest_position'].reshape(self.__n_swarm,self.__m,self.__dim)
            lbest_cur=np.min(grouped_pbest,axis=-1)
            index=np.argmin(grouped_pbest,axis=-1)
            
            lbest_pos_cur=grouped_pbest_pos[range(self.__n_swarm),index]
            # grouped_pbest_fea=self.__pbest_feature.reshape(self.__batch_size,self.__n_swarm,self.__m,self.__node_dim)
            # cur_lbest_feature=grouped_pbest_fea[batch_index,swarm_index,index]

            filter_lbest=lbest_cur<self.__particles['lbest_cost']
            self.__lbest_index=np.where(filter_lbest,index+np.arange(self.__n_swarm)*self.__m,self.__lbest_index)

            # update success_num
            
            success=np.sum(grouped_pbest<self.__particles['lbest_cost'][:,None],axis=-1)
            
            self.__success_num+=success
            
            self.__particles['lbest_cost']=np.where(filter_lbest,lbest_cur,self.__particles['lbest_cost'])
            self.__particles['lbest_position']=np.where(filter_lbest[:,None],lbest_pos_cur,self.__particles['lbest_position'])
            self.__lbest_no_improve=np.where(filter_lbest,np.zeros_like(self.__lbest_no_improve),self.__lbest_no_improve+1)

    def __get_iwt(self):
        if len(self.__parameter_set)<self.__LA or np.sum(self.__success_num)<=self.__LP:
            # iwt=0.5*np.random.uniform(low=1,high=self.__n_swarm)+0.4
            self.__iwt=0.5*np.random.rand(self.__n_swarm)+0.4

        else:
            self.__iwt=np.random.normal(loc=np.median(self.__parameter_set),scale=0.1,size=(self.__n_swarm,))

    def __update(self,problem):
        rand1=np.random.rand(self.__NP,1)
        rand2=np.random.rand(self.__NP,1)
        c1=self.__c1
        c2=self.__c2
        v_pbest=rand1*(self.__particles['pbest_position']-self.__particles['current_position'])
        if self.__cur_mode=='ls':
            v_lbest=rand2*(self.__particles['lbest_position'][self.__group_index]-self.__particles['current_position'])
            self.__get_iwt()
            new_velocity=self.__iwt[self.__group_index][:,None]*self.__particles['velocity']+c1*v_pbest+c2*v_lbest
        elif self.__cur_mode=='gs':
            v_gbest=rand2*(self.__particles['gbest_position'][None,:]-self.__particles['current_position'])
            new_velocity=self.__w*self.__particles['velocity']+c1*v_pbest+c2*v_gbest
        new_velocity=np.clip(new_velocity,-self.__max_velocity,self.__max_velocity)
        raw_position=self.__particles['current_position']+new_velocity
        new_position = np.clip(raw_position,problem.lb,problem.ub)
        new_cost=self.__get_costs(problem,new_position)
        filters = new_cost < self.__particles['pbest']
        # new_cbest_val,new_cbest_index=torch.min(new_cost,dim=1)
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)

        filters_best_val=new_cbest_val<self.__particles['gbest_val']
        # update particles
        new_particles = {'current_position': new_position, # bs, ps, dim
                            'c_cost': new_cost, # bs, ps
                            'pbest_position': np.where(np.expand_dims(filters,axis=-1),
                                                        new_position,
                                                        self.__particles['pbest_position']),
                            'pbest': np.where(filters,
                                                new_cost,
                                                self.__particles['pbest']),
                            'velocity': new_velocity,
                            'gbest_val':np.where(filters_best_val,
                                                    new_cbest_val,
                                                    self.__particles['gbest_val']),
                            'gbest_position':np.where(np.expand_dims(filters_best_val,axis=-1),
                                                        new_position[new_cbest_index],
                                                        self.__particles['gbest_position']),
                            'lbest_position':self.__particles['lbest_position'],
                            'lbest_cost':self.__particles['lbest_cost']
                            }
        self.__particles=new_particles

        if self.__fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__particles['gbest_val'])

        if self.__cur_mode=='ls':
            self.__update_lbest()
        
    def __update_parameter_set(self):
        max_success_index=np.argmax(self.__success_num)
        if len(self.__parameter_set)<self.__LA:
            self.__parameter_set.append(self.__iwt[max_success_index])
        else:
            del self.__parameter_set[0]
            self.__parameter_set.append(self.__iwt[max_success_index])

    def __quasi_Newton(self):
        # print('enter quasi newton')
        sorted_index=np.argsort(self.__particles['lbest_cost'])
        refine_index=sorted_index[:int(self.__n_swarm//4)]
        # print(f'refine_index.shape:{refine_index.shape}')
        refine_pos=self.__particles['lbest_position'][refine_index]
        for i in range(refine_pos.shape[0]):
            res=minimize(self.__problem.eval,refine_pos[i],method='BFGS',options={'maxiter':9})
            self.__fes+=res.nfev
            # print('lbest_cost.shape:{}'.format(self.__particles['lbest_cost'].shape))
            if self.__particles['lbest_cost'][refine_index[i]]>res.fun:
                # print('success')
                self.__particles['lbest_position'][refine_index[i]]=res.x
                self.__particles['lbest_cost'][refine_index[i]]=res.fun
                # uodate pbest
                self.__particles['pbest_position'][self.__lbest_index[refine_index[i]]]=res.x
                self.__particles['pbest'][self.__lbest_index[refine_index[i]]]=res.fun

    def run_episode(self, problem):
        self.__reset(problem)
        while self.__fes<self.__max_fes:
            while self.__fes<0.95*self.__max_fes:
                self.__cur_mode='ls'
                self.__gen+=1
                if self.__w_decay:
                    self.__w-=0.5/(self.__max_fes/self.__NP)

                # learning period:
                self.__success_num-=self.__success_num
                for j in range(self.__LP):
                    self.__update(problem)
                self.__update_parameter_set()
                if self.__gen%self.__L==0:
                    self.__quasi_Newton()

                if self.__gen%self.__R==0:
                    self.__random_regroup()
                    self.__update_lbest(init=True)

            while self.__fes<self.__max_fes:
                self.__cur_mode='gs'
                self.__update(problem)

            if problem.optimum is None:
                done = self.__fes>=self.__max_fes
            else:
                done = self.__fes>=self.__max_fes or self.__particles['gbest_val']<=1e-8
            if done:
                if len(self.cost) >= self.config.n_logpoint + 1:
                    self.cost[-1] = self.__particles['gbest_val']
                else:
                    self.cost.append(self.__particles['gbest_val'])
                break

        return {'cost': self.cost, 'fes': self.__fes}
    