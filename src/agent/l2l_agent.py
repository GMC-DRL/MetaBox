import torch
from torch import nn
from agent.basic_agent import Basic_Agent
from agent.utils import *


def scale(x,lb,ub):
    x=torch.sigmoid(x)
    x=lb+(ub-lb)*x
    return x


class L2L_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        config.lr=1e-5
        self.__config=config
        self.hidden_size=32
        self.proj_size=config.dim
        torch.set_default_dtype(torch.float64)
        self.net=nn.LSTM(input_size=config.dim+2,hidden_size=self.hidden_size,proj_size=config.dim)
        self.optimizer = torch.optim.Adam([{'params': self.net.parameters(), 'lr': config.lr}])
        self.__learning_step=0

        save_class(self.__config.agent_save_dir,'checkpoint0',self)
        self.__cur_checkpoint=1

    def train_episode(self, env):
        T=100
        train_interval=10
        t=0
        dim=self.__config.dim

        # init input to zeros
        input=torch.zeros((self.__config.dim+2),dtype=torch.float64)
        input=input[None,None,:]
        y_sum=0
        # init h & c to zeros
        h=torch.zeros((self.proj_size),dtype=torch.float64)[None,None,:]
        c=torch.zeros((self.hidden_size),dtype=torch.float64)[None,None,:]
        exceed_max_ls=False
        env.reset()
        while t < T:
            out,(h,c)=self.net(input,(h,c))
            # get new x
            x=out[0,0]
            # print(x)
            y,_,_=env.step(x)
            # x=scale(x,problem.lb,problem.ub)
            # # evaluate x
            # if problem.optimum is None:
            #     y=problem.eval(x)
            # else:
            #     y=problem.eval(x)-problem.optimum
            # y=y.to(torch.float32)
            # print(f'x.dtype:{x.dtype}')
            # print(f'y.dtype:{y.dtype}')

            # print(y)
            y_sum+=y

            # update input
            input=torch.cat((x,torch.tensor([y]),torch.tensor([1])))[None,None,:]
            
            t+=1

            # update network
            if t % train_interval == 0:
                loss=y_sum
                self.optimizer.zero_grad()
                loss.mean().backward()

                # for name, parms in self.net.named_parameters():	
                #     print('-->name:', name)
                #     print('-->grad_value:',parms.grad)
                self.optimizer.step()
                y_sum=y_sum.detach()
                h=h.detach()
                c=c.detach()
                input=input.detach()
                self.__learning_step+=1
                if self.__learning_step >= (self.__config.save_interval * self.__cur_checkpoint):
                    save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
                    self.__cur_checkpoint+=1
                if self.__learning_step >= self.__config.max_learning_step:
                    exceed_max_ls=True
                    break
                # loss.detach()
        # return exceed_max_ls
        return exceed_max_ls,{'normalizer': env.optimizer.cost[0],
                               'gbest': env.optimizer.cost[-1],
                               'return': 0,         # set to 0 since for non-RL approach there is no return
                               'learn_steps': self.__learning_step}


    # rollout_episode need transform 
    def rollout_episode(self,env) :
        torch.set_grad_enabled(False)
        T=100
        
        dim=self.__config.dim
        
        fes=0
        
        best=None
        cost=[]
        
        t=0
        input=torch.zeros((self.__config.dim+2),dtype=torch.float64)
        input=input[None,None,:]
        
        h=torch.zeros((self.proj_size),dtype=torch.float64)[None,None,:]
        c=torch.zeros((self.hidden_size),dtype=torch.float64)[None,None,:]
        env.reset()
        while t < T:
            out,(h,c)=self.net(input,(h,c))
            x=out[0,0]
            y,_,is_done=env.step(x.detach().numpy())
            # x=scale(x,problem.lb,problem.ub)
            # # print(x)
            # if problem.optimum is None:
            #     y=problem.eval(x)
            # else:
            #     y=problem.eval(x)-problem.optimum
            fes+=1
            # print(y)
            if best is None:
                best=y
            elif y<best:
                best=y
                
            input=torch.cat((x,torch.tensor([y]),torch.tensor([1])))[None,None,:]
            if is_done:
                break
            t+=1
            
        torch.set_grad_enabled(True)
        return {'cost':env.optimizer.cost[::2],'fes':fes}
