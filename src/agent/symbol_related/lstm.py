'''implement lstm'''
import torch
import torch.nn as nn

from .expression import *
import numpy as np
import math
from .utils import *

binary_code_len=4


class LSTM(nn.Module):
    def __init__(self,opts,tokenizer) -> None:
        super().__init__()
        self.opts=opts
        self.max_layer=opts.max_layer
        self.output_size=tokenizer.vocab_size
        self.hidden_size=opts.hidden_dim
        self.num_layers=opts.num_layers
        self.max_c=opts.max_c
        self.min_c=opts.min_c
        self.fea_size=opts.fea_dim
        self.tokenizer=tokenizer
        self.interval=opts.c_interval
        self.lstm=nn.LSTM(int(2**self.max_layer-1)*binary_code_len,self.hidden_size,self.num_layers,batch_first=True)
        self.output_net=nn.Linear(self.hidden_size,self.output_size)
        
        self.x_to_c=nn.Linear(self.fea_size,self.hidden_size)
        self.constval_net=nn.Linear(self.hidden_size,int((self.max_c-self.min_c)//self.interval))
        self.num_c=int((self.max_c-self.min_c)//self.interval)


    def forward(self,x,save_data=False,fix_action=None):

        bs=x.shape[0]
        device=x.device

        log_prob_whole=torch.zeros(bs).to(device)
        pre_seq=[]

        # initial input,h,c for lstm
        h_0=torch.zeros((self.num_layers,bs,self.hidden_size)).to(device)
        
        c_0=self.x_to_c(x)

        h=h_0
        c=c_0.unsqueeze(dim=0).repeat(self.num_layers,1,1)

        if save_data:
            memory=Memory()


        # generate seqence
        if not fix_action:
            len_seq=int(2**self.max_layer-1)
            seq=(torch.ones((bs,len_seq),dtype=torch.long)*-1)
            const_vals=torch.zeros((bs,len_seq))

            x_in=torch.zeros((bs,1,len_seq*binary_code_len)).to(device)

            # the generating position of the seq
            position=torch.zeros((bs,),dtype=torch.long)
            working_index=torch.arange(bs)
            # generate sequence
            while working_index.shape[0]>0:
                # print(f'h.shape:{h.shape},c.shape:{c.shape}, x_in.shape:{x_in.shape}, working_index: {working_index}')
                output,(h,c)=self.lstm(x_in,(h,c))
                # output,(h,c)=self.lstm(x_in)
                out=self.output_net(output)

                # 如果position为全-1，则mask为全0
                mask=get_mask(seq[working_index],self.tokenizer,position,self.max_layer)
                
                # mask=get_mask(pre_seq,self.tokenizer,position)
                log_prob,choice,binary_code=get_choice(out,mask)
                # prefix_seq.append(choice)
                
                # get c
                c_index=self.tokenizer.is_consts(choice)
                if np.any(c_index):
                    out_c=self.constval_net(output[c_index])
                    log_prob_c,c_val=get_c(out_c,self.min_c,self.interval)
                    log_prob_whole[working_index[c_index]]+=log_prob_c
                    const_vals[working_index[c_index],position[c_index]]=c_val.cpu()

                # store if needed 
                if save_data:
                    memory.c_index.append(c_index)
                    memory.position.append(position)
                    memory.working_index.append(working_index)
                    memory.mask.append(mask)
                    memory.x_in.append(x_in.clone().detach())

                # udpate
                # need to test!!!!
                x_in=x_in.clone().detach()
                binary_code = binary_code.to(device)
                for i in range(binary_code_len):
                    x_in[range(len(working_index)),0,position*binary_code_len+i]=binary_code[:,i]
                

                log_prob_whole[working_index]+=log_prob
                


                seq[working_index,position]=choice.cpu()
                
                position=get_next_position(seq[working_index],choice,position,self.tokenizer)
                
                # update working index when position is -1
                filter_index=(position!=-1)
                working_index=working_index[filter_index]
                # print(f'filter_index: {filter_index}, working_index: {working_index.shape[0]}')
                position=position[filter_index]
                x_in=x_in[filter_index]
                h=h[:,filter_index]
                c=c[:,filter_index]
                if save_data:
                    memory.filter_index.append(filter_index)
            
            # if self.opts.require_baseline:
            #     rand_seq,rand_c_seq=self.get_random_seq(bs)
            #     if not save_data:
            #         return seq.numpy(),const_vals.numpy(),log_prob_whole,rand_seq,rand_c_seq
            #     else:
            #         memory.seq=seq
            #         memory.c_seq=const_vals
            #         return seq.numpy(),const_vals.numpy(),log_prob_whole,rand_seq,rand_c_seq,memory.get_dict()
            
            if not save_data:
                # 返回等长的序列，数组表示的二叉树
                return seq.numpy(),const_vals.numpy(),log_prob_whole
            else:
                memory.seq=seq
                memory.c_seq=const_vals

                return seq.numpy(),const_vals.numpy(),log_prob_whole,memory.get_dict()
        else:
            # fix_action get the new log_prob
            x_in=fix_action['x_in']     # x_in shape: (len, [bs,1,31*4])
            mask=fix_action['mask']     # mask shape: (len, [bs,vocab_size])
            working_index=fix_action['working_index']   # working_index
            # seq=torch.FloatTensor(fix_action['seq']).to(device)
            seq=fix_action['seq']
            c_seq=fix_action['c_seq']
            # c_seq=torch.FloatTensor(fix_action['c_seq']).to(device)
            position=fix_action['position']
            c_indexs=fix_action['c_index']
            filter_index=fix_action['filter_index']

            for i in range(len(x_in)):
                output,(h,c)=self.lstm(x_in[i],(h,c))
                # output,(h,c)=self.lstm(x_in[i])
                out=self.output_net(output)

                w_index=working_index[i]
                pos=position[i]
                log_prob=get_choice(out,mask[i],fix_choice=seq[w_index,pos].to(device))
                log_prob_whole[w_index]+=log_prob

                c_index=c_indexs[i]
                # todo get c log_prob
                if np.any(c_index):
                    out_c=self.constval_net(output[c_index])
                    log_prob_c=get_c(out_c,self.min_c,self.interval,fix_c=c_seq[w_index[c_index],pos[c_index]])
                    log_prob_whole[w_index[c_index]]+=log_prob_c

                # update h & c
                h=h[:,filter_index[i]]
                c=c[:,filter_index[i]]
                
            return log_prob_whole
        
    def get_random_seq(self,bs):
        len_seq=int(2**self.max_layer-1)
        seq=(torch.ones((bs,len_seq),dtype=torch.long)*-1)
        const_vals=torch.zeros((bs,len_seq))
        position=torch.zeros((bs,),dtype=torch.long)

        working_index=torch.arange(bs)
        # generate sequence
        while working_index.shape[0]>0 :

            output=torch.rand((working_index.shape[0],1,self.output_size))

            mask=get_mask(seq[working_index],self.tokenizer,position,self.max_layer)
            
            _,choice,_=get_choice(output,mask)
            
            c_index=self.tokenizer.is_consts(choice)
            
            if np.any(c_index):
                bs=output[c_index].shape[0]
                out_c=torch.rand((bs,1,self.num_c))
                _,c_val=get_c(out_c,self.min_c,self.interval)
                const_vals[working_index[c_index],position[c_index]]=c_val

            seq[working_index,position]=choice

            position=get_next_position(seq[working_index],choice,position,self.tokenizer)
            
            # update working index when position is -1
            filter_index=(position!=-1)
            working_index=working_index[filter_index]
            position=position[filter_index]
            

        return seq.numpy(),const_vals.numpy()
    
if __name__ == '__main__':
    model=LSTM(fea_size=10,hidden_size=20,output_size=15,num_layers=1,max_length=10)
    x=torch.rand((8,10),dtype=torch.float)
    seq,log_prob=model(x)
    print(f'seq:{seq}')
    print(f'log_prob:{log_prob}')