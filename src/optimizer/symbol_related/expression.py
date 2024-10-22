'''imprement the expression related operation'''

import torch
import numpy as np
from .tokenizer import Tokenizer
import math


# expression related function


def get_mask(pre_seq,tokenizer,position,max_layer):
    if len(pre_seq.shape)==1:
        pre_seq=[pre_seq]
    bs,_=pre_seq.size()
    old_device=pre_seq.device
    pre_seq=pre_seq.cpu().numpy()
    position=position.cpu().numpy()
    masks=[]
    for sub_seq,pos in zip(pre_seq,position):
        # if position==-1: set mask all to be zero
        if pos == -1:
            mask=np.zeros(tokenizer.vocab_size)
            masks.append(mask)
            continue
        # init mask
        mask=np.ones(tokenizer.vocab_size)
        # rule: token in the root should not be operands
        if pos==0:
            mask[tokenizer.leaf_index]=0
            # mask[tokenizer.encode('sign')]=0
            # mask[tokenizer.encode('sin')]=0
            # mask[tokenizer.encode('cos')]=0
            
            # mask[tokenizer.encode('*')]=0
            mask[tokenizer.encode('-')]=0
        else:
            # rule: Avoid invalid operations of + -
            father_token=tokenizer.decode(sub_seq[(pos-1)//2])
            
            if (tokenizer.is_binary(father_token) and pos%2 == 0) or tokenizer.is_unary(father_token):
                neg_ancestor,target_vocab=find_prefix_of_token_ancestor(tokenizer,sub_seq,pos,'-')
                # rule: direct child of - should not be - or +
                if neg_ancestor == (pos-1)//2:
                    mask[tokenizer.encode('+')]=0
                    mask[tokenizer.encode('-')]=0
                    # rule: direct child of - located in root should not be x
                    if neg_ancestor == 0:
                        mask[tokenizer.encode('x')]=0
                
                if target_vocab is not None:
                    pre_vocab=along_continuous_plus(tokenizer,sub_seq,neg_ancestor)
                    
                    if pre_vocab is not None:
                        mask_index=test_pre(target_vocab[1:],pre_vocab,tokenizer)
                        mask[mask_index]=0
            
            
            if father_token == '+' or (tokenizer.is_binary(father_token) and pos%2 == 0) or tokenizer.is_unary(father_token):
                plus_ancestor,target_vocab=find_prefix_of_token_ancestor(tokenizer,sub_seq,pos,'+')
                # print(f'plus_ancestor:{plus_ancestor}')
                if target_vocab is not None:
                    visited=np.zeros_like(sub_seq)
                    if father_token=='+' and left_or_right(pos,plus_ancestor)=='l':
                        visited[2*plus_ancestor+1]=1
                        target_vocab=get_prefix(sub_seq,2*plus_ancestor+1)
                    else:
                        visited[2*plus_ancestor+2]=1
                        target_vocab=get_prefix(sub_seq,2*plus_ancestor+2)
                    
                    sub_root_list=get_along_continuous_plus_with_minus(tokenizer,sub_seq,plus_ancestor,visited)
                    
                    pre_vocab=[get_prefix(sub_seq,sub_root) for sub_root in sub_root_list]
                    if pre_vocab is not None:
                        mask_index=test_pre(target_vocab,pre_vocab,tokenizer)
                        mask[mask_index]=0
            # rule: pure calculation between constant values is not allowed
            if have_continous_const(sub_seq,pos,tokenizer):
                mask[tokenizer.constants_index]=0
            
            # rule: [sin cos sign] cannot directly nest with each other (if they are in the basis symbol set)
            # if father_token in ['sin','cos']:
            #     mask[tokenizer.encode('sign')]=0
            #     mask[tokenizer.encode('sin')]=0
            #     # mask[tokenizer.encode('cos')]=0
            # if father_token == 'sign':
            #     mask[tokenizer.encode('sign')]=0
                
            # rule: the direct children of + should not be constant values
            if father_token == '+' or father_token == '-':
                mask[tokenizer.constants_index]=0

            
            
            if father_token == '+':
                # children of sign should not be sign (if sign is in the basis symbol set)
                # mask[tokenizer.encode('sign')]=0

                # rule: x+x, gbest+gbest ... is not allowed
                if pos%2==0: 
                    left_token=tokenizer.decode(sub_seq[pos-1])
                    if tokenizer.is_leaf(left_token) and left_token!='randx':
                        mask[sub_seq[pos-1]]=0
            
            # rule: children of * should not be the same
            if father_token == '*':
                mask[tokenizer.encode('*')]=0
                mask[tokenizer.encode('-')]=0
                if pos%2==0:
                    left_id=sub_seq[pos-1]
                    if not tokenizer.is_consts(left_id):
                        mask[tokenizer.non_const_index]=0
                    else:
                        mask[tokenizer.constants_index]=0
        
            # ! optional: set the minimum layer of the equation tree (you can uncomment the following code if needed)
            if which_layer(position=pos)<=2:
                if father_token=='*':
                    mask[tokenizer.var_index]=0
                elif (tokenizer.is_binary(father_token) and pos%2 == 0 and tokenizer.is_leaf(tokenizer.decode(sub_seq[pos-1]))) or tokenizer.is_unary(father_token):
                    mask[tokenizer.leaf_index]=0

            # rule: the leaves should not be operators
            if pos >= int(2**(max_layer-1)-1):
                mask[tokenizer.operator_index]=0
        # if np.all(mask<=0.2):
        #     # mask[tokenizer.leaf_index]=1
        #     print(f'mask:{mask}, pos:{pos}, seq:{sub_seq}')
        masks.append(mask)
    
    return torch.FloatTensor(masks).to(old_device)

def which_layer(position):
    level = math.floor(math.log2(position + 1))
    return level+1

def left_or_right(position,root):
    tmp=position
    while tmp!=root:
        position=(position-1)//2
        if position == root:
            if 2*root+1 == tmp:
                return 'l'
            else:
                return 'r'
        tmp=position


def have_continous_const(seq,position,tokenizer):
    father_index=(position-1)//2
    father_token=tokenizer.decode(seq[father_index])
    if tokenizer.is_unary(father_token):
        return True
    if tokenizer.is_binary(father_token):
        if position==father_index*2+1:
            return False
        elif tokenizer.is_consts(seq[father_index*2+1]):
            return True

def continus_mul_c(seq,position,tokenizer):
    list=[]
    sub_root=(position-1)//2
    if tokenizer.decode(seq[sub_root])=='*':
        visited=np.zeros_like(seq)
        visited[position]=1
        
        return get_along_continuous_mul(tokenizer,seq,sub_root,visited)
    else:
        return False

def get_along_continuous_mul(tokenizer,seq,begin,visited):
    
    # list.append(begin)
    visited[begin]=1
    
    if begin!=0 and visited[(begin-1)//2]!=1:
        father_token=tokenizer.decode(seq[(begin-1)//2])
        if father_token=='*':
            if get_along_continuous_mul(tokenizer,seq,(begin-1)//2,visited):
                return True
    
    if visited[begin*2+1]==0 and seq[begin*2+1]!=-1:
        left_child_token=tokenizer.decode(seq[begin*2+1])
        if left_child_token=='*':
            if get_along_continuous_mul(tokenizer,seq,begin*2+1,visited):
                return True
        elif left_child_token[0]=='C':
            return True
    
    if visited[begin*2+2]==0 and seq[begin*2+2]!=-1:
        right_child_token=tokenizer.decode(seq[begin*2+2])
        if right_child_token=='*':
            if get_along_continuous_mul(tokenizer,seq,begin*2+2,visited):
                return True
        elif right_child_token[0]=='C':
            return True
    
    return False

def test_pre(target_vocab,pre_vocab,tokenizer):
    target_len=len(target_vocab)
    mask_index=[]
    for pre_prefix in pre_vocab:
        if len(pre_prefix)==target_len+1 and np.all(pre_prefix[:-1]==target_vocab):
            last_token=tokenizer.decode(pre_prefix[-1])
            if last_token != 'randx' and last_token[0] != 'C':
                mask_index.append(pre_prefix[-1])
            
            
    return mask_index
        

def get_along_continuous_plus_with_minus(tokenizer,seq,begin,visited):
    list=[]
    
    # list.append(begin)
    visited[begin]=1
    

    if begin!=0 and visited[(begin-1)//2]==0:
        father_token=tokenizer.decode(seq[(begin-1)//2])
        if father_token=='+':
            l=get_along_continuous_plus_with_minus(tokenizer,seq,(begin-1)//2,visited)
            list.extend(l)
   
    
    if visited[begin*2+1]==0 and seq[begin*2+1]!=-1:
        left_child_token=tokenizer.decode(seq[begin*2+1])
        if left_child_token=='+':
            l=get_along_continuous_plus_with_minus(tokenizer,seq,begin*2+1,visited)
            list.extend(l)
        elif left_child_token == '-':
            list.append(2*(begin*2+1)+1)
    
    if visited[begin*2+2]==0 and seq[begin*2+2]!=-1:
        right_child_token=tokenizer.decode(seq[begin*2+2])
        if right_child_token=='+':
            l=get_along_continuous_plus_with_minus(tokenizer,seq,begin*2+2,visited)
            list.extend(l)
        elif left_child_token == '-':
            list.append(2*(begin*2+2)+1)
    
    return list

def get_along_continuous_plus(tokenizer,seq,begin,visited):
    list=[]
    # list.append(begin)
    along_root=False
    visited[begin]=1
    if begin == 0 and seq[begin] == tokenizer.encode('+'):
        along_root = True

    if begin!=0 and visited[(begin-1)//2]==0:
        father_token=tokenizer.decode(seq[(begin-1)//2])
        if father_token=='+':
            l,flag=get_along_continuous_plus(tokenizer,seq,(begin-1)//2,visited)
            list.extend(l)
            if flag:
                along_root=True
   
    
    if visited[begin*2+1]==0 and seq[begin*2+1]!=-1:
        left_child_token=tokenizer.decode(seq[begin*2+1])
        if left_child_token=='+':
            l,flag=get_along_continuous_plus(tokenizer,seq,begin*2+1,visited)
            list.extend(l)
            if flag:
                along_root=True
        else:
            list.append(begin*2+1)
    
    if visited[begin*2+2]==0 and seq[begin*2+2]!=-1:
        right_child_token=tokenizer.decode(seq[begin*2+2])
        if right_child_token=='+':
            l,flag=get_along_continuous_plus(tokenizer,seq,begin*2+2,visited)
            list.extend(l)
            if flag:
                along_root=True
        else:
            list.append(begin*2+2)
    
    return list,along_root

def along_continuous_plus(tokenizer,seq,neg_ancestor):
    list=[]
    sub_root=(neg_ancestor-1)//2
    if tokenizer.decode(seq[sub_root])=='+':
        visited=np.zeros_like(seq)
        visited[neg_ancestor]=1
        continuous_plus_token_list,along_root=get_along_continuous_plus(tokenizer,seq,sub_root,visited)
        
        pre_vocab=[get_prefix(seq,sub_root) for sub_root in continuous_plus_token_list]

        if along_root:
            pre_vocab.append([tokenizer.encode('x')])
        return pre_vocab
    else:
        return None
    


def find_prefix_of_token_ancestor(tokenizer,seq,position,token):
    
    while True:
        father_index=(position-1)//2
        father_token=tokenizer.decode(seq[father_index])
        if father_token!=token:
            position=father_index
            if position==0:
                break
        else:
            return father_index,get_prefix(seq,father_index)
    return -1,None

def get_prefix(seq,sub_root):
    if  sub_root>=len(seq) or seq[sub_root]==-1:
        return []
    list=[]
    list.append(seq[sub_root])
    list.extend(get_prefix(seq,2*sub_root+1))
    list.extend(get_prefix(seq,2*sub_root+2))
    return list

def get_prefix_with_consts(seq,consts,sub_root):
    if  sub_root>=len(seq) or seq[sub_root]==-1:
        return [],[]
    list_expr=[]
    list_c=[]
    list_expr.append(seq[sub_root])
    list_c.append(consts[sub_root])
    left_output=get_prefix_with_consts(seq,consts,2*sub_root+1)
    list_expr.extend(left_output[0])
    list_c.extend(left_output[1])
    right_output=get_prefix_with_consts(seq,consts,2*sub_root+2)
    
    list_expr.extend(right_output[0])
    list_c.extend(right_output[1])
    return list_expr,list_c

def get_next_position(seq,choice,position,tokenizer):
    old_device=position.device
    position=position.cpu().numpy()
    choice=choice.cpu().numpy()
    seq=seq.cpu().numpy()
    next_position=[]
    for i in range(len(position)):
        c=choice[i]
        pos=position[i]
        sub_seq=seq[i]
        if c in tokenizer.operator_index:
            next_position.append(2*pos+1)
        else:
            append_index=-1
            while True:
                father_index=(pos-1)//2
                if father_index<0:
                    break
                if sub_seq[father_index] in tokenizer.binary_index and sub_seq[2*father_index+2]==-1:
                    append_index=father_index*2+2
                    break
                pos=father_index
            next_position.append(append_index)
        
    return torch.tensor(next_position,dtype=torch.long).to(old_device)
# 
def get_str_prefix(seq,const_vals,tokenizer):
    str_expr=[]
    c=[]
    for i,token_id in enumerate(seq):
        if token_id != -1:
            str_expr.append(tokenizer.decode(token_id))
            c.append(const_vals[i])
    return str_expr,c


def prefix_to_infix(
    expr, constants, tokenizer: Tokenizer
):
    stack = []
    for i, symbol in reversed(list(enumerate(expr))):
        if tokenizer.is_binary(symbol):
            if len(stack) < 2:
                return False, None
            tmp_str = "(" + stack.pop() + symbol + stack.pop() + ")"
            stack.append(tmp_str)
        elif tokenizer.is_unary(symbol) or symbol == "abs":
            if len(stack) < 1:
                return False, None
            if symbol in tokenizer.SPECIAL_SYMBOLS:
                stack.append(tokenizer.SPECIAL_SYMBOLS[symbol].format(stack.pop()))
            else:
                stack.append(symbol + "(" + stack.pop() + ")")
        elif tokenizer.is_leaf(symbol):
            if symbol == "C":
                stack.append(str(constants[i]))
            elif "C" in symbol:
                exponent = int(symbol[1:])
                stack.append(str(constants[i] * 10 ** exponent))
            else:
                stack.append(symbol)

    if len(stack) != 1:
        return False, None

    return True, stack.pop()

from typing import List

from sympy import lambdify

def expr_to_func(sympy_expr, variables: List[str]):
    

    return lambdify(
        variables,
        sympy_expr,
        modules=["numpy"],
    )