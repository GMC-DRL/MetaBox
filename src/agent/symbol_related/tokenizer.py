# import tensorflow as tf
import torch
import numpy as np


'''tokenize the input or output'''

class Tokenizer:

    SPECIAL_SYMBOLS = {}

    SPECIAL_FLOAT_SYMBOLS = {}

    SPECIAL_OPERATORS = {}

    SPECIAL_INTEGERS = {}

    def __init__(self):
        self.start = "<START>"
        self.start_id = 1
        self.end = "<END>"
        self.end_id = 2
        self.pad = "<PAD>"
        self.pad_id = 0
        self.vocab = [self.pad, self.start, self.end]

    def encode(self, expr):
        raise NotImplementedError()

    def decode(self, expr):
        raise NotImplementedError()

    def is_unary(self, token):
        raise NotImplementedError()

    def is_binary(self, token):
        raise NotImplementedError()

    def is_leaf(self, token):
        raise NotImplementedError()

    def get_constant_ids(self):
        pass


class MyTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.variables = [
            "x",
            "gb",
            "gw",
            "dx",
            "randx",
            "pb"
            ]
        self.binary_ops = ["+", "*"]
        self.unary_ops = [
            # "sin",
            # "cos",
            "-",
            # "sign"
        ]
        self.constants = [f"C{i}" for i in range(-1, 1)]
        self.leafs = self.constants + self.variables
        self.vocab = list(self.binary_ops) + list(self.unary_ops) + self.leafs
        self.lookup_table = dict(zip(self.vocab,range(len(self.vocab))))
        self.leaf_index=np.arange(len(self.vocab))[len(self.vocab)-len(self.leafs):]
        self.operator_index=np.arange(len(self.vocab)-len(self.leafs))
        self.binary_index=np.arange(len(self.binary_ops))
        self.unary_index=np.arange(len(self.unary_ops))+len(self.binary_ops)
        self.vocab_size=len(self.vocab)
        self.constants_index=self.leaf_index[:len(self.constants)]
        self.non_const_index=list(set(range(self.vocab_size))-set(self.constants_index))
        self.var_index=self.leaf_index[len(self.constants):]

    def decode(self, expr):
        return self.vocab[expr]
    
    def encode(self, expr):
        return self.lookup_table[expr]
    
    def is_consts(self, id):
        if torch.is_tensor(id):
            id=id.cpu()
        return np.isin(id,self.constants_index)
        # return id in self.constants_index

    def is_binary(self, token):
        return token in self.binary_ops
    
    def is_unary(self, token):
        return token in self.unary_ops

    def is_leaf(self, token):
        return token in self.leafs

    def is_var(self,token):
        return token in self.variables