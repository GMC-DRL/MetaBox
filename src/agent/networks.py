import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        """
        :param config: a list of dicts like
                 [{'in':2,'out':4,'drop_out':0.5,'activation':'ReLU'},
                  {'in':4,'out':8,'drop_out':0,'activation':'Sigmoid'},
                  {'in':8,'out':10,'drop_out':0,'activation':'None'}],
                and the number of dicts is customized.
        """
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        self.net_config = config
        for layer_id, layer_config in enumerate(self.net_config):
            linear = nn.Linear(layer_config['in'], layer_config['out'])
            self.net.add_module(f'layer{layer_id}-linear', linear)
            drop_out = nn.Dropout(layer_config['drop_out'])
            self.net.add_module(f'layer{layer_id}-drop_out', drop_out)
            if layer_config['activation'] != 'None':
                activation = eval('nn.'+layer_config['activation'])()
                self.net.add_module(f'layer{layer_id}-activation', activation)

    def forward(self, x):
        return self.net(x)
