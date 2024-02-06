import torch 
import torch.nn as nn
from torch.autograd import Variable
import model.CRUCell as CRUCell

class _CRU(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, bias = True):
        super(_CRU, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.bias = bias
        self.out_dim = out_dim
        
        self.cell_list = nn.ModuleList()
        self.cell_list.append(CRUCell._CRUCell(self.in_dim,
                                               self.hid_dim, 
                                               self.bias))
        for l in range(1, self.num_layers):
            self.cell_list.append(CRUCell._CRUCell(self.hid_dim,
                                                   self.hid_dim,
                                                   self.bias))

        self.fc = nn.Sequential(
            nn.Linear(self.hid_dim*self.num_layers, self.out_dim),
            nn.ELU()
            )

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            

    def forward(self, input, hid_state=None):

        if hid_state is None:
            hid_state = Variable(torch.zeros(self.num_layers, 3, input.size(0), self.hid_dim)).cpu()
        else:
            hid_state = hid_state.to(self.device)

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(hid_state[layer,:, :, :])
        
        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hid_layer = self.cell_list[layer](input[:, t, :], hidden[layer]).to(self.device)
                else:
                    hid_layer = self.cell_list[layer](hidden[layer - 1],hidden[layer]).to(self.device)
                
                hidden[layer] = hid_layer

            outs.append(hid_layer)
        
        feature = torch.mean(outs[-1],axis=1)
        out = torch.sum(outs[-1],axis=0)
        out = self.fc(out)
        
        return out, feature