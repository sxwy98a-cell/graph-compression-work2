import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
import torch
from torch_geometric.nn import HeteroConv, SAGEConv, GraphConv
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('author', 'affiliated_with', 'institution'): GraphConv(-1, hidden_channels),
                ('author', 'writes', 'paper'): GraphConv(-1, hidden_channels),
                ('paper', 'cites', 'paper'): GraphConv(-1, hidden_channels),
                ('paper', 'has_topic', 'field_of_study'): GraphConv(-1, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            orig_x_dict = x_dict
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict['author'] = x_dict.get('author', orig_x_dict['author'])
        return self.lin(x_dict['paper'])