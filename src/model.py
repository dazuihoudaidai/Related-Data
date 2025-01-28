#coding=UTF-8
import math
import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, radius
from torch.nn import Linear, Sequential, Dropout, LeakyReLU, BatchNorm1d

# Message passing for Heterogeneous interaction
class GraphInteractionLayer(MessagePassing):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        kwargs.setdefault('aggr', 'add') 
        super(GraphInteractionLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # MLP layers for node features
        self.node_mlp_cov = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.output_dim)
        )
        
        self.node_mlp_ncov = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.output_dim)
        )

        # MLP layers for node coordinates
        self.coord_mlp_cov = nn.Sequential(nn.Linear(9, self.input_dim), nn.SiLU())
        self.coord_mlp_ncov = nn.Sequential(nn.Linear(9, self.input_dim), nn.SiLU())

    def forward(self, node_features, edge_index_intra, edge_index_inter, positions=None, size=None):
        # Process intra-graph edges
        row_cov, col_cov = edge_index_intra
        coord_diff_cov = positions[row_cov] - positions[col_cov]
        radial_cov = self.coord_mlp_cov(_rbf(torch.norm(coord_diff_cov, dim=-1), D_min=0., D_max=6., D_count=9, device=node_features.device))
        intra_nodes_output = self.propagate(edge_index=edge_index_intra, x=node_features, radial=radial_cov, size=size)

        # Process inter-graph edges
        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = positions[row_ncov] - positions[col_ncov]
        radial_ncov = self.coord_mlp_ncov(_rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=node_features.device))
        inter_nodes_output = self.propagate(edge_index=edge_index_inter, x=node_features, radial=radial_ncov, size=size)

        # Combine results
        result = self.node_mlp_cov(node_features + intra_nodes_output) + self.node_mlp_ncov(node_features + inter_nodes_output)
        return result

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        return x_j * radial

# Attention Mechanism to Refine Node Features
class AttentionModule(nn.Module):
    def __init__(self, channels, reduction_factor):
        super().__init__()
        self.mlp = Sequential(
            Linear(channels, channels // reduction_factor, bias=False),
            nn.ReLU(inplace=True),
            Linear(channels // reduction_factor, channels, bias=False),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.mlp:
            if isinstance(layer, Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, node_features, batch_indices, size=None):
        max_aggregated = scatter(node_features, batch_indices, dim=0, dim_size=size, reduce='max')
        sum_aggregated = scatter(node_features, batch_indices, dim=0, dim_size=size, reduce='sum')

        max_out = self.mlp(max_aggregated)
        sum_out = self.mlp(sum_aggregated)

        refined_weights = torch.sigmoid(max_out + sum_out)
        refined_weights = refined_weights[batch_indices]
        
        return node_features * refined_weights


# Fully Connected Layers for Final Prediction
class FullyConnectedLayer(nn.Module):
    def __init__(self, graph_layer_dim, fc_layer_dim, num_fc_layers, dropout_rate, num_tasks):
        super(FullyConnectedLayer, self).__init__()
        self.graph_layer_dim = graph_layer_dim
        self.fc_layer_dim = fc_layer_dim
        self.num_fc_layers = num_fc_layers
        self.dropout_rate = dropout_rate
        self.output_layer = nn.ModuleList()

        for i in range(self.num_fc_layers):
            if i == 0:
                self.output_layer.append(nn.Linear(self.graph_layer_dim, self.fc_layer_dim))
                self.output_layer.append(nn.Dropout(self.dropout_rate))
                self.output_layer.append(nn.LeakyReLU())
                self.output_layer.append(nn.BatchNorm1d(self.fc_layer_dim))
            elif i == self.num_fc_layers - 1:
                self.output_layer.append(nn.Linear(self.fc_layer_dim, num_tasks))
            else:
                self.output_layer.append(nn.Linear(self.fc_layer_dim, self.fc_layer_dim))
                self.output_layer.append(nn.Dropout(self.dropout_rate))
                self.output_layer.append(nn.LeakyReLU())
                self.output_layer.append(nn.BatchNorm1d(self.fc_layer_dim))

    def forward(self, h):
        for layer in self.output_layer:
            h = layer(h)

        return h

# Main Model Class
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Sequential(Linear(input_dim, hidden_dim), nn.SiLU())
        self.graph_conv_1 = GraphInteractionLayer(hidden_dim, hidden_dim)
        self.graph_conv_2 = GraphInteractionLayer(hidden_dim, hidden_dim)
        self.graph_conv_3 = GraphInteractionLayer(hidden_dim, hidden_dim)
        self.final_fully_connected = FullyConnectedLayer(hidden_dim, hidden_dim, 4, 0.15, 1)

    def forward(self, data):
        node_features, edge_index_intra, edge_index_inter, positions = data.x, data.edge_index_intra, data.edge_index_inter, data.pos

        # Process through initial linear layer
        node_features = self.input_layer(node_features)

        # Pass through graph convolution layers
        node_features = self.graph_conv_1(node_features, edge_index_intra, edge_index_inter, positions)
        node_features = self.graph_conv_2(node_features, edge_index_intra, edge_index_inter, positions)
        node_features = self.graph_conv_3(node_features, edge_index_intra, edge_index_inter, positions)

        # Aggregate node features
        node_features = global_add_pool(node_features, data.batch)
        
        # Final prediction layer
        node_features = self.final_fully_connected(node_features)

        return node_features.view(-1)