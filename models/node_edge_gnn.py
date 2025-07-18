import torch
from torch import Tensor
from torch.nn import Identity
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from utils.model_utils import make_mlp, get_activation_func

from .base_model import BaseModel

class NodeEdgeGNN(BaseModel):
    '''
    Model that uses message passing to update both node and edge features. Can predict for both simultaneously.
    '''
    def __init__(self,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 residual: bool = True,
                 mlp_layers: int = 2,
                 enforce_flipped_edge_attr: bool = True,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: str = None,
                 decoder_layers: int = 0,
                 decoder_activation: str = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0
        self.with_residual = residual
        self.enforce_flipped_edge_attr = enforce_flipped_edge_attr


        # Encoder
        encoder_decoder_hidden = hidden_features*2
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=self.input_node_features, output_size=hidden_features,
                                                hidden_size=encoder_decoder_hidden, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)
            self.edge_encoder = make_mlp(input_size=self.input_edge_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)

        input_node_size = hidden_features if self.with_encoder else self.input_node_features
        output_node_size = hidden_features if self.with_decoder else self.output_node_features
        input_edge_size = hidden_features if self.with_encoder else self.input_edge_features
        output_edge_size = hidden_features if self.with_decoder else self.output_edge_features
        self.convs = self._make_gnn(input_node_size=input_node_size, output_node_size=output_node_size,
                                    input_edge_size=input_edge_size, output_edge_size=output_edge_size,
                                    num_layers=num_layers, mlp_layers=mlp_layers, activation=activation, device=self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.output_node_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)
            self.edge_decoder = make_mlp(input_size=hidden_features, output_size=self.output_edge_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if self.with_residual:
            self.residual = Identity()
            self.res_activation = get_activation_func(activation, device=self.device)

    def _make_gnn(self, input_node_size: int, output_node_size: int, input_edge_size: int, output_edge_size: int,
                  num_layers: int, mlp_layers: int, activation: str, device: str):
        if num_layers == 1:
            return NodeEdgeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                                num_layers=mlp_layers, activation=activation, device=device)

        layers = []
        for _ in range(num_layers):
            layers.append((
                NodeEdgeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                             num_layers=mlp_layers, activation=activation, device=device),
                'x, edge_index, edge_attr -> x, edge_attr',
            ))
        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index, edge_attr = graph.x.clone(), graph.edge_index.clone(), graph.edge_attr.clone()
        x0, edge_attr0 = x, edge_attr

        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.convs(x, edge_index, edge_attr)

        if self.with_decoder:
            x = self.node_decoder(x)
            edge_attr = self.edge_decoder(edge_attr)

        if self.with_residual:
            x = self.res_activation(x + self.residual(x0[:, -self.output_node_features:]))
            edge_attr = self.res_activation(edge_attr + self.residual(edge_attr0[:, -self.output_edge_features:]))

        if self.enforce_flipped_edge_attr:
            edge_attr = self._overwrite_edge_attr(edge_attr, edge_index, graph.num_graphs)

        return x, edge_attr

    def _overwrite_edge_attr(self, edge_attr: Tensor, edge_index: Tensor, num_graphs: int) -> Tensor:
        temp = edge_index.clone()
        temp[0, :], temp[1, :] = temp[1, :], temp[0, :]
        undirected_mask = torch.all(torch.isin(edge_index, temp), dim=0)

        # Assume first half of edges are the corresponding flipped edges for the second half (PyG standard)
        num_undirected_edges = undirected_mask.sum().item()
        num_edges_per_graph = num_undirected_edges // num_graphs
        first_half_mask = torch.arange(num_edges_per_graph) < (num_edges_per_graph // 2)
        first_half_mask = first_half_mask.repeat(num_graphs)

        new_undir_edge_attr = edge_attr[undirected_mask].squeeze()
        # Overwrite second half of undirected edge attributes
        new_undir_edge_attr[~first_half_mask] = new_undir_edge_attr[first_half_mask] * -1

        new_edge_attr = edge_attr.clone().squeeze()
        new_edge_attr[undirected_mask] = new_undir_edge_attr
        new_edge_attr = new_edge_attr.view(-1, self.output_edge_features)

        return new_edge_attr

class NodeEdgeConv(MessagePassing):
    """
    Message = MLP
    Aggregate = sum
    Update = MLP
    """
    def __init__(self, node_in_channels: int, node_out_channels: int, edge_in_channels: int, edge_out_channels: int,
                 num_layers: int = 2, activation: str = 'prelu', device: str = 'cpu'):
        super().__init__(aggr='sum')
        msg_input_size = (node_in_channels * 2 + edge_in_channels)
        msg_hidden_size = msg_input_size * 2
        self.msg_mlp = make_mlp(input_size=msg_input_size, output_size=edge_out_channels,
                            hidden_size=msg_hidden_size, num_layers=num_layers,
                            activation=activation, device=device)

        update_input_size = (node_in_channels + edge_out_channels)
        update_hidden_size = update_input_size * 2
        self.update_mlp = make_mlp(input_size=update_input_size, output_size=node_out_channels,
                                 hidden_size=update_hidden_size, num_layers=num_layers,
                                 activation=activation, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x, edge_attr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def propagate(self, edge_index, **kwargs):
        coll_dict = self._collect(self._user_args, edge_index, [None, None], kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        out = self.aggregate(msg, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        out = self.update(out, **update_kwargs)
        return out, msg

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out: Tensor, x: Tensor):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))
