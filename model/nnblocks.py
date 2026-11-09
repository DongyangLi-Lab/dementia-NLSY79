import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class _BasicBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=False):
        super(_BasicBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        out = self.block(self.norm(x))
        return out


class _GraphBasicBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=False):
        super(_GraphBasicBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        # Don't use `gnn.Sequential`, which causes bugs with pickle dump and load
        self.block = gnn.SAGEConv(input_size, output_size)

    def forward(self, x, edge_index):
        out = self.block(self.norm(x), edge_index)
        return out


class _GCNBasicBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=False):
        super(_GCNBasicBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        self.block = gnn.GCNConv(input_size, output_size)

    def forward(self, x, edge_index):
        out = self.block(self.norm(x), edge_index)
        return out


class _GATBasicBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=False, heads=1):
        super(_GATBasicBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        self.block = gnn.GATConv(input_size, output_size, heads=heads, concat=False)

    def forward(self, x, edge_index):
        out = self.block(self.norm(x), edge_index)
        return out


class _ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=False):
        super(_ResidualBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(inplace=True),
            nn.Linear(output_size, output_size),
        )

        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Identity(),
            )

    def forward(self, x):
        residual = x
        out = self.block(self.norm(x))
        out += self.shortcut(residual)
        return out


class _GraphResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=False):
        super(_GraphResidualBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        # Don't use `gnn.Sequential`, which causes bugs with pickle dump and load
        # self.block = gnn.Sequential('x, edge_index', [
        #     (gnn.SAGEConv(input_size, output_size), 'x, edge_index -> x'),
        #     nn.ReLU(inplace=True),
        #     (gnn.SAGEConv(output_size, output_size), 'x, edge_index -> x'),
        # ])
        self.conv1 = gnn.SAGEConv(input_size, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = gnn.SAGEConv(output_size, output_size)

        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Identity(),
            )

    def forward(self, x, edge_index):
        residual = x

        # Block computations
        out = self.conv1(self.norm(x), edge_index)
        out = self.relu(out)
        out = self.conv2(out, edge_index)

        out += self.shortcut(residual)
        return out


class _GCNResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=False):
        super(_GCNResidualBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        self.conv1 = gnn.GCNConv(input_size, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = gnn.GCNConv(output_size, output_size)

        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Identity(),
            )

    def forward(self, x, edge_index):
        residual = x

        # Block computations
        out = self.conv1(self.norm(x), edge_index)
        out = self.relu(out)
        out = self.conv2(out, edge_index)

        out += self.shortcut(residual)
        return out


class _GATResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=False, heads=1):
        super(_GATResidualBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        self.conv1 = gnn.GATConv(input_size, output_size, heads=heads, concat=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = gnn.GATConv(output_size, output_size, heads=heads, concat=False)

        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Identity(),
            )

    def forward(self, x, edge_index):
        residual = x

        # Block computations
        out = self.conv1(self.norm(x), edge_index)
        out = self.relu(out)
        out = self.conv2(out, edge_index)

        out += self.shortcut(residual)
        return out


class MLP(nn.Module):
    def __init__(
            self, input_size, hidden_layer_sizes, output_size, dropout_p=0.0,
            use_residual=True, use_batch_norm=False, multiple_dropout=False):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.n_hidden = len(self.hidden_layer_sizes) - 1

        # Multiple dropout for feature selection
        #     1. apply dropout in eval stage;
        #     2. drop neurons rather than weights;
        #     3. must dropout for each layer? Not necessary.
        self.dropout_p = dropout_p
        self.multiple_dropout = multiple_dropout

        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        if self.use_residual:
            self.block = _ResidualBlock
        else:
            self.block = _BasicBlock
        self.block = functools.partial(
            self.block, use_batch_norm=self.use_batch_norm)

        # NOTE: For grad-based feature selection method, input layer's bias is False
        # to make it convenient to implement, we use `nn.Linear` as input only
        self.input = nn.Linear(
            self.input_size, self.hidden_layer_sizes[0], bias=True)
        self.hiddens = nn.ModuleList([
            self.block(hidden_layer_sizes[h], hidden_layer_sizes[h+1])
            for h in range(self.n_hidden)
        ])
        self.output = self.block(hidden_layer_sizes[-1], output_size)

    @property
    def drop_mode(self):
        if self.multiple_dropout:
            return not self.training
        else:
            return self.training

    def forward(self, x):
        x = F.relu(self.input(x), inplace=True)
        for hidden in self.hiddens:
            x = F.relu(hidden(x), inplace=True)
            x = F.dropout(x, p=self.dropout_p, training=self.drop_mode)
        x = F.sigmoid(self.output(x))
        return x.squeeze(1)


class GNN(nn.Module):
    def __init__(
            self, input_size, hidden_layer_sizes, output_size, dropout_p=0.0, alpha=0.95,
            use_residual=True, use_batch_norm=False, multiple_dropout=False):
        super(GNN, self).__init__()

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.n_hidden = len(self.hidden_layer_sizes) - 1

        # Dropout
        #     1. drop neurons rather than weights;
        #     2. apply dropout in eval stage;
        #     3. must dropout for each layer?
        self.dropout_p = dropout_p
        self.multiple_dropout = multiple_dropout
        # Special to GraphNet
        self.alpha = alpha

        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        if self.use_residual:
            self.block = _GraphResidualBlock
        else:
            self.block = _GraphBasicBlock
        self.block = functools.partial(
            self.block, use_batch_norm=self.use_batch_norm)

        # NOTE: For grad-based feature selection method, input layer's bias is False
        # to make it convenient to implement, we use `nn.Linear` as input only
        self.input = nn.Linear(
            self.input_size, self.hidden_layer_sizes[0], bias=True)
        self.hiddens = nn.ModuleList([
            self.block(hidden_layer_sizes[h], hidden_layer_sizes[h+1])
            for h in range(self.n_hidden)
        ])
        self.output = self.block(hidden_layer_sizes[-1], output_size)

    @property
    def drop_mode(self):
        if self.multiple_dropout:
            return not self.training
        else:
            return self.training

    def create_edge_index(self, x):
        similarity_matrix = torch.abs(F.cosine_similarity(
            x[None, :, :], x[:, None, :], dim=-1))
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps
        row, col = torch.where(adj_matrix)
        edge_index = torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0)
        return edge_index

    def forward(self, x):
        edge_index = self.create_edge_index(x)

        x = F.relu(self.input(x), inplace=True)
        for hidden in self.hiddens:
            x = F.relu(hidden(x, edge_index), inplace=True)
            x = F.dropout(x, p=self.dropout_p, training=self.drop_mode)
        x = F.sigmoid(self.output(x, edge_index))
        return x.squeeze(1)


class GCN(nn.Module):
    def __init__(
            self, input_size, hidden_layer_sizes, output_size, dropout_p=0.0, alpha=0.95,
            use_residual=True, use_batch_norm=False, multiple_dropout=False):
        super(GCN, self).__init__()

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.n_hidden = len(self.hidden_layer_sizes) - 1

        # Dropout
        self.dropout_p = dropout_p
        self.multiple_dropout = multiple_dropout
        # Special to GraphNet
        self.alpha = alpha

        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        if self.use_residual:
            self.block = _GCNResidualBlock
        else:
            self.block = _GCNBasicBlock
        self.block = functools.partial(
            self.block, use_batch_norm=self.use_batch_norm)

        self.input = nn.Linear(
            self.input_size, self.hidden_layer_sizes[0], bias=True)
        self.hiddens = nn.ModuleList([
            self.block(hidden_layer_sizes[h], hidden_layer_sizes[h+1])
            for h in range(self.n_hidden)
        ])
        self.output = self.block(hidden_layer_sizes[-1], output_size)

    @property
    def drop_mode(self):
        if self.multiple_dropout:
            return not self.training
        else:
            return self.training

    def create_edge_index(self, x):
        similarity_matrix = torch.abs(F.cosine_similarity(
            x[None, :, :], x[:, None, :], dim=-1))
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps
        row, col = torch.where(adj_matrix)
        edge_index = torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0)
        return edge_index

    def forward(self, x):
        edge_index = self.create_edge_index(x)

        x = F.relu(self.input(x), inplace=True)
        for hidden in self.hiddens:
            x = F.relu(hidden(x, edge_index), inplace=True)
            x = F.dropout(x, p=self.dropout_p, training=self.drop_mode)
        x = F.sigmoid(self.output(x, edge_index))
        return x.squeeze(1)


class GAT(nn.Module):
    def __init__(
            self, input_size, hidden_layer_sizes, output_size, dropout_p=0.0, alpha=0.95,
            use_residual=True, use_batch_norm=False, multiple_dropout=False, heads=1):
        super(GAT, self).__init__()

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.n_hidden = len(self.hidden_layer_sizes) - 1

        # Dropout
        self.dropout_p = dropout_p
        self.multiple_dropout = multiple_dropout
        # Special to GraphNet
        self.alpha = alpha
        # Special to GAT
        self.heads = heads

        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        if self.use_residual:
            self.block = functools.partial(
                _GATResidualBlock, use_batch_norm=self.use_batch_norm, heads=self.heads)
        else:
            self.block = functools.partial(
                _GATBasicBlock, use_batch_norm=self.use_batch_norm, heads=self.heads)

        self.input = nn.Linear(
            self.input_size, self.hidden_layer_sizes[0], bias=True)
        self.hiddens = nn.ModuleList([
            self.block(hidden_layer_sizes[h], hidden_layer_sizes[h+1])
            for h in range(self.n_hidden)
        ])
        self.output = self.block(hidden_layer_sizes[-1], output_size)

    @property
    def drop_mode(self):
        if self.multiple_dropout:
            return not self.training
        else:
            return self.training

    def create_edge_index(self, x):
        similarity_matrix = torch.abs(F.cosine_similarity(
            x[None, :, :], x[:, None, :], dim=-1))
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps
        row, col = torch.where(adj_matrix)
        edge_index = torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0)
        return edge_index

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.create_edge_index(x)

        x = F.relu(self.input(x), inplace=True)
        for hidden in self.hiddens:
            x = F.relu(hidden(x, edge_index), inplace=True)
            x = F.dropout(x, p=self.dropout_p, training=self.drop_mode)
        x = F.sigmoid(self.output(x, edge_index))
        return x.squeeze(1)

