import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from STAGATE_pyG.gat_conv import GATConv


class Flatten(nn.Module):
    """
    Flatten layer to reshape the input tensor to a 2D tensor (batch_size, -1).
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


def mean_act(x):
    """
    Activation function for mean predictions.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Activated tensor with values clamped between 1e-5 and 1e6.
    """
    return torch.clamp(torch.exp(x), 1e-5, 1e6)


def disp_act(x):
    """
    Activation function for dispersion predictions.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Activated tensor with values clamped between 1e-4 and 1e4.
    """
    return torch.clamp(F.softplus(x), 1e-4, 1e4)


def pi_act(x):
    """
    Activation function for zero-inflation probability.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Activated tensor with values between 0 and 1.
    """
    return torch.sigmoid(x)


class GCALayer(nn.Module):
    """
    Graph Channel Attention (GCA) layer.

    Args:
        gate_channels (int): Number of channels for the gate.
        reduction_ratio (int): Reduction ratio for the channel attention mechanism.
        pool_types (list): Types of pooling operations to apply ('avg', 'max').
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(GCALayer, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = torch.mean(x, dim=0, keepdim=True)
            elif pool_type == 'max':
                pool = torch.max(x, dim=0, keepdim=True)[0]

            channel_att_raw = self.mlp(pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).expand_as(x)
        return x * scale


class DUSTED(nn.Module):
    """
    DUSTED: A dual-attention spatial transcriptomics enhanced denoiser model.

    Args:
        hidden_dims (list): List of dimensions for input, hidden, and output layers.
    """
    def __init__(self, hidden_dims):
        super(DUSTED, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.disp = GATConv(num_hidden, in_dim, heads=1, concat=False,
                            dropout=0, add_self_loops=False, bias=False)
        self.mean = GATConv(num_hidden, in_dim, heads=1, concat=False,
                            dropout=0, add_self_loops=False, bias=False)
        self.pi = GATConv(num_hidden, in_dim, heads=1, concat=False,
                          dropout=0, add_self_loops=False, bias=False)
        self.gca1 = GCALayer(in_dim)

    def forward(self, features, edge_index, scale_factor, alpha=1.5):
        h1 = self.gca1(features)
        h1 = alpha * h1 + features
        h1 = F.elu(self.conv1(h1, edge_index))
        h2 = self.conv2(h1, edge_index)

        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)

        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        pi = pi_act(self.pi(h3, edge_index))
        disp = disp_act(self.disp(h3, edge_index))
        mean = mean_act(self.mean(h3, edge_index))
        mean = (mean.T * scale_factor).T

        return mean, disp, pi, h2


class AE(nn.Module):
    """
    Autoencoder (AE) model for feature extraction and dimensionality reduction.

    Args:
        n_enc_1 (int): Number of units in the first encoder layer.
        n_enc_2 (int): Number of units in the second encoder layer.
        n_enc_3 (int): Number of units in the third encoder layer.
        n_dec_1 (int): Number of units in the first decoder layer.
        n_dec_2 (int): Number of units in the second decoder layer.
        n_dec_3 (int): Number of units in the third decoder layer.
        n_input (int): Number of input features.
        n_z (int): Number of latent variables.
    """
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        
        # Encoder layers
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)
        
        # Decoder layers
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        # Encoding
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        # Decoding
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class GAE(nn.Module):
    """
    Graph Autoencoder (GAE) model for graph-based feature extraction.

    Args:
        hidden_dims (list): List of dimensions for input, hidden, and output layers.
    """
    def __init__(self, hidden_dims):
        super(GAE, self).__init__()
        in_dim, num_hidden, out_dim = hidden_dims

        # Graph convolutional layers
        self.conv1 = GCNConv(in_dim, num_hidden, cached=True)
        self.conv2 = GCNConv(num_hidden, out_dim, cached=True)
        self.conv3 = GCNConv(out_dim, num_hidden, cached=True)
        self.conv4 = GCNConv(num_hidden, in_dim, cached=True)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = F.elu(self.conv3(h2, edge_index))
        h4 = self.conv4(h3, edge_index)

        return h2, h4


class DCA_ZINB(nn.Module):
    """
    Denoising Autoencoder (DCA) with ZINB output layer.

    Args:
        n_input (int): Number of input features.
        n_z (int): Number of latent variables.
        n_enc_1 (int): Number of units in the first encoder layer.
        n_enc_2 (int): Number of units in the second encoder layer.
        n_enc_3 (int): Number of units in the third encoder layer.
        n_dec_1 (int): Number of units in the first decoder layer.
        n_dec_2 (int): Number of units in the second decoder layer.
        n_dec_3 (int): Number of units in the third decoder layer.
        enc_dropout (float, optional): Dropout rate for encoder layers. Defaults to 0.5.
        dec_dropout (float, optional): Dropout rate for decoder layers. Defaults to 0.5.
    """
    def __init__(self, n_input, n_z, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, enc_dropout=0.5, dec_dropout=0.5):
        super(DCA_ZINB, self).__init__()

        # Encoder layers
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        # Decoder layers
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)

        # ZINB output layers
        self.pi = nn.Linear(n_dec_3, n_input)
        self.disp = nn.Linear(n_dec_3, n_input)
        self.mean = nn.Linear(n_dec_3, n_input)

    def forward(self, x, scale_factor):
        # Encoding
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        # Decoding
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))

        # ZINB parameters
        pi = pi_act(self.pi(dec_h3))
        disp = disp_act(self.disp(dec_h3))
        mean = mean_act(self.mean(dec_h3))
        mean = (mean.T * scale_factor).T

        return enc_h1, enc_h2, enc_h3, z, mean, disp, pi


class DCA_NB(nn.Module):
    """
    Denoising Autoencoder (DCA) with NB output layer.

    Args:
        n_input (int): Number of input features.
        n_z (int): Number of latent variables.
        n_enc_1 (int): Number of units in the first encoder layer.
        n_enc_2 (int): Number of units in the second encoder layer.
        n_enc_3 (int): Number of units in the third encoder layer.
        n_dec_1 (int): Number of units in the first decoder layer.
        n_dec_2 (int): Number of units in the second decoder layer.
        n_dec_3 (int): Number of units in the third decoder layer.
        enc_dropout (float, optional): Dropout rate for encoder layers. Defaults to 0.5.
        dec_dropout (float, optional): Dropout rate for decoder layers. Defaults to 0.5.
    """
    def __init__(self, n_input, n_z, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, enc_dropout=0.5, dec_dropout=0.5):
        super(DCA_NB, self).__init__()

        # Encoder layers
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        # Decoder layers
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)

        # NB output layers
        self.disp = nn.Linear(n_dec_3, n_input)
        self.mean = nn.Linear(n_dec_3, n_input)

    def forward(self, x, scale_factor):
        # Encoding
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        # Decoding
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))

        # NB parameters
        disp = disp_act(self.disp(dec_h3))
        mean = mean_act(self.mean(dec_h3))
        mean = (mean.T * scale_factor).T

        return enc_h1, enc_h2, enc_h3, z, mean, disp