import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from STAGATE_pyG.utils import Transfer_pytorch_Data
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from loss import NB_loss, ZINB_loss

cudnn.deterministic = True
cudnn.benchmark = True


def train_GAE(adata, model, n_epochs=500, lr=0.00025, key_added='SEGAE', loss_mode='mse',
              gradient_clipping=5.0, weight_decay=0.0001, verbose=True,
              random_seed=0, save_loss=True, save_reconstruction=False,
              device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    Train a Graph Autoencoder (GAE) model on spatial transcriptomics data.

    Args:
        adata (AnnData): Annotated data matrix.
        model (nn.Module): GAE model to be trained.
        n_epochs (int, optional): Number of training epochs. Defaults to 500.
        lr (float, optional): Learning rate. Defaults to 0.00025.
        key_added (str, optional): Key under which to add the results to adata. Defaults to 'SEGAE'.
        loss_mode (str, optional): Loss function to use ('mse', 'nb', 'zinb'). Defaults to 'mse'.
        gradient_clipping (float, optional): Maximum norm for gradient clipping. Defaults to 5.0.
        weight_decay (float, optional): Weight decay (L2 regularization). Defaults to 0.0001.
        verbose (bool, optional): Whether to print training progress. Defaults to True.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 0.
        save_loss (bool, optional): Whether to save the loss values during training. Defaults to True.
        save_reconstruction (bool, optional): Whether to save the reconstructed data. Defaults to False.
        device (torch.device, optional): Device to use for training. Defaults to CUDA if available.

    Returns:
        AnnData: Annotated data matrix with added results.
    """

    # Set random seeds for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    # Preprocess the data
    adata.X = sp.csr_matrix(adata.X)
    raw_sparse_data = sp.csr_matrix(adata.raw.X)
    raw_data = torch.Tensor(raw_sparse_data.toarray())

    if 'highly_variable' in adata.var.columns:
        adata_vars = adata[:, adata.var['highly_variable']]
    else:
        adata_vars = adata

    if verbose:
        print('Size of Input: ', adata_vars.shape)

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net does not exist! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_vars)
    scale_factor = torch.tensor(adata.obs['scale_factor'], dtype=torch.float)
    data.scale_factor = scale_factor.to(device)
    data.raw = raw_data.to(device)

    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    min_loss = float('inf')
    best_model_state = None
    model.train()
    loss_list = []

    loss_functions = {
        'mse': lambda: F.mse_loss(data.x, model(data.x, data.edge_index)[1]),
        'nb': lambda: NB_loss(data.raw, *model(data.x, data.edge_index, data.scale_factor)[:2], device),
        'zinb': lambda: ZINB_loss(data.raw, *model(data.x, data.edge_index, data.scale_factor)[:3], device),
    }

    if loss_mode not in loss_functions:
        raise ValueError("Invalid loss_mode. Choose from 'mse', 'nb', or 'zinb'.")

    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = loss_functions[loss_mode]()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        loss_list.append(loss.item())

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_model_state = model.state_dict().copy()

    print('min_loss:', min_loss)
    model.load_state_dict(best_model_state)

    # Plotting the training loss curve
    plt.plot(range(1, n_epochs + 1), loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    model.eval()
    if save_reconstruction:
        if loss_mode == 'mse':
            z, out = model(data.x, data.edge_index)
            ReX = out.squeeze(0).cpu().detach().numpy()
            ReX[ReX < 0] = 0
            rep = z.cpu().detach().numpy()
            adata.obsm[key_added] = rep
            adata.layers[key_added] = ReX
        elif loss_mode in ['nb', 'zinb']:
            mean, disp, pi, z = model(data.x, data.edge_index, data.scale_factor)
            ReX = mean.cpu().detach().numpy()
            ReX[ReX < 0] = 0
            rep = z.cpu().detach().numpy()
            adata.obsm[key_added] = rep
            adata.layers[key_added] = ReX

    if save_loss:
        adata.uns[key_added + '_loss'] = min_loss

    return adata


def train_DCA(model, inputs, raw_data, loss_fn, optimizer, num_epochs, device, scale_factor=torch.tensor(1.0)):
    """
    Train a Denoising Autoencoder (DCA) model.

    Args:
        model (nn.Module): The DCA model to be trained.
        inputs (torch.Tensor): Input data.
        raw_data (torch.Tensor): Raw data to compare against.
        loss_fn (str): Loss function to use ('MSELoss', 'NB_loss', 'ZINB_loss').
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to use for training.
        scale_factor (torch.Tensor, optional): Scale factor tensor. Defaults to 1.0.

    Returns:
        nn.Module: The trained model with the best loss.
    """
    model.to(device)
    min_loss = float('inf')
    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        inputs = inputs.to(device)
        raw_data = raw_data.to(device)
        scale_factor = scale_factor.to(device)

        optimizer.zero_grad()

        if loss_fn == 'MSELoss':
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs[0], raw_data)
        elif loss_fn == 'NB_loss':
            outputs = model(inputs, scale_factor)
            loss = NB_loss(raw_data, outputs[4], outputs[5], device)
        elif loss_fn == 'ZINB_loss':
            outputs = model(inputs, scale_factor)
            loss = ZINB_loss(raw_data, outputs[4], outputs[5], outputs[6], device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        average_loss = total_loss

        if average_loss < min_loss:
            min_loss = average_loss
            best_model_state = model.state_dict()

        losses.append(average_loss)

    print('Training finished.')
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    model.load_state_dict(best_model_state)
    return model