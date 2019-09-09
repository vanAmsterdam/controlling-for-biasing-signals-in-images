"""Defines the neural network, losss function and metrics"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import models
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import io

# from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4
# from model.gresnet import ResNet, BasicBlock, SResNet

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class DeepConcatRegressor(nn.Module):
    """
    Class for deeper regression tails
    """
    def __init__(self, params):
        super(DeepConcatRegressor, self).__init__()
        self.params = params

        # fcs
        self.fcs = nn.ModuleList([])
        if fc_in in params.tail_fcs:
            self.fcs.append(nn.ModuleList[
                nn.Linear()
                ])

class ConstantPad(nn.Module):
    """
    Module for concatenating a constant to a dimension
    """
    def __init__(self, params, in_shape, padding_dim=1):
        super(ConstantPad, self).__init__()
        self.padding_dim = padding_dim
        self.ones = torch.ones((1,) + in_shape, requires_grad=False).to(params.device)

    def forward(self, x, t):
        t_pad = t * self.ones
        # print(x.shape)
        # print(t_pad.shape)

        return torch.cat([x, t_pad], self.padding_dim)

class ConcatRegressor(nn.Module):
    """
    Module for concatenating feature info in final layer
    Always includes conditioning on t, optionally on x
    """
    def __init__(self, in_features=144, concat_dim=1):
        super(ConcatRegressor, self).__init__()
        self.fc = nn.Linear(in_features, 1)
        self.t  = nn.Linear(concat_dim, 1, bias=False)
        nn.init.constant_(self.t.weight, 0)

    def forward(self, x, t):
        return self.fc(x) + self.t(t)


class SqueezeInteractionRegressor(nn.Module):
    """
    Squeeze of img to single dimension (by setting params.regressor_z_dim=1), 
    do interaction on z and zt
    Ignore x for interaction
    """
    def __init__(self, in_features=None):
        # note in_features are not needed, as per definition they are 2
        # keep for compatibility of instantion call

        super(SqueezeInteractionRegressor, self).__init__()
        self.fc   = nn.Linear(2,1)
        self.t    = nn.Linear(1,1, bias=False)
        self.zt   = nn.Linear(1,1, bias=False)

    def forward(self, x, t):
        assert x.shape[1] == 2, "Use SqueezeInteractionRegressor only when x.shape[1] == 2 (so img is x and z)"
        y = self.fc(x) # fc of complete image + bias (gives b_x, b_z, intercept)
        y += self.t(t) # average treatment effect (gives b_t, no intercept)

        # plit off z (the not-x part of the image)
        z  = x[:,1:]        
        y += self.zt(z*t) # treatment interaction (gives b_zt, no intercept)

        return y

class SplitInteractionRegressor(nn.Module):
    """
    Add interaction term to final layer

    Split off the last half channels of z to model interaction
    e.g. 
    
    - fc-layer based on (x, z1, z2) -> y (including bias term)
    - addition of treatment effect t
    - addition of interaction: y += ((z3, z4) * t).sum(1)



    """
    def __init__(self, in_features=144):
        super(SplitInteractionRegressor, self).__init__()
        self.fc_in = torch.tensor(1 + (in_features - 1) / 2).int() # x + half of z
        self.fc = nn.Linear(self.fc_in, 1)
        self.t  = nn.Linear(1,1, bias=False)
        nn.init.constant_(self.t.weight, 0)

    def forward(self, x, t):
        y = self.fc(x[:,:self.fc_in])        # prognostic factors (also based on 'x' node)
        y += self.t(t)                     # average treatment effect
        y += (t * x[:,self.fc_in:]).sum(1).view(-1,1)    # addition of interaction between later nodes of z and treatment
        
        return y


class BaseRegressor(nn.Module):
    """
    Module for NOT concatenating feature info in final layer (but includes optional t argument for forward pass)
    """
    def __init__(self, in_features=144):
        super(BaseRegressor, self).__init__()
        self.fc = nn.Linear(in_features, 1, bias=True)

    def forward(self, x, t=None):
        return self.fc(x)


class LinearBiasConditioning(nn.Module):
    """
    Add a linear function of something to condition on (e.g. treatment) as a 
    the bias to a layer
    """
    def __init__(self, shape, in_dim=1, params=None):
        """
        shape: shape of output layer to append to
        function get's added to the len(shape) + 1'th dimension of the input (e.g. (batch_dim, x, y, FEATURE))
        """
        super(LinearConditioning, self).__init__()

        self.params = params
        self.fwd    = nn.Linear(in_dim, np.prod(shape)).view(shape)

    def forward(self, x):
        return self.fwd(x)

class MINE(nn.Module):
    """
    Code contributed by Maximilian Ilse
    Statistics network to calculate mutual information (https://arxiv.org/abs/1801.04062)
    """
    def __init__(self, in_size, h_size=None, min_h_size=250):
        '''
        in_size:    size of tensors that come in
        h_size:     size of hidden layer, if None: min(2*in_size, min_h_size)
        min_h_size: minimal size of hidden
        
        '''
        super(MINE, self).__init__()
        if h_size is None:
            h_size = max(min_h_size, int(2*in_size))

        self.T = nn.Sequential(
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, 1))

        for module in self.T.modules():
            if hasattr(module, 'weight'):
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x, y):
        y_shuffled = y[torch.randperm(y.shape[0])]

        T_joint    = self.T(torch.cat((x, y), dim=1))
        T_marginal = self.T(torch.cat((x, y_shuffled), dim=1))

        return T_joint, T_marginal

    def lower_bound(self, x, y):
        T_joint, T_marginal = self.forward(x, y)
        # mine = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)) + 1e-6)
        mine = torch.mean(T_joint) - (torch.logsumexp(T_marginal, dim=0) - torch.tensor(T_marginal.shape[0]).float().log())

        return -mine.squeeze()

class JSD(nn.Module):
    """
    estimate mutual information with jensen-shannon 
    """
    def __init__(self, in_size, h_size=None, min_h_size=250):
        '''
        in_size:    size of tensors that come in
        h_size:     size of hidden layer, if None: min(2*in_size, min_h_size)
        min_h_size: minimal size of hidden
        
        '''
        super(JSD, self).__init__()
        if h_size is None:
            h_size = max(min_h_size, int(2*in_size))

        self.T = nn.Sequential(
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, 1))

        for module in self.T.modules():
            if hasattr(module, 'weight'):
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x, y):
        y_shuffled = y[torch.randperm(y.shape[0])]

        T_joint    = self.T(torch.cat((x, y), dim=1))
        T_marginal = self.T(torch.cat((x, y_shuffled), dim=1))

        return T_joint, T_marginal

    def lower_bound(self, x, y):
        T_joint, T_marginal = self.forward(x, y)
        mi_hat = -F.softplus(-T_joint).mean() - F.softplus(T_marginal).mean()
        # return -mi_hat.squeeze()
        return mi_hat.squeeze()

class SimpleEncoder(nn.Module):
    def __init__(self, params, setting):
        super(SimpleEncoder,self).__init__()
        self.params = params
        self.setting =  setting

        self.fwd = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1,1)),
            Flatten()
        )

    def forward(self, x, t=None):
        return self.fwd(x)
        
encoders = {'simple': SimpleEncoder}

class CausalNet(nn.Module):
    def __init__(self, params, setting):
        super(CausalNet, self).__init__()
        self.params  = params
        self.setting = setting

        # storage for betas from OLS
        # keep in model to port from train to valid
        self.betas_bias   = torch.zeros((params.regressor_z_dim+2,1), requires_grad=False) 
        self.betas_causal = torch.zeros((params.regressor_z_dim+1,1), requires_grad=False)

        print("instantiating net")

        self.encoder = encoders[setting.encoder](params, setting)
        if setting.encoder == 'simple':
            fc_in_features = 144
        else:
            raise NotImplementedError(f'different encoder than simple currently not implemented: {setting.encoder})')

        # pick the right type of regressor, possibly allowing for interactions
        if params.conditioning_place == "regressor":
            if params.train_interaction:
                if params.regressor_z_dim==1:
                    Regressor = SqueezeInteractionRegressor
                else:
                    Regressor = SplitInteractionRegressor
            else:
                Regressor = ConcatRegressor
        else:
            raise NotImplementedError('only conditioning in final layer is implemented now; check earlier versions of code')

        # same size in and out fcs
        self.fcs = nn.ModuleList(params.num_fc*[
            nn.Linear(fc_in_features, fc_in_features), 
            nn.ReLU(inplace=True),
            nn.Dropout(params.dropout_rate)
            ])

        # fc layer to final regression layer
        # NOTE keep track if a ReLU is needed here (probably not)
        self.fcr = nn.Linear(fc_in_features, params.regressor_z_dim + params.regressor_x_dim)

        # final regressor to y; this takes in entire last layer and treatment
        self.regressor = Regressor(params.regressor_z_dim+params.regressor_x_dim, concat_dim=1)

        # initialize weights
        for layer_group in [self.encoder, self.fcs, self.fcr, self.regressor]:
            for module in layer_group.modules():
                if hasattr(module, 'weight'):
                    torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x, t=None, epoch=None):
        # prepare dictionary for keeping track of output tensors
        outs = {}

        # convolutional stage to get 'features'
        h = self.encoder(x)

        # pass through a sequence of same-size in-out fc-layers for 'non-linear interactions'
        for i, module in enumerate(self.fcs):
            h = module(h)

        # squeeze to lower size for final regression layer
        finalactivations = self.fcr(h)

        # store tensors ('bottlenecks' from which correlations / MIs are calculated)
        outs['bnx'] = finalactivations[:,:self.params.regressor_x_dim] # activations that represent x
        outs['bnz'] = finalactivations[:,self.params.regressor_x_dim:] # activations that represent z (=everything else)

        # predict y from final activations and treatment
        outs['y'] = self.regressor(finalactivations, t)

        return outs
 
def freeze_conv_layers(model, keep_layers = ["bnx", "bny", "bnbnx", "bnbny", "fcx", "fcy", "t"], last_frozen_layer=None):
    for name, param in model.named_parameters():
        if name.split(".")[0] not in keep_layers:
            param.requires_grad = False
        else:
            print("keeping grad on for parameter {}".format(name))

def speedup_t(model, params):
    lr_t = params.lr_t_factor * params.learning_rate
    optimizer = torch.optim.Adam(model.regressor.t.parameters(), lr = lr_t)
    if params.speedup_intercept:
        optimizer.add_param_group({'params': model.regressor.fc.bias, 'lr': lr_t})

    for name, param in model.named_parameters():
        # print(f"parameter name: {name}")
        if name.split(".")[1] == "t":
            print("Using custom lr for param: {}".format(name))
        elif name.endswith("fc.bias") and params.speedup_intercept:
            print("Using cudtom lr for param: {}".format(name))
        else:
            optimizer.add_param_group({'params': param, 'lr': params.learning_rate, 'weight_decay': params.wd})
    return optimizer


def softfreeze_conv_layers(model, params, fast_layers = ["bnx", "bny", "bnbnx", "bnbny", "fcx", "fcy"], last_frozen_layer=None):
    optimizer = torch.optim.Adam(model.t.parameters(), lr=params.learning_rate)
    for name, param in model.named_parameters():
        if name in fast_layers:
            optimizer.add_param_group({'params': param})
        elif name.split(".")[0] == "t":
            pass
        else:
            optimizer.add_param_group({'params': param, 'lr': params.learning_rate / 10})

    return optimizer

def get_loss_fn(setting, **kwargs):
    if setting.num_classes == 2:
        print("Loss: cross-entropy")
        def loss_fn(outputs, labels, **kwargs):
            criterion = nn.CrossEntropyLoss(**kwargs)
            target = labels.type(torch.cuda.LongTensor)
            # print(target.size())
            # print(outputs.size())
            return criterion(outputs, target)
    else: 
        print("Loss: MSE")
        def loss_fn(outputs, labels, **kwargs):
            criterion = nn.MSELoss()
            # return torch.sqrt(criterion(outputs.squeeze(), labels.squeeze()))
            return criterion(outputs.squeeze(), labels.squeeze())
    return loss_fn

def bottleneck_loss(bottleneck_features):
    z_mean    = bottleneck_features, outputs, labels
    z_stddev  = bottleneck_features, outputs, labels
    mean_sq   = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev, outputs, labels
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq + 1.0e-6) - 1)

def get_bn_loss_fn(params):
    if params.bn_loss_type == "variational-gaussian":
        def loss_fn(outputs):
            # take mean and sd over batch dimension
            z_mean    = outputs.mean(0)
            z_stddev  = outputs.std(0)
            mean_sq   = z_mean * z_mean
            stddev_sq = z_stddev * z_stddev
            return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq + 1.0e-6) - 1)
    else:
        raise NotImplementedError
    
    return loss_fn

def rmse(setting, model, outputs, labels, data=None):
    return np.sqrt(np.mean(np.power((outputs - labels), 2)))

def bias(setting, model, outputs, labels, data=None):
    weights = model.t.weight.detach().cpu().numpy()
    return np.squeeze(weights)[-1] - 1

def b_t(setting, model, outputs, labels, data=None):
    weight = model.regressor.t.weight.detach().cpu().numpy()
    return weight

def intercept(setting, model, outputs, labels, data=None):
    # oracle = pd.read_csv(os.path.join(setting.data_dir, "oracle.csv"))
    bias = model.cnn.fc2.bias.detach().cpu().numpy()
    return bias
    # for now: use ATE = 1

def ate(setting, model, outputs, labels, data):
    # data should always have treatment in first columns
    if data.ndim == 1:
        t = data
    else:
        t = data[:,0].squeeze()

    treated   = outputs[np.where(t)]
    untreated = outputs[np.where(t == 0)]

    return treated.mean() - untreated.mean()

# def ite(setting, model, outputs, labels, data):
#     assert data.shape[1] == 3
#     y0 = data[:,1]
#     y1 = data[:,2]
#     return y1 - y0

# def ite_mean(setting, model, outputs, labels, data):
#     ites = ite(setting, model, outputs, labels, data)
#     return ites.mean()

def total_loss(setting, model, outputs, labels, data=None):
    # total_loss_fn = get_loss_fn(setting, reduction="sum")
    total_loss_fn = nn.MSELoss(reduction="sum")
    outputs = torch.tensor(outputs, requires_grad=False).squeeze()
    labels  = torch.tensor(labels, requires_grad=False).squeeze()
    return total_loss_fn(outputs, labels)




# def bias(model, outputs, labels):
#     if "fc1" in list(model.children()):
#         weights = model.fc1.weight.detach().cpu().numpy()
#         return weights[0,-1]
#     else:
#         return np.zeros((1,))


def accuracy(setting, model, outputs, labels, data=None):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def ppv(setting, model, outputs, labels, data=None):
    if setting.num_classes == 2:
        pos_preds = np.argmax(outputs, axis=1)==1
        if pos_preds.sum() > 0:
            return accuracy(setting, model, outputs[pos_preds,:], labels[pos_preds])
        else:
            return np.nan
    else:
        return 0.

def npv(setting, model, outputs, labels, data=None):
    if setting.num_classes == 2:
        neg_preds = np.argmax(outputs, axis=1)==0
        if neg_preds.sum() > 0:
            return accuracy(setting, model, outputs[neg_preds,:], labels[neg_preds])
        else:
            return np.nan
    else:
        return 0.

def cholesky_least_squares(X, Y, intercept=True):
    """
    Perform least squares regression with cholesky decomposition 
    intercept: add intercept to X
    adapted from https://gist.github.com/gngdb/611d8f180ef0f0baddaa539e29a4200e
    which was adapted from http://drsfenner.org/blog/2015/12/three-paths-to-least-squares-linear-regression/
    """
    if X.ndimension() == 1:
        X.unsqueeze_(1)    
    if intercept:
        X = torch.cat([torch.ones_like(X[:,0].unsqueeze(1)),X], dim=1)
    
    XtX, XtY = X.permute(1,0).mm(X), X.permute(1,0).mm(Y)
    betas, _ = torch.gesv(XtY, XtX)

    return betas.squeeze()

def mse_loss(output, target):
    criterion = nn.MSELoss()
    return criterion(output, target)

def spearmanrho(outputs, labels):
    '''
    calculate spearman (non-parametric) rank statistic
    '''
    try:
        return spearmanr(outputs.squeeze(), labels.squeeze())[0]
    except ValueError:
        print('value error in spearmanr, returning 0')
        return np.array(0)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
all_metrics = {
    'total_loss': total_loss,
    'bottleneck_loss': bottleneck_loss,
    'accuracy': accuracy,
    'rmse': rmse,
    'bias': bias,
    'ate': ate,
    'intercept': intercept,
    'b_t': b_t,
    'ppv': ppv,
    'npv': npv,
    'spearmanrho': spearmanrho
    # 'ite_mean': ite_mean
    # could add more metrics such as accuracy for each token type
}

# from here: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def get_of_diag(x):
    '''
    Set the diagonal elements of a matrix to zero, and flatten the rest
    '''
    assert type(x) is np.ndarray
    
    x = x[~np.eye(x.shape[0],dtype=bool)]
    return x.reshape(-1,1)

def make_scatter_plot(x,y,c=None,
                      xlabel: str=None,ylabel: str=None,title: str=None):
    '''
    make scatter plots for tensorboard
    '''
    if c is not None:
        g = sns.jointplot(x.reshape(-1,1),y.reshape(-1,1), kind='reg')
        # g = sns.jointplot(x.reshape(-1,1),y.reshape(-1,1), joint_kws=dict(scatter_kws=dict(c=c.reshape(-1,1))), kind='reg')
    else:
        g = sns.jointplot(x.reshape(-1,1),y.reshape(-1,1), kind='reg')
    g.set_axis_labels(xlabel, ylabel)
    g.ax_joint.set_title(xlabel+ " vs " + ylabel)
    return g.fig    
    # fig = plt.figure()
    # plt.scatter(x,y,c=c)
    # if xlabel is not None:
    #     plt.xlabel(xlabel)
    # if ylabel is not None:
    #     plt.ylabel(ylabel)
    # if title is not None:
    #     plt.title(title)
    # else:
    #     plt.title(xlabel+' vs '+ylabel)
    # return fig
