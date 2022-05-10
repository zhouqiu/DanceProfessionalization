import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)  # batch size & channels
        # print(b)
        # print(c)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def ZeroPad1d(sizes):
    return nn.ConstantPad1d(sizes, 0)


def ConvLayers(kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', use_bias=True):

    """
    returns a list of [pad, conv] => should be += to some list, then apply sequential
    """

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = ZeroPad1d
    else:
        # assert 0, "Unsupported padding type: {}".format(pad_type)
        return [nn.Conv1d(in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride, bias=use_bias)]


    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return [pad((pad_l, pad_r)),
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride, bias=use_bias)]



def get_acti_layer(acti='relu', inplace=True):

    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'sigmoid':
        return [nn.Sigmoid()]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None):

    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'in':
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'adain':
        return [AdaptiveInstanceNorm1d(norm_dim)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


def ConvBlock(kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', dropout=None,
              norm='none', acti='lrelu', acti_first=False, use_bias=True, inplace=True):
    """
    returns a list of [pad, conv, norm, acti] or [acti, pad, conv, norm]
    """
    # print(norm)
    layers = ConvLayers(kernel_size, in_channels, out_channels, stride=stride, pad_type=pad_type, use_bias=use_bias)
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_channels)
    acti_layers = get_acti_layer(acti, inplace=inplace)

    if acti_first:
        return acti_layers + layers
    else:
        return layers + acti_layers




def LinearBlock(in_dim, out_dim, dropout=None, norm='none', acti='relu'):

    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers





def acti_layer(acti='relu', inplace=True):

    if acti == 'relu':
        return nn.ReLU(inplace=inplace)
    elif acti == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif acti == 'tanh':
        return nn.Tanh()
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def norm_layer(norm='none', norm_dim=None):

    if norm == 'bn':
        return nn.BatchNorm1d(norm_dim)
    elif norm == 'in':
        return nn.InstanceNorm1d(norm_dim, affine=True)
    elif norm == 'adain':
        return AdaptiveInstanceNorm1d(norm_dim)
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
