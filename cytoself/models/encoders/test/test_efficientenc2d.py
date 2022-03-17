from torch import nn
from torchvision.ops.misc import ConvNormActivation
from cytoself.models.encoders.efficientenc2d import efficientenc_b0, MBConv


block_args = [
    {'expand_ratio': 1, 'kernel': 3, 'stride': 1, 'input_channels': 32, 'out_channels': 16, 'num_layers': 1},
    {'expand_ratio': 6, 'kernel': 3, 'stride': 2, 'input_channels': 16, 'out_channels': 24, 'num_layers': 2},
]


def test_efficientenc_b0_default():
    model = efficientenc_b0(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)


def test_efficientenc_b0_channels():
    model = efficientenc_b0(block_args, in_channels=2, out_channels=64)
    assert len(model.features._modules) == len(block_args) + 2
    assert isinstance(model.features[0], ConvNormActivation)
    assert isinstance(model.features[-1], ConvNormActivation)
    assert model.features[0][0].in_channels == 2
    assert model.features[-1][0].out_channels == 64