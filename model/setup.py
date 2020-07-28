from .mobilenet import mobilenet_v2
# from .alexnet import alexnet
# from .vgg import vgg11

def setup_model(arch):
    arch = arch.lower()
    if arch == 'mobilenet':
        net = mobilenet_v2(pretrained=True, num_classes = 1)
    # elif arch == 'alexnet':
    #     net = alexnet(bitlength, pretrained=pretrained)
    # elif arch == 'vgg':
    #     net = vgg11(bitlength, pretrained=pretrained)
    else:
        raise ValueError('Unknown model type!', arch)

    return net