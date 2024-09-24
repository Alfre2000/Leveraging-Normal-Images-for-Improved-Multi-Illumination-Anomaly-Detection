import torch
from ad_types import Method
from architectures.rd4ad.de_resnet import de_wide_resnet50_2
from architectures.rd4ad.resnet import WideResNet, BN_layer


def load_model(model_path, method: Method, grouped: bool) -> torch.nn.Module:
    """
    Load the model from the given model path.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, bn = WideResNet(), BN_layer(3)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(model_path, map_location=device)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    decoder.eval()
    bn.eval()
    return encoder, bn, decoder
