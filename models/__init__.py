from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
]

MODELS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}


def get_model_names():
    return sorted(MODELS.keys())


def get_model(model_name, pretrain=True):
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {get_model_names()}"
        )

    return MODELS[model_name](pretrained=pretrain)