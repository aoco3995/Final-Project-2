import torch
from model import CNN


def get_model(log_path):

    model = CNN(5)
    model.load_state_dict(torch.load(log_path))
    model.eval()

    return model


if __name__ == '__main__':
    get_model()

