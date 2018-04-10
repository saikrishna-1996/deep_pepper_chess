import torch

def load_pretrained(model,fname):
    pretrained_state_dict = torch.load(fname)
    model_dict = model.state_dict()
    model_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    return model