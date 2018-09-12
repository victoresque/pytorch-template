import torch.nn.functional as F

def get_loss_function(loss_fn_name):
    try:
        loss_fn = eval(loss_fn_name)
    except NameError as e:
        raise NameError(f"Loss function '{loss_fn_name}' not found.")

    return loss_fn

def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)
