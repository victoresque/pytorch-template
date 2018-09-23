from torch.nn.modules import loss


def get_loss_function(loss_name, *args, **kwargs):
    """Creates and returns instance of loss function class.

    All *args and **kwargs are passed to initialization of loss class.

    :param loss_name: Valid values of 'loss_name' are all class names inside 'torch.nn.modules.loss'.
        See 'https://pytorch.org/docs/stable/nn.html#loss-functions'.
    :return: Instance of loss function class.
    """
    # TODO: Implement functionality that scans for classes inside this file making them available.
    try:
        loss_fn = getattr(loss, loss_name)

    except AttributeError:
        raise AttributeError("Loss function '{}' not found.".format(loss_name))

    return loss_fn(*args, **kwargs)


class CustomLossClass(loss._Loss):
    """Your custom loss function implementation"""
    def __init__(self):
        super(CustomLossClass, self).__init__()

        # Your initialization goes here

        raise NotImplementedError

    def forward(self, output, target):
        # Forward calculation of custom loss function goes here.

        raise NotImplementedError

