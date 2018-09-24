from torchvision import transforms


def get_composed_transforms(transform_list):
    """Create composed transforms according to configuration.

    :param transform_list: list of dictionaries, key 'op' is name of transform (from
        'torchvision.transforms', all other keys are handled as keyword arguments.
    :return: returns torch transforms.Compose object of all selected transforms.
    """
    trsfm_list = []

    for trsfm in transform_list:
        transform_name = trsfm.pop('op')
        fn_call = f"transforms.{transform_name}"

        kwargs = {}
        for k, v in trsfm.items():
            kwargs[k] = eval(v)

        try:
            trsfm_list.append(eval(fn_call)(**kwargs))
        except AttributeError:
            # TODO: replace with logging, later
            print(f"Transformation problem: transform '{transform_name}' not found. Check config.json")
            raise
        except TypeError:
            # TODO: replace with logging, later
            print(f"Transformation problem related to transform '{transform_name}'. Check config.json")
            raise

    return transforms.Compose(trsfm_list)

