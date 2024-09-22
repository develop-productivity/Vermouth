import os.path as osp
from torchvision import datasets
from. utils import DATASET_ROOT, get_classes_templates, TEMPLATE_JSON_PATH
from .dataset.objectnet import ObjectNetBase
from .dataset.imagenet import ImageNet as ImageNetBase
import json


class MNIST(datasets.MNIST):
    """Simple subclass to override the property"""
    class_to_idx = {str(i): i for i in range(10)}


def get_target_dataset(name: str, train=False, transform=None, target_transform=None):
    """Get the torchvision dataset that we want to use.
    If the dataset doesn't have a class_to_idx attribute, we add it.
    Also add a file-to-class map for evaluation
    """
    train = True if train == 'train' else False
    if name == "cifar10":
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, transform=transform,
                                   target_transform=target_transform, download=True)
    elif name == "stl10":
        dataset = datasets.STL10(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                 target_transform=target_transform, download=True)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    elif name == "pets":
        dataset = datasets.OxfordIIITPet(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                         target_transform=target_transform, download=True)

        # lower case every key in the class_to_idx
        dataset.class_to_idx = {k.lower(): v for k, v in dataset.class_to_idx.items()}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._images, dataset._labels)}
    elif name == "flowers":
        dataset = datasets.Flowers102(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                      target_transform=target_transform, download=True)
        classes = list(get_classes_templates('flowers')[0].keys())  # in correct order
        dataset.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)}
    elif name == "aircraft":
        dataset = datasets.FGVCAircraft(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                        target_transform=target_transform, download=True)

        # replace backslash with underscore -> need to be dirs
        dataset.class_to_idx = {
            k.replace('/', '_'): v
            for k, v in dataset.class_to_idx.items()
        }

        dataset.file_to_class = {
            fn.split("/")[-1].split(".")[0]: lab
            for fn, lab in zip(dataset._image_files, dataset._labels)
        }
        # dataset.file_to_class = {
        #     fn.split("/")[-1].split(".")[0]: lab
        #     for fn, lab in zip(dataset._image_files, dataset._labels)
        # }

    elif name == "food":
        dataset = datasets.Food101(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                   target_transform=target_transform, download=True)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    elif name == "eurosat":
        if train:
            raise ValueError("EuroSAT does not have a train split.")
        dataset = datasets.EuroSAT(root=DATASET_ROOT, transform=transform, target_transform=target_transform,
                                   download=True)
        dataset.file_to_class = {
            fn.split("/")[-1]: lab
            for fn, lab in dataset.samples
        }
    elif name == 'imagenet':
        # assert not train
        
        split = "train" if train else "val"
        # dataset = datasets.ImageNet(root=DATASET_ROOT, transform=transform, target_transform=target_transform,
        #                            download=True)
        base = ImageNetBase(transform, location=osp.join(DATASET_ROOT, 'ImageNet'))
        dataset = datasets.ImageFolder(root=osp.join(DATASET_ROOT, 'ImageNet', f"imagenet/{split}"), transform=transform,
                                       target_transform=target_transform)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(base.classnames)}  # {cls: i for i, cls in enumerate(base.classnames)}
        # dataset.class_to_idx = None  # {cls: i for i, cls in enumerate(base.classnames)}
        dataset.classes = base.classnames
        dataset.file_to_class = None
    elif name == 'objectnet':
        base = ObjectNetBase(transform, DATASET_ROOT)
        dataset = base.get_test_dataset()
        dataset.class_to_idx = dataset.label_map
        dataset.file_to_class = None  # todo
    elif name == "caltech101":
        if train:
            raise ValueError("Caltech101 does not have a train split.")
        dataset = datasets.Caltech101(root=DATASET_ROOT, target_type="category", transform=transform,
                                      target_transform=target_transform, download=True)

        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.categories)}
        dataset.file_to_class = {str(idx): dataset.y[idx] for idx in range(len(dataset))}
    elif name == "mnist":
        dataset = MNIST(root=DATASET_ROOT, train=train, transform=transform, target_transform=target_transform,
                        download=True)
    else:
        raise ValueError(f"Dataset {name} not supported.")

    if name in {'mnist', 'cifar10', 'stl10', 'aircraft'}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1]
            for idx in range(len(dataset))
        }

    assert hasattr(dataset, "class_to_idx"), f"Dataset {name} does not have a class_to_idx attribute."
    assert hasattr(dataset, "file_to_class"), f"Dataset {name} does not have a file_to_class attribute."
    return dataset

def get_prompts(dataset: str):
    assert dataset in ['oxford_pets', 'flowers', 'mnist', 'cifar10', 'food101', 'caltech101', 'imagenet','dtd', 'ucf101', 
                            'stanford_cars', 'aircraft', 'eurosat', 'sun397']
    with open(TEMPLATE_JSON_PATH, 'r') as f:
        all_templates = json.load(f)

    if dataset not in all_templates:
        raise NotImplementedError(f"Dataset {dataset} not implemented. Only {list(all_templates.keys())} are supported.")
    entry = all_templates[dataset]

    if "classes" not in entry:
        raise ValueError(f"Dataset {dataset} does not have a `classes` entry.")
    if "templates" not in entry:
        raise ValueError(f"Dataset {dataset} does not have a `templates` entry.")

    classes_dict, templates = entry["classes"], entry["templates"]

    # single template
    # template = templates[0]
    # prompts = [template.format(category)  category in classes_dict]

    prompts = []
    for category in classes_dict:
        prompt_per_category = [template.format(category)  for template in templates]
        prompts.append(prompt_per_category)
    return prompts


if __name__ == "__main__":
    prompts = get_prompts('caltech101')
    for prompt in prompts:
        print(prompt)