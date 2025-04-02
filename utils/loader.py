import torch
import numpy as np
import clip
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
from prompt import Prompt_classes


def get_imagenet_dataset(args, preprocess, mode='train'):
    path = f'{args.dir}/imagenet_1k/{mode}'
    return datasets.ImageFolder(root=path, transform=preprocess)


def get_ood_dataset(args, preprocess, dataset_name):
    paths = {
        'iNaturalist': f'{args.dir}/MOS/iNaturalist',
        'SUN': f'{args.dir}/MOS/SUN',
        'Places': f'{args.dir}/MOS/Places',
        'dtd': f'{args.dir}/MOS/dtd/images'
    }
    return datasets.ImageFolder(root=paths[dataset_name], transform=preprocess)


def get_clip_texts(args, imagenet_classes, device=None):
    with torch.no_grad():
        if args.prompt_name == 'The nice':
            prompt = "The nice {}"
        elif args.prompt_name == 'a photo of a':
            prompt = "a photo of a {}"
        else:
            prompt = "{}"
        texts_in = clip.tokenize([prompt.format(c) for c in imagenet_classes])
        if device:
            texts_in = texts_in.to(device)
    return texts_in


def train_loader(args, preprocess, device=None):
    imagenet_classes, _ = Prompt_classes('imagenet')
    imagenet_train = get_imagenet_dataset(args, preprocess, 'train')
    in_dataloader = DataLoader(imagenet_train, shuffle=True, batch_size=args.bs, num_workers=32)

    texts_in = get_clip_texts(args, imagenet_classes)
    return in_dataloader, texts_in


def test_loader(args, preprocess, device):
    imagenet_classes, _ = Prompt_classes('imagenet')
    imagenet_test = get_imagenet_dataset(args, preprocess, 'val')
    ood_dataset = get_ood_dataset(args, preprocess, args.ood_dataset)

    in_dataloader = DataLoader(imagenet_test, shuffle=False, batch_size=args.bs, num_workers=32)
    out_dataloader = DataLoader(ood_dataset, shuffle=False, batch_size=args.bs, num_workers=32)

    texts_in = get_clip_texts(args, imagenet_classes, device)

    return in_dataloader, out_dataloader, texts_in


def test_loader_list_MOS(args, preprocess, device):
    imagenet_classes, _ = Prompt_classes('imagenet')
    imagenet_test = get_imagenet_dataset(args, preprocess, 'val')
    datasets_names = ['iNaturalist', 'SUN', 'Places', 'dtd']

    in_dataloader = DataLoader(imagenet_test, shuffle=False, batch_size=args.bs, num_workers=32)
    out_dataloader = [DataLoader(get_ood_dataset(args, preprocess, name), shuffle=False, batch_size=args.bs, num_workers=32)
                      for name in datasets_names]

    texts_in = get_clip_texts(args, imagenet_classes, device)

    return in_dataloader, out_dataloader, texts_in


class image_title_dataset(Dataset):
    def __init__(self, annotation='', transforms=None, args=None):
        self.transforms = transforms
        self.annotation = annotation
        self.args = args
        files = pd.read_csv(annotation, sep=' ', header=None)
        self.data = files[0]
        self.targets = files[1]

    def __len__(self):
        return len(self.data)

    def classes(self):
        return self.targets

    def __getitem__(self, idx):
        base_path = 'images_classic' if 'test_textures.txt' in self.annotation else 'images_largescale'
        img_path = f'{self.args.dir}/openood/{base_path}/{self.data[idx]}'
        image = self.transforms(Image.open(img_path).convert('RGB'))
        return image, self.targets[idx]


def test_loader_list_OpenOOD(args, preprocess, device):
    imagenet_classes, _ = Prompt_classes('imagenet')
    dataset_names = ['test_imagenet.txt', 'test_ssb_hard.txt', 'test_ninco.txt',
                     'test_inaturalist.txt', 'test_textures.txt', 'test_openimage_o.txt']

    loaders = [DataLoader(image_title_dataset(f'{args.dir}/openood/benchmark_imglist/imagenet/{name}', preprocess, args),
                          shuffle=False, batch_size=args.bs, num_workers=32) for name in dataset_names]

    in_dataloader = loaders[0]
    out_dataloader = loaders[1:]

    texts_in = get_clip_texts(args, imagenet_classes, device)

    return in_dataloader, out_dataloader, texts_in
