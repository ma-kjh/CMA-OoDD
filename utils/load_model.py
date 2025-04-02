import clip

import torch
from torch import nn
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    RandomResizedCrop,
)


def _convert_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px: int, is_train: bool):
    normalize = Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    if is_train:
        return Compose(
            [
                RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
    else:
        return Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )


def set_model_clip(args):
    preprocess = _transform(224, args.is_train)
    model = None
    classification_head = None
    if args.is_train:
        model, _ = clip.load(args.models, device="cpu")
        if args.ckpt == "./":
            pass
        elif args.multiprocessing_distributed:
            torch.cuda.set_device(args.rank)
            model = model.cuda(args.rank)
            model = nn.parallel.DistributedDataParallel(model, deivce_idx=[args.rank])
        else:
            model = nn.DataParallel(model)
            checkpoint = torch.load(f"{args.ckpt}")
            model.load_state_dict(checkpoint["model_state_dict"])

        return model, preprocess
    else:
        model, _ = clip.load(args.models, device="cpu")
        if args.ckpt != "./":
            model = nn.DataParallel(model)
            checkpoint = torch.load(f"{args.ckpt}")
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = nn.DataParallel(model)

        return model, classification_head, preprocess
