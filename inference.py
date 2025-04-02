import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip
from prompt import Prompt_classes
from utils.load_model import set_model_clip
from utils.loader import test_loader_list_MOS
from utils.metrics import get_measures

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(
    description="Evaluates CLIP Out-of-distribution Detection",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--models", type=str, default="ViT-B/16")
parser.add_argument("--clip", type=str, default="openai")
parser.add_argument("--ckpt", type=str, default="./")
parser.add_argument("--ood-dataset", type=str, default="iNaturalist")
parser.add_argument("--methods", type=str, default="flyp")
parser.add_argument("--benchmark", type=str, default="imagenet")
parser.add_argument("--prompt-name", type=str, default="The nice")
parser.add_argument("--dir", type=str, default="/data")
parser.add_argument("--bs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sim", type=int, default=1.0)
parser.add_argument("--is-train", default=False, action="store_true")
args = parser.parse_args()


# Load Model
print("model load !")
model, _, preprocess = set_model_clip(args)
in_dataloader, out_dataloader, texts_in = test_loader_list_MOS(args, preprocess, device)
imagenet_classes, _ = Prompt_classes("imagenet")
print("Prompt -The nice-")
texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes]).to(device)
model.to(device)
model = model.eval()
print("model load finished !")


def compute_image_features(dataloader, model):
    encoded_images = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.cuda()
            embeddings = model.module.encode_image(images).cpu()
            encoded_images.append(embeddings)
    features = torch.cat(encoded_images)
    return features / features.norm(dim=-1, keepdim=True)


def compute_logits(image_features, text_features):
    return image_features @ text_features.T


def concatenate_and_scale(logits, neg_logits, scale=0.01):
    return torch.cat([logits, neg_logits], dim=1) / scale


# Main logic starts here
imagenet_images_norm = compute_image_features(in_dataloader, model)

with torch.no_grad():
    imagenet_texts = model.module.encode_text(texts_in)
    imagenet_texts_cpu = imagenet_texts.cpu()
    n = np.load('./Neglabel/neg_label_10000.npy')
    neg_text = model.module.encode_text(clip.tokenize([i for i in n]).to(device))
imagenet_texts_cpu_norm = imagenet_texts_cpu / imagenet_texts_cpu.norm(dim=-1,keepdim=True)
neg_text = neg_text / neg_text.norm(dim=-1, keepdim=True)

# Compute logits
imagenet_logits = compute_logits(imagenet_images_norm, imagenet_texts_cpu_norm)

# Accuracy calculation
labels = torch.load("./CLIP_im1k_features/val/valtarget.pt")
acc = (imagenet_logits.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()).sum()
print(f"ACC : {acc / 50000} !")

# Compute features and logits for OOD datasets
ood_names = ["iNaturalist", "SUN", "Places", "Textures"]
ood_features = {}
for name, loader in zip(ood_names, out_dataloader):
    print(name)
    ood_features[name] = compute_image_features(loader, model)
ood_logits = {
    name: compute_logits(features, imagenet_texts_cpu_norm)
    for name, features in ood_features.items()
}

# Negative logits computation
neg_text_cpu = neg_text.cpu()
imagenet_neg_logits = compute_logits(imagenet_images_norm, neg_text_cpu)
ood_neg_logits = {
    name: compute_logits(features, neg_text_cpu)
    for name, features in ood_features.items()
}

# Concatenate logits with negative logits and compute softmax scores
to_np = lambda x : x.data.numpy()  

imagenet_random_logits_np = to_np(
    F.softmax(concatenate_and_scale(imagenet_logits, imagenet_neg_logits), dim=1)
)
ood_random_logits_np = {
    name: to_np(
        F.softmax(concatenate_and_scale(ood_logits[name], ood_neg_logits[name]), dim=1)
    )
    for name in ood_names
}

# Sum scores for original logits
imagenet_in_random_logits = (
    imagenet_random_logits_np[:, :1000].sum(axis=1).reshape(-1, 1)
)
ood_in_random_logits = {
    name: logits[:, :1000].sum(axis=1).reshape(-1, 1)
    for name, logits in ood_random_logits_np.items()
}

print("NegLabel !")
for name in ood_names:
    print(name, get_measures(imagenet_in_random_logits, ood_in_random_logits[name]))

# MCM computation
imagenet_logits_np = to_np(F.softmax(imagenet_logits, dim=1))
ood_logits_np = {
    name: to_np(F.softmax(logits, dim=1)) for name, logits in ood_logits.items()
}

MCM_imagenet = imagenet_logits_np.max(axis=1).reshape(-1, 1)
MCM_ood = {
    name: logits.max(axis=1).reshape(-1, 1) for name, logits in ood_logits_np.items()
}

print("MCM !")
for name in ood_names:
    print(name, get_measures(MCM_imagenet, MCM_ood[name]))
