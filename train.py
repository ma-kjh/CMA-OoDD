import argparse

import torch

from utils.common import setup_seed
from utils.load_model import set_model_clip
from utils.loader import train_loader
from utils.train_utils import train


def process_args():
    """Parses command-line arguments for fine-tuning a CLIP model.

    This function defines and parses command-line arguments required for 
    fine-tuning a CLIP model.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training CLIP Out-of-distribution Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ViT-B/16",
        help="Name of the CLIP model to use (e.g., ViT-B/16).",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default="openai",
        help="CLIP variant to use (e.g., openai, laion).",
    )
    parser.add_argument(
        "--ckpt", type=str, default="./", help="Path to the model checkpoint directory."
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="CMA",
        help="OOD detection method to apply (e.g., FLYP, CMA).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="imagenet",
        help="Benchmark dataset for evaluation.",
    )
    parser.add_argument(
        "--dir", type=str, default="./", help="Directory path for loading data."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--is-train", default=True, action="store_true", help="Enable training mode."
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        default="a photo of a",
        help="Text prompt",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--bs", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument(
        "--lam",
        type=float,
        default=0.0,
        help="Lambda regularization parameter for CMA.",
    )
    args = parser.parse_args()
    return args


def main():
    """Main function for training the CMA.
    """
    args = process_args()
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = set_model_clip(args)
    model.to(device)
    in_dataloader, texts_in = train_loader(args, preprocess, device)
    model.train()
    train(args, model, in_dataloader, texts_in, device)


if __name__ == "__main__":
    main()
