import os

import torch
from prompt import Prompt_classes
from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def train(args, model, in_dataloader, in_texts, device):
    save_acc = 0
    print(args)

    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    imagenet_classes, _ = Prompt_classes("imagenet")
    devices = list(range(torch.cuda.device_count()))

    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=devices)

    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}], lr=args.lr, weight_decay=0.1
    )

    num_batches = len(in_dataloader.dataset) // args.bs + 1
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs * num_batches,
        cycle_mult=1.0,
        max_lr=args.lr,
        min_lr=0,
        warmup_steps=500,
        gamma=1.0,
    )

    EPOCH = args.epochs

    for epoch in range(1, EPOCH + 1):
        model.train()

        for i, (batch) in enumerate(tqdm(in_dataloader)):
            optimizer.zero_grad()
            ground_truth_text = torch.arange(
                batch[0].shape[0], dtype=torch.long, device=device
            )

            images, labels = batch

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            images = images.to(device)
            texts = in_texts[labels]

            image_embeddings, text_embeddings, scale = model(images, texts)
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            )
            norm_text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )

            logits_per_image = scale[0] * image_embeddings @ norm_text_embeddings.T
            logits_per_text = logits_per_image.T

            image_loss = loss_img(logits_per_image, ground_truth_text)
            text_loss = loss_txt(logits_per_text, ground_truth_text)

            t_loss_data_text = -torch.logsumexp(logits_per_text, dim=1)
            t_loss_data_image = -torch.logsumexp(logits_per_image, dim=1)

            total_loss = (text_loss + args.lam * t_loss_data_text.mean()) / 2 + (
                            image_loss + args.lam * t_loss_data_image.mean()) / 2

            total_loss.backward()
            optimizer.step()
            scheduler.step()

        save_name = f"./"

        if not os.path.exists(save_name):
            os.makedirs(save_name)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"{save_name}/model_{args.methods}_{epoch}.pt",
        )
