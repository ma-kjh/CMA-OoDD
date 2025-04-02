# [CVPR 2025] Enhanced OoD Detection through Cross-Modal Alignment of Multi-modal Representations

This repository contains the official implementation of CMA.

> **Enhanced OoD Detection through Cross-Modal Alignment of Multi-modal Representations**  
> [Jeonghyeon Kim](https://scholar.google.co.kr/citations?user=u6DjYLsAAAAJ&hl=ko),  [Sangheum Hwang](https://scholar.google.co.kr/citations?user=QtI8XmgAAAAJ&hl=ko)

📄 **Paper**: \[[arXiv](https://arxiv.org/abs/2503.18817)\] \[[Project Page](https://ma-kjh.github.io/CMA-OoDD-Project/)\]  

<p align="center">
    <img alt="CMA Overview" src="https://github.com/ma-kjh/CMA-OoDD/blob/main/main.jpg" width="50%"/>
</p>

## Abstract
_Prior research on out-of-distribution detection (OoDD) has primarily focused on single-modality models. Recently, with the advent of large-scale pretrained vision-language models such as CLIP, OoDD methods utilizing such multi-modal representations through zero-shot and prompt learning strategies have emerged. However, these methods typically involve either freezing the pretrained weights or only partially tuning them, which can be suboptimal for downstream datasets. In this paper, we highlight that **multi-modal fine-tuning (MMFT) can achieve notable OoDD performance**. Despite some recent works demonstrating the impact of fine-tuning methods for OoDD, there remains significant potential for performance improvement. We investigate the limitation of na\"ive fine-tuning methods, examining why they fail to fully leverage the pretrained knowledge. Our empirical analysis suggests that this issue could stem from the modality gap within in-distribution (ID) embeddings. To address this, **we propose a training objective that enhances cross-modal alignment by regularizing the distances between image and text embeddings of ID data.** This adjustment helps in better utilizing pretrained textual information by aligning similar semantics from different modalities (i.e., text and image) more closely in the hyperspherical representation space. **We theoretically demonstrate that the proposed regularization corresponds to the maximum likelihood estimation of an energy-based model on a hypersphere.** Utilizing ImageNet-1k OoD benchmark datasets, we show that our method, combined with post-hoc OoDD approaches leveraging pretrained knowledge (e.g., NegLabel), significantly outperforms existing methods, **achieving state-of-the-art OoDD performance and leading ID accuracy.**_

---

## 🔧 Requirements

Ensure the following dependencies are installed:

```
conda env create -f environment.yaml -n cma
```
 - **Python**: 3.9.18
 - **Torch**: 1.12.0+cu116
 - **Torchvision**: 0.13.0+cu116
 - **Numpy**: 1.25.2
 - **Scikit-learn**: 1.4.2
---

## 📂 Dataset Preparation

### In-Distribution (ID) Datasets

For fine-tuning and evaluation on the **ImageNet-1k** setting, ensure datasets are structured in `<data_dir>` as follows:

```
data_dir/
│── imagenet_1k/
│   ├── train/
│   │   ├── ...
│   ├── val/
│   │   ├── ...
```

### Out-of-Distribution (OoD) Datasets

Refer to [MOS](https://arxiv.org/abs/2105.01879) and [OpenOOD](https://arxiv.org/abs/2306.09301) repositories to download the OoD datasets.

#### **Evaluation Benchmarks**
| Benchmark  | Datasets |
|------------|------------------------------------|
| **MOS** | iNaturalist, SUN, Places, Textures |
| **OpenOOD v1.5** | SSB-hard, NINCO, iNaturalist, Textures, OpenImage-O |

---

## 📌 Negative Labels (NegLabels)

We use **negative labels** derived from the **WordNet database**, located in `./NegLabel/txtfiles`.

Our approach follows the **negmining method** described in [NegLabel](https://arxiv.org/abs/2403.20078).  
Extracted 10,000 texts are stored in:

```
./NegLabel/neg_text_10000.npy
```

---

## 🚀 Quickstart

### 1️⃣ **Cross-Modal Alignment (CMA) Training Code**
An example of how the **CMA loss function** is computed:

```python
images, labels = batch
texts = in_distribution_texts[labels]

image_embeddings, text_embeddings, scale = model(images, texts)
norm_image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
norm_text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

logits_per_image = scale * norm_image_embeddings @ norm_text_embeddings.T
logits_per_text = logits_per_image.T

image_loss = loss_img(logits_per_image, ground_truth_image)
text_loss = loss_txt(logits_per_text, ground_truth_text)

CMA_text = -torch.logsumexp(logits_per_text, dim=1)
CMA_image = -torch.logsumexp(logits_per_image, dim=1)

total_loss = (image_loss + args.lam * CMA_image.mean()) / 2 + (text_loss + args.lam * CMA_text.mean()) / 2
```

### 2️⃣ **Training**
Run the following command to train the CMA model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --is-train --prompt-name <prompt> --lam 0.001 --epochs 10 --bs 512 --lr 1e-5 --dir <data dir>
```

### 3️⃣ **Inference**
Download the pretrained model checkpoint from **[Google Drive](https://drive.google.com/drive/folders/1k6trOT-zeVsT9WfbvavMPh2wwNBAFReK?usp=share_link)** and run:

```bash
python inference.py --ckpt ./ckpt/CMA-ckpt.pt
```

---

## 🥰 Acknowledgement

We adopt these codes to create this repository.
```
https://github.com/openai/CLIP
https://github.com/deeplearning-wisc/MCM
https://github.com/XueJiang16/NegLabel
https://github.com/locuslab/FLYP
```
## 📝 Citation
```
@article{kim2025enhanced,
  title={Enhanced OoD Detection through Cross-Modal Alignment of Multi-Modal Representations},
  author={Kim, Jeonghyeon and Hwang, Sangheum},
  journal={arXiv preprint arXiv:2503.18817},
  year={2025}
}
```
---

## 📢 Contact

For any issues or questions, please raise an [issue](https://github.com/ma-kjh/CMA/issues) or contact the authors.

---

