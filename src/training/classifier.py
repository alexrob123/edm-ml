"""Definining the classifier model for AFHQv2 and FFHQ datasets."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# CONFIG
FINETUNED_DINO_FNAME = "dino_finetuned.pth"
NUM_CLASSES = {
    "ffhq-64x64": 2,
    "afhqv2-64x64": 3,
    "LP-50eb47c0": 16,
}
MODEL_PATHS = {
    "ffhq-64x64": "checkpoints/finetuned_dino_ffhq-64x64.pth",
    "afhqv2-64x64": "checkpoints/finetuned_dino_afhqv2-64x64.pth",
}

# FIX: build a common structure


class Classifier(nn.Module):
    def __init__(self, url):
        super(Classifier, self).__init__()

        if "ffhq" in url:
            dataset_name = "ffhq-64x64"
        elif "afhq" in url:
            dataset_name = "afhqv2-64x64"
        else:
            url = Path(url).expanduser()
            # Suppose url structure is dataset_name/dataset.zip
            if url.name == "dataset.zip":
                dataset_path = url.parent
                dataset_name = url.parent.name
            # Suppose url structure is dataset_name/model_name/generated_images.zip
            elif url.name == "generated_images.zip":
                dataset_path = url.parent.parent
                dataset_name = url.parent.parent.name
            else:
                raise ValueError(f"Unexpected url: {url}")

        if dataset_name in MODEL_PATHS:
            self.dataset_name = dataset_name
            self.path_model = MODEL_PATHS[dataset_name]
            self.num_classes = NUM_CLASSES[dataset_name]
        else:
            self.dataset_name = dataset_path.name
            self.path_model = dataset_path / FINETUNED_DINO_FNAME
            self.num_classes = NUM_CLASSES[self.dataset_name]

        logger.info(f"Loading classifier for {self.dataset_name}...")
        logger.info(f"\t path: {self.path_model}")
        logger.info(f"\t classes: {self.num_classes}")

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base",
            num_labels=self.num_classes,
        )

        ckpt = torch.load(self.path_model, map_location="cpu")
        if "model" in ckpt.keys():
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()

    def forward(self, x):
        x = self.processor(images=x, return_tensors="pt", do_rescale=False).to(
            self.model.device
        )
        return self.model(**x)


class FeatureExtractor(nn.Module):
    def __init__(self, url):
        super(FeatureExtractor, self).__init__()

        if "ffhq" in url:
            dataset_name = "ffhq-64x64"
        elif "afhq" in url:
            dataset_name = "afhqv2-64x64"
        else:
            url = Path(url).expanduser()
            # Suppose url structure is dataset_name/dataset.zip
            if url.name == "dataset.zip":
                dataset_path = url.parent
                dataset_name = url.parent.name
            # Suppose url structure is dataset_name/model_name/generated_images.zip
            elif url.name == "generated_images.zip":
                dataset_path = url.parent.parent
                dataset_name = url.parent.parent.name
            else:
                raise ValueError(f"Unexpected url: {url}")

        if dataset_name in MODEL_PATHS:
            self.dataset_name = dataset_name
            self.path_model = MODEL_PATHS[dataset_name]
            self.num_classes = NUM_CLASSES[dataset_name]
        else:
            self.dataset_name = dataset_path.name
            self.path_model = dataset_path / FINETUNED_DINO_FNAME
            self.num_classes = NUM_CLASSES[self.dataset_name]

        logger.info(f"Loading classifier for {self.dataset_name}...")
        logger.info(f"\t path: {self.path_model}")
        logger.info(f"\t classes: {self.num_classes}")

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base",
            num_labels=self.num_classes,
            output_hidden_states=True,
        )

        ckpt = torch.load(self.path_model, map_location="cpu")
        if "model" in ckpt.keys():
            self.model.load_state_dict(ckpt["model"], strict=True)
        else:
            self.model.load_state_dict(ckpt, strict=True)

        self.model.eval()

    def forward(self, x, **kwargs):
        """
        Forward pass for feature extraction.
        Args:
            x: Input images.
            **kwargs: Additional arguments for the model.
        Returns:
            Model outputs.
        """
        x = self.processor(images=x, return_tensors="pt", do_rescale=False).to(
            self.model.device
        )
        return self.model(**x).hidden_states[-1][:, 0, :].squeeze(1)


def prepare_dino_model(num_labels):
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base",
        use_fast=True,
    )

    model = AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-base",
        num_labels=num_labels,
    )
    model = nn.DataParallel(model)  # FIX: REMOVE OR CHANGE TO DDP

    logger.info(f"Model architecture: \n{model.module}")

    # Freeze DINOv2 backbone
    for param in model.module.dinov2.parameters():
        param.requires_grad = False

    # Unfreeze last 4 layers of the transformer
    for layer in model.module.dinov2.encoder.layer[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    return processor, model
