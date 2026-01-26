"""Definining the classifier model for AFHQv2 and FFHQ datasets."""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

# CONFIG
NUM_CLASSES = {"ffhq-64x64": 2, "afhqv2-64x64": 3}
PATHS = {
    "ffhq-64x64": "checkpoints/finetuned_dino_ffhq-64x64.pth",
    "afhqv2-64x64": "checkpoints/finetuned_dino_afhqv2-64x64.pth",
    "celeba-64x64": "ckpts/final/50eb47c0-edm_64x64/dino_finetuning_lp.pth",
}


class Classifier(nn.Module):
    def __init__(self, url):
        super(Classifier, self).__init__()
        if "ffhq" in url:
            dataset_name = "ffhq-64x64"
        elif "afhq" in url:
            dataset_name = "afhqv2-64x64"
        elif "celeba" in url:
            dataset_name = "celeba-64x64"
        print(
            f"Loading classifier for {dataset_name}..."
            f" Classes: {NUM_CLASSES[dataset_name]}, Path: {PATHS[dataset_name]}"
        )
        self.dataset_name = dataset_name
        self.num_classes = NUM_CLASSES[dataset_name]
        self.path_model = PATHS[dataset_name]
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base", num_labels=self.num_classes
        )
        self.model.load_state_dict(torch.load(self.path_model, map_location="cpu"))
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
        print(
            f"Loading classifier for {dataset_name}..."
            f" Classes: {NUM_CLASSES[dataset_name]}, Path: {PATHS[dataset_name]}"
        )
        self.dataset_name = dataset_name
        self.num_classes = NUM_CLASSES[dataset_name]
        self.path_model = PATHS[dataset_name]
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base",
            num_labels=self.num_classes,
            output_hidden_states=True,
        )
        self.model.load_state_dict(
            torch.load(self.path_model, map_location="cpu"), strict=True
        )
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
