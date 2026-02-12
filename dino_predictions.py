import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from edm_ml.metrics import compute_multiclass_metrics, df_metrics_per_class
from training import dataset

######################################################################


def main(args):
    BATCH_SIZE = 128

    ######################################################################
    ######################################################################
    ######################################################################

    data_dir = Path(args.data_dir).expanduser()
    data_file = args.dataset
    DATA_PATH = data_dir / data_file

    GEN_PATH = data_dir / args.generation

    CKPT_PATH = Path.cwd() / args.ckpt_path

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(DATA_PATH)
    print(GEN_PATH)
    print(CKPT_PATH)

    class_dirs = [d for d in GEN_PATH.iterdir() if d.is_dir()]
    print(len(class_dirs))

    NUM_LABELS = len(class_dirs)

    for class_dir in class_dirs:
        print(class_dir)

    ######################################################################
    ######################################################################
    # REFERENCE DATASET
    ######################################################################
    ######################################################################
    print("REFERENCE DATASET")

    origin_data_file = data_file.split("-")[0] + ".zip"
    origin_data_file

    path = data_dir / origin_data_file

    with ZipFile(path) as z:
        with z.open("dataset.json", "r") as j:
            data = json.load(j)

    labels = data["labels"]
    labelsets = data["labelsets"]
    labelspace = data["labelspace"]
    print("labelspace", labelspace)

    counts = Counter(dict(labels).values())
    counts = dict(sorted(counts.items()))
    counts

    id2labelset = {i: ls for i, ls in enumerate(labelsets)}

    df = pd.DataFrame([id2labelset, counts]).T
    df.columns = ["Labelset", "Count"]
    df

    total_count = df["Count"].sum()
    df["Proportion"] = df["Count"] / total_count * 100

    df_ = df
    styler_ = df_.style.format(decimal=".", thousands=",", precision=2).hide(
        axis="index"
    )
    print(styler_.to_latex())

    ######################################################################
    ######################################################################
    # GEN DATA
    ######################################################################
    ######################################################################

    # Prepare list to store results
    data_gen = []

    for class_dir in tqdm(class_dirs, desc="Iterating over class dirs"):
        name = class_dir.name
        class_id = name.split("class")[-1]

        # Count files in the directory (ignore subdirectories)
        file_count = sum(1 for f in class_dir.iterdir() if f.is_file())

        data_gen.append(
            {"class_name": name, "class_id": class_id, "file_count": file_count}
        )

    # Convert to DataFrame
    df = pd.DataFrame(data_gen)

    total_count = df["file_count"].sum()
    df["Proportion"] = df["file_count"] / total_count * 100

    df_ = df
    styler_ = df_.style.format(decimal=".", thousands=",", precision=2).hide(
        axis="index"
    )
    print(styler_.to_latex())

    ######################################################################
    ######################################################################
    # MODEL AND PREDICTIONS
    ######################################################################
    ######################################################################
    print("MODEL")

    def prepare_model(num_labels):
        processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base",
            use_fast=True,
        )

        model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base",
            num_labels=num_labels,
        )
        model = nn.DataParallel(model)  # FIX: REMOVE OR CHANGE TO DDP

        # Freeze DINOv2 backbone
        for param in model.module.dinov2.parameters():
            param.requires_grad = False

        # Unfreeze last 4 layers of the transformer
        for layer in model.module.dinov2.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

        return processor, model

    processor, model = prepare_model(NUM_LABELS)

    assert CKPT_PATH.exists()
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    model.module.load_state_dict(ckpt["model"])
    model.to(DEVICE)

    ######################################################################
    print("PREDICTION")

    def lp_batch_processing(x, y=None):
        x = x.to(torch.float32) / 255.0
        if y is None:
            return x
        else:
            y = torch.argmax(y, dim=1)  # convert from one-hot to class index
            return x, y

    def lp_pred_processing(outputs):
        _, preds = torch.max(outputs.logits, 1)
        return preds

    true_labels = []
    pred_labels = []

    for class_dir in tqdm(class_dirs, desc="Iterating over class dirs"):
        name = class_dir.name
        id = name.split("class")[-1]

        class_dataset = dataset.ImageFolderDataset(
            path=class_dir,
            resolution=64,
            use_labels=False,
            max_size=None,
            xflip=False,
        )

        class_loader = DataLoader(class_dataset, batch_size=BATCH_SIZE, shuffle=False)

        pred_batch_pbar = tqdm(class_loader, leave=False, desc="Prediction")
        for x, _ in pred_batch_pbar:
            x = lp_batch_processing(x)
            x = x.to(DEVICE)
            n_samples = x.size(0)
            true_labels.append([id] * n_samples)

            with torch.no_grad():
                inputs = processor(images=x, return_tensors="pt", do_rescale=False)
                inputs = inputs.to(DEVICE)
                outputs = model(**inputs)

                preds = lp_pred_processing(outputs)
                pred_labels.append(preds)

    true_labels_flat = [int(x) for sub in true_labels for x in sub]
    pred_labels_flat = [int(x) for sub in pred_labels for x in sub]

    all_true_labels = torch.tensor(true_labels_flat)
    all_pred_labels = torch.tensor(pred_labels_flat)

    print("Number of labels", len(all_true_labels), len(all_pred_labels))

    ######################################################################
    print("LABELSETS METRICS")

    metrics = compute_multiclass_metrics(
        all_true_labels.detach().cpu().numpy(),
        all_pred_labels.detach().cpu().numpy(),
        NUM_LABELS,
    )

    print(metrics["accuracy"])

    df = df_metrics_per_class(metrics)
    df["Attrs"] = [v for v in id2labelset.values()]

    print("\n")
    df_ = df
    styler_ = df_.style.format(decimal=".", thousands=",", precision=2)
    lat = styler_.to_latex()
    print(lat)

    ######################################################################
    print("LABELS METRICS")

    def lp_to_bl(classes, label_space, class2labelset):
        """Convert Label Powerset classes to Binary Labelset"""

        # Prepare vector of 0
        Y = np.zeros((len(classes), len(label_space)))

        # Map label to idx in label_space
        label2idx = {lab: i for i, lab in enumerate(label_space)}

        # Fill with ones in the right spots
        for i, c in enumerate(classes):
            indices = [label2idx[label] for label in class2labelset[c]]
            Y[i, indices] = 1

        return Y

    y_true_lp = all_true_labels.detach().cpu().numpy()
    y_pred_lp = all_pred_labels.detach().cpu().numpy()

    Y_true = lp_to_bl(y_true_lp, labelspace, id2labelset)
    Y_pred = lp_to_bl(y_pred_lp, labelspace, id2labelset)

    precision, recall, f1, support = precision_recall_fscore_support(
        Y_true,
        Y_pred,
        average=None,
    )

    df_metrics_per_label = pd.DataFrame(
        {
            "Label": labelspace,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
        }
    )

    print(df_metrics_per_label)
    print(df_metrics_per_label["Support"].sum())


if __name__ == "__main__":
    args = SimpleNamespace(
        data_dir="~/data/CelebA/ml-lp",
        dataset="50eb47c0-edm_64x64.zip",
        generation="50eb47c0-gen_model3",
        ckpt_path="ckpts/final/50eb47c0-edm_64x64/dino_finetuning_lp.pth",
    )
    main(args)
