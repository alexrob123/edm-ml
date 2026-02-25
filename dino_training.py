"""Finetuning final layer of pretrained model and training classification head."""

# Adding method
# -------------
# Change in the number of classes considered (2^nl for lp, nl for br
# Change in the way the predictions is derived from loggits
# FIX: change accuracy computation for br method

import logging
from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from datatools.utils import extract_dataset_name
from training import dataset
from training.classifier import prepare_dino_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


def br_batch_processing(x, y):
    x = x.to(torch.float32) / 255.0
    return x, y


def lp_batch_processing(x, y):
    x = x.to(torch.float32) / 255.0
    y = torch.argmax(y, dim=1)  # convert from one-hot to class index
    return x, y


def br_pred_processing(outputs):
    preds = torch.sigmoid(outputs.logits) > 0.5
    return preds


def lp_pred_processing(outputs):
    _, preds = torch.max(outputs.logits, 1)
    return preds


def br_accuracy_computation(true, pred):
    correct_per_class = ((pred == true).float()).sum(dim=0)  # [C]
    total_per_class = true.shape[0]
    accuracy_per_class = correct_per_class / total_per_class


def lp_accuracy_computation(true, pred):
    pass


####################################################################################################
####################################################################################################
####################################################################################################


@click.command()
@click.option(
    "--name",
    "name",
    type=str,
    default="dino",
    help="Experiment name used in checkpoint path.",
)
@click.option(
    "--method",
    "method",
    type=click.Choice(["br", "lp"]),
    required=True,
    help="Multi-label learning method.",
)
@click.option(
    "--data-path",
    "data_path",
    type=click.Path(exists=True),
    required=True,
    metavar="DIR|ZIP",
    help="Directory containing dataset files or zip file.",
)
@click.option(
    "--num-labels",
    "num_labels",
    type=int,
    required=True,
    help="Number of labels in dataset (not labelsets).",
)
@click.option(
    "--batch",
    "batch_size",
    type=int,
    default=256,
    show_default=True,
    help="Batch size for train and validation dataloaders.",
)
@click.option(
    "--epochs",
    "num_epochs",
    type=int,
    default=10,
    show_default=True,
    help="Number of epochs for training.",
)
@click.option(
    "--seed",
    "seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed for randomness.",
)
@click.option(
    "--evaluate",
    "evaluate",
    is_flag=True,
    help="Run a single evaluation epoch.",
)
def main(
    name,
    method,
    data_path,
    num_labels,
    batch_size,
    num_epochs,
    seed,
    evaluate,
):
    logger.info(f"{'EVALUATION' if evaluate else 'TRAINING'}")

    ### CONFIG ###
    data_path = Path(data_path).expanduser()
    dataset_name = extract_dataset_name(data_path)

    if name is None:
        ckpt_dir = Path.cwd() / "checkpoints" / dataset_name
        eval_dir = Path.cwd() / "outputs" / dataset_name
    else:
        ckpt_dir = Path.cwd() / "checkpoints" / name / dataset_name
        eval_dir = Path.cwd() / "outputs" / name / dataset_name

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = "dino_finetuning.pth"
    eval_file = "dino_finetuned.pth"

    ckpt_path = ckpt_dir / ckpt_file
    eval_path = eval_dir / eval_file

    logger.info(f"Data in {data_path}")
    logger.info(f"Checkpoints in {ckpt_path}")
    logger.info(f"Evaluation results in {eval_path}")

    EVALUATE = evaluate
    METHOD = method

    NUM_LABELS_FOR_MODEL = {
        "br": num_labels,
        "lp": 2**num_labels,
    }
    CRITERION = {  # reduction="mean" by default,
        "br": nn.BCEWithLogitsLoss(),
        "lp": nn.CrossEntropyLoss(label_smoothing=0.1),
    }
    BATCH_PROCESSING = {
        "br": br_batch_processing,
        "lp": lp_batch_processing,
    }
    PRED_PROCESSING = {
        "br": br_pred_processing,
        "lp": lp_pred_processing,
    }
    ACC_COMPUTATION = {
        "br": None,
        "lp": None,
    }

    SEED = seed
    BATCH_SIZE = batch_size
    NUM_EPOCHS = num_epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### DATA ###

    full_dataset = dataset.ImageFolderDataset(
        path=data_path,
        resolution=64,  # FIX: PUT TO 224 FOR DINOv2 ?
        use_labels=True,
        max_size=None,
        xflip=False,
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    indices = torch.randperm(len(full_dataset), generator=g)
    split = int(0.9 * len(full_dataset))
    train_indices, val_indices = indices[:split], indices[split:]

    logger.info(f"# training samples {len(train_indices)}")
    logger.info(f"# validation samples {len(val_indices)}")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    ### MODEL ###

    processor, model = prepare_dino_model(NUM_LABELS_FOR_MODEL[METHOD])
    model.to(DEVICE)

    ### SETUP ###

    criterion = CRITERION[METHOD]

    optimizer = torch.optim.AdamW(
        [
            {"params": model.module.classifier.parameters(), "lr": 1e-3},
            {"params": model.module.dinov2.encoder.layer[-4:].parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ### CKPT ###

    monitor = {
        "epoch": 0,
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
    }

    if EVALUATE and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        model.module.load_state_dict(ckpt["model"])

        logger.info(f"Evaluating checkpoint — epoch {monitor['epoch']} ")

    elif EVALUATE:
        raise ValueError(f"Can't evaluate ckpt {ckpt_path} because it does not exist")

    elif ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        monitor.update(ckpt.get("monitor", {}))

        logger.info(f"Training from restored checkpoint — epoch {monitor['epoch']}")

    else:
        logger.info("Training from scratch")

    ### LOOP ###

    start_epoch = monitor["epoch"]
    best_eval = 1e100 if not monitor["val_loss"] else min(monitor["val_loss"])

    epoch_pbar = tqdm(range(start_epoch + 1, NUM_EPOCHS + 1), desc="Training")

    for epoch in epoch_pbar:
        monitor["epoch"] = epoch

        ### TRAIN EPOCH ###

        if EVALUATE:
            monitor["train_loss"].append(0)

        else:
            train_batch_pbar = tqdm(
                train_loader,
                leave=False,
                desc=f"Epoch {epoch}/{NUM_EPOCHS}",
            )

            train_loss = 0
            train_count = 0

            for x, y in train_batch_pbar:
                x, y = BATCH_PROCESSING[METHOD](x, y)
                x, y = x.to(DEVICE), y.to(DEVICE)
                train_count += x.size(0)

                inputs = processor(images=x, return_tensors="pt", do_rescale=False)
                inputs.to(DEVICE)
                outputs = model(**inputs)

                loss = criterion(outputs.logits, y)
                train_loss += loss.item() * x.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_batch_pbar.set_postfix({"Loss": loss.item()})

            scheduler.step()

            monitor["train_loss"].append(train_loss / train_count)

        ### VAL EPOCH ###

        val_batch_pbar = tqdm(val_loader, leave=False, desc="Validation")

        val_loss = 0
        val_count = 0
        val_correct = 0

        for x, y in val_batch_pbar:
            x, y = BATCH_PROCESSING[METHOD](x, y)
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_count += x.size(0)

            with torch.no_grad():
                inputs = processor(images=x, return_tensors="pt", do_rescale=False)
                inputs = inputs.to(DEVICE)
                outputs = model(**inputs)

                loss = criterion(outputs.logits, y)
                val_loss += loss.item() * x.size(0)

                preds = PRED_PROCESSING[METHOD](outputs)
                val_correct += (preds == y).sum().item()

                val_batch_pbar.set_postfix({"Loss": loss.item()})

        accuracy = val_correct / val_count

        monitor["val_loss"].append(val_loss / val_count)
        monitor["accuracy"].append(accuracy)

        # END EPOCH

        epoch_pbar.set_postfix(
            {
                "train_loss": monitor["train_loss"][-1],
                "val_loss": monitor["val_loss"][-1],
                "accuracy": accuracy * 100,
            }
        )

        if EVALUATE:
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "monitor": monitor,
                },
                eval_path,
            )
            # Do not run further epochs.
            break

        elif monitor["val_loss"][-1] < best_eval:
            best_eval = monitor["val_loss"][-1]
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "monitor": monitor,
                },
                ckpt_path,
            )
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "monitor": monitor,
                },
                data_path.parent / "dino_finetuned.pth",
            )

    logger.info("THE END")


if __name__ == "__main__":
    main()

    # data_dir: ~/data/CelebA/AlignedCropped/_edm64
    # dataset: 50eb47c0

    # nohup uv run dino.py --data-dir ~/data/CelebA/edm-64x64/ --dataset 50eb47c0.zip --num_classes 16 > dino.log 2>&1 &
    # nohup uv run dino.py -dd ~/data/CelebA/edm-64x64/ -d 50eb47c0.zip -nc 16 -bs 128 -ne 10 > dino.log 2>&1 &
