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

from datatools.multilabel import BATCH_PROCESSING, PRED_PROCESSING
from datatools.utils import extract_dataset_name
from training import dataset
from training.classifier import prepare_dino_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Accuracy computation


def br_accuracy_computation(true, pred, method=None):
    if method is None:
        return (pred == true).float().mean().item()
    elif method == "subset":
        return (pred == true).all(dim=1).float().mean().item()
    elif method == "per_class":
        return (((pred == true).float()).sum(dim=0) / true.shape[0]).tolist()
    elif method == "hamming_loss":
        return (pred != true).float().mean().item()
    else:
        raise ValueError(f"Unknown method: {method}")


def lp_accuracy_computation(true, pred):
    pass


ACC_COMPUTATION = {
    "br": br_accuracy_computation,
    "lp": lp_accuracy_computation,
}


####################################################################################################
####################################################################################################
####################################################################################################

# fmt: off

@click.command()

@click.option("--name",                 help="Experiment name",                            type=str, default="dino")
@click.option("--data", "-d",           help="Path to the dataset", metavar="DIR|ZIP",     type=click.Path(exists=True), required=True)
@click.option("--num-labels", "-nl",    help="Num labels in dataset (not labelsets)",      type=int, required=True,)
@click.option("--method", "-mll",       help="MLL method",                                 type=click.Choice(["br", "lp"]), required=True)
@click.option("--batch", "batch_size",  help="Batch size",                                 type=int, default=256, show_default=True)
@click.option("--epochs", "num_epochs", help="Number of epochs for training",              type=int, default=10, show_default=True)
@click.option("--seed",                 help="Seed for randomness",                        type=int, default=0, show_default=True)
@click.option( "--evaluate",            help="Run 1 evaluation epoch instead of training", is_flag=True)

# fmt: on

def main(name, data, num_labels, method, batch_size, num_epochs, seed, evaluate):
    logger.info(f"{'EVALUATION' if evaluate else 'TRAINING'}")

    # Config

    data_path = Path(data).expanduser()
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
    logger.info(f"Checkpoint in {ckpt_path}")
    logger.info(f"Evaluation in {eval_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data

    full_dataset = dataset.ImageFolderDataset(
        path=data_path,
        resolution=64,
        use_labels=True,
        max_size=None,
        xflip=False,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(full_dataset), generator=g)
    split = int(0.9 * len(full_dataset))
    train_indices, val_indices = indices[:split], indices[split:]

    logger.info(f"# training samples {len(train_indices)}")
    logger.info(f"# validation samples {len(val_indices)}")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model

    processor, model = prepare_dino_model(
        {
            "br": num_labels,
            "lp": 2**num_labels,
        }[method]
    )
    model.to(device)

    # Setup

    criterion = {  # reduction="mean" by default,
        "br": nn.BCEWithLogitsLoss(),
        "lp": nn.CrossEntropyLoss(label_smoothing=0.1),
    }[method]

    optimizer = torch.optim.AdamW(
        [
            {"params": model.module.classifier.parameters(), "lr": 1e-3},
            {"params": model.module.dinov2.encoder.layer[-4:].parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Chekpoint

    monitor = {
        "epoch": 0,
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
    }

    if evaluate and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        
        model.module.load_state_dict(ckpt["model"])
        monitor = ckpt["monitor"]

        logger.info(f"Evaluating checkpoint — epoch {monitor['epoch']} ")

    elif evaluate:
        raise ValueError(f"Can't evaluate ckpt {ckpt_path} because it does not exist")

    elif ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)

        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        monitor.update(ckpt.get("monitor", {}))

        logger.info(f"Training from restored checkpoint — epoch {monitor['epoch']}")

    else:
        logger.info("Training from scratch")

    # Loop

    start_epoch = monitor["epoch"]
    best_eval = 1e100 if not monitor["val_loss"] else min(monitor["val_loss"])

    epoch_pbar = tqdm(range(start_epoch + 1, num_epochs + 1), desc="Training")

    for epoch in epoch_pbar:
        monitor["epoch"] = epoch

        # Train epoch

        if evaluate:
            monitor["train_loss"].append(0)

        else:
            train_batch_pbar = tqdm(
                train_loader,
                leave=False,
                desc=f"Epoch {epoch}/{num_epochs}",
            )

            train_loss = 0
            train_count = 0

            for x, y in train_batch_pbar:
                x, y = BATCH_PROCESSING[method](x, y)
                x, y = x.to(device), y.to(device)
                train_count += x.size(0)

                inputs = processor(images=x, return_tensors="pt", do_rescale=False)
                inputs = inputs.to(device)
                outputs = model(**inputs)

                loss = criterion(outputs.logits, y)
                train_loss += loss.item() * x.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_batch_pbar.set_postfix({"Loss": loss.item()})

            scheduler.step()

            monitor["train_loss"].append(train_loss / train_count)

        # Val epoch

        val_batch_pbar = tqdm(val_loader, leave=False, desc="Validation")

        val_loss = 0
        val_count = 0
        val_correct = 0

        for x, y in val_batch_pbar:
            x, y = BATCH_PROCESSING[method](x, y)
            x, y = x.to(device), y.to(device)
            val_count += x.size(0)

            with torch.no_grad():
                inputs = processor(images=x, return_tensors="pt", do_rescale=False)
                inputs = inputs.to(device)
                outputs = model(**inputs)

                loss = criterion(outputs.logits, y)
                val_loss += loss.item() * x.size(0)

                preds = PRED_PROCESSING[method](outputs)
                val_correct += (preds == y).float().sum(dim=0)

                val_batch_pbar.set_postfix({"Loss": loss.item()})

        accuracy = (
            (val_correct.mean(dim=0) / val_count).item()
            if val_correct.size()
            else (val_correct / val_count).item()
        )

        monitor["val_loss"].append(val_loss / val_count)
        monitor["accuracy"].append(accuracy)

        # End epoch

        epoch_pbar.set_postfix(
            {
                "train_loss": monitor["train_loss"][-1],
                "val_loss": monitor["val_loss"][-1],
                "accuracy": accuracy * 100,
            }
        )

        if evaluate:
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


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
