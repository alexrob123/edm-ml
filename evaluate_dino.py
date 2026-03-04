import json
import logging
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datatools.multilabel import BATCH_PROCESSING, PRED_PROCESSING
from datatools.utils import make_json_serializable
from evaluation.supervised import compute_multiclass_metrics, compute_multihot_metrics
from training import dataset
from training.classifier import prepare_dino_model

logging.basicConfig(
    format="[%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


# Handle for error in label alignement between generated and referecence images
# THIS IS A TEMPORARY FIX, SHOULD BE REMOVED ONCE THE ISSUE IS SOLVED
# dino model has been trained with good labels: dino-labels, hence predicts dino-labels
# gen model has been trained with bad ref labels: gen-labels, hence gen images are labeled with gen-labels
# therefore when evaluating dino on gen data,
# -> pred labels are in dino-labels
# -> "true" labels are in gen-labels
# => we map gen-labels (true labels) to dino-labels (pred labels)
# ----------------------------------------------------------------------------------------------------

APPLY_LABEL_MAPPING = False

LABEL_MAPPING = {
    0: 7,
    1: 1,
    2: 13,
    3: 14,
    4: 10,
    5: 4,
    6: 3,
    7: 8,
    8: 15,
    9: 2,
    10: 6,
    11: 5,
    12: 9,
    13: 12,
    14: 0,
    15: 11,
}


def map_labels(x: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
    if not mapping:
        raise ValueError("mapping is empty")

    x_long = x.long()
    max_key = max(mapping)

    lut = torch.full((max_key + 1,), -1, dtype=torch.long, device=x.device)
    keys = torch.tensor(list(mapping.keys()), dtype=torch.long, device=x.device)
    vals = torch.tensor(list(mapping.values()), dtype=torch.long, device=x.device)
    lut[keys] = vals

    if x_long.min() < 0 or x_long.max() > max_key:
        raise KeyError("x contains values outside mapping key range")

    y = lut[x_long]
    if (y < 0).any():
        raise KeyError("x contains keys missing from mapping")

    return y


# ----------------------------------------------------------------------------------------------------

DINO_METRICS_FNAME = "dino_metrics.jsonl"

####################################################################################################
####################################################################################################
####################################################################################################

# fmt: off

@click.command()

@click.option("--data", "-d",          help="Path to the dataset", metavar="DIR|ZIP",          type=click.Path(exists=True), required=True)
@click.option("--num-labels", "-nl",   help="Num labels in dataset (not labelsets)",           type=int, required=True,)
@click.option("--method", "-mll",      help="MLL method",                                      type=click.Choice(["br", "lp"]), required=True)
@click.option("--model", "-m",         help="Path to the model checkpoint.", metavar="PTH",    type=click.Path(exists=True), required=True)
@click.option("--batch", "batch_size", help="Batch size",                                      type=int, default=256, show_default=True)
@click.option("--out", "-o",           help="Path for saving evaluation", metavar="DIR|JSONL", type=click.Path(exists=True), required=False)

# fmt: on

def main(data, num_labels, method, model, batch_size, out):
    """Evaluate Fine-tuned DINOv2 model on generated images and compute metrics"""

    # Config

    data_path = Path(data).expanduser()
    model_path = Path(model).expanduser()
    assert data_path.exists(), f"Data path {data_path} does not exist."
    assert model_path.exists(), f"Model path {model_path} does not exist."

    if out is None: 
        output_path = data_path.parent / DINO_METRICS_FNAME
    else :
        output_path = Path(out).expanduser()
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / DINO_METRICS_FNAME
    elif output_path.suffix and output_path.suffix == ".jsonl":
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Invalid output path: {out}")
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output path: {output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_labels = {
        "br": num_labels,
        "lp": 2**num_labels,
    }[method]

    # Load fine-tuned DINOv2 model.

    logger.info("Model")

    processor, model = prepare_dino_model(num_labels)

    ckpt = torch.load(
        model_path,
        map_location=device,
    )
    if "model" in ckpt.keys():
        model.module.load_state_dict(ckpt["model"])
    else:
        model.module.load_state_dict(ckpt)

    # if "monitor" in ckpt.keys():
    #     monitor = ckpt["monitor"]
    #     logger.info(f"Loaded training monitor: {monitor.keys()}")

    #     monitor_df = pd.DataFrame(
    #         {
    #             "train_loss": monitor["train_loss"],
    #             "val_loss": monitor["val_loss"],
    #             "accuracy": monitor["accuracy"],
    #         }
    #     )
    #     monitor_df = format_colnames(monitor_df)
    #     print(monitor_df)

    model.to(device)

    # Predict labels for generated images

    logger.info("Prediction")

    gen_dataset = dataset.ImageFolderDataset(
        path=data_path,
        resolution=64,
        use_labels=True,
        max_size=None,
        xflip=False,
    )
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    true_labels = []
    pred_labels = []

    pred_batch_pbar = tqdm(gen_loader, leave=False, desc="Prediction")
    for x, y in pred_batch_pbar:
        x, y = BATCH_PROCESSING[method](x, y=y)
        true_labels.append(y)

        with torch.no_grad():
            inputs = processor(images=x, return_tensors="pt", do_rescale=False)
            inputs = inputs.to(device)
            outputs = model(**inputs)

            preds = PRED_PROCESSING[method](outputs)
            pred_labels.append(preds)

    true_label_tensor = torch.concatenate(true_labels)
    pred_label_tensor = torch.concatenate(pred_labels)

    if APPLY_LABEL_MAPPING:
        logger.info("Applying label mapping to align predicted labels with true labels")
        true_label_tensor = map_labels(true_label_tensor, LABEL_MAPPING)

    # Predict labels for generated images.
    # ----------------------------------------------------------------------------------------------------
    logger.info("Metrics")

    metrics = {"br": compute_multihot_metrics, "lp": compute_multiclass_metrics}[method](
        true_label_tensor.detach().cpu().numpy(),
        pred_label_tensor.detach().cpu().numpy(),
        num_classes=num_labels,
    )

    logger.info(f"Accuracy: {metrics['accuracy']}")
    
    

    with open(output_path, "w") as file:
        json.dump(make_json_serializable(metrics), file, indent=4)

    # metrics_df = pd.DataFrame([metrics])
    # metrics_df["Attrs"] = [v for v in id2labelset.values()]
    # metrics_df = df_metrics_per_class(metrics)

    # metrics_df = format_mean_std(metrics_df, decimals=3)
    # metrics_df = format_colnames(metrics_df)

    # save_latex_table(
    #     metrics_df,
    #     out_dir=output_path,
    #     fname=f"dino_labelsets_metrics_{gen_ds}",
    # )

    # ######################################################################
    # logger.info("Labels metrics")

    # y_true_lp = true_label_tensor.detach().cpu().numpy()
    # y_pred_lp = pred_label_tensor.detach().cpu().numpy()

    # Y_true = lp_to_bl(y_true_lp, labelspace, id2labelset)
    # Y_pred = lp_to_bl(y_pred_lp, labelspace, id2labelset)

    # precision, recall, f1, support = precision_recall_fscore_support(
    #     Y_true,
    #     Y_pred,
    #     average=None,
    # )

    # df_metrics_per_label = pd.DataFrame(
    #     {
    #         "Label": labelspace,
    #         "Precision": precision,
    #         "Recall": recall,
    #         "F1": f1,
    #         "Support": support,
    #     }
    # )

    # df_metrics_per_label = format_colnames(df_metrics_per_label)

    # save_latex_table(
    #     df_metrics_per_label,
    #     out_dir=output_path,
    #     fname=f"dino_labels_metrics_{gen_ds}",
    # )


if __name__ == "__main__":
    main()
