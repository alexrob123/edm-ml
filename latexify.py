"""This file aims to read results and convert them to LaTeX tables."""

import json
import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd

from datatools.multilabel import read_br_dataset_meta, read_lp_dataset_meta
from datatools.utils import extract_dataset_name, zip_labels

logging.basicConfig(
    format="[%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


####################################################################################################
# Compare Unconditional and Conditional Generation
####################################################################################################


conditional_fnames = [
    f"~/data/CelebA/edited/LP-50eb47c0-model3-supp/{i}/evaluation.jsonl"
    for i in range(16)
]


def compare_generations(fname, conditional_fnames):

    evaluations = []
    for fname in [fname] + conditional_fnames:
        with open(fname, "r") as f:
            data = json.load(f)
            evaluations.append(data)


def build_label_df(labels, **kwargs):
    counts = dict(sorted(Counter(labels).items()))

    df = pd.DataFrame(counts.items(), columns=["Label", "Count"])
    df = df.set_index("Label")

    df["Proportion"] = df["Count"] / df["Count"].sum() * 100

    for col_name, mapping in kwargs.items():
        df[col_name] = pd.Series(mapping)

    return df


def validate_same_index(dataframes):
    ref_index = dataframes[0].index

    for df in dataframes[1:]:
        if not ref_index.equals(df.index):
            raise ValueError("DataFrame index mismatch between ")


def merge_dataframes_with_duplicate_handling(dataframes, names=None):
    if names is None:
        names = [str(i + 1) for i in range(len(dataframes))]

    validate_same_index(dataframes)

    merged = pd.DataFrame(index=dataframes[0].index)
    seen_cols = []
    for df in dataframes:
        seen_cols.extend(df.columns.tolist())
    unique_cols = list(dict.fromkeys(seen_cols))

    for col in unique_cols:
        col_series = [
            (source_name, df[col])
            for source_name, df in zip(names, dataframes)
            if col in df.columns
        ]

        if len(col_series) == 1:
            _, s = col_series[0]
            merged[col] = s
            continue

        _, ref_series = col_series[0]
        if all(ref_series.equals(s) for _, s in col_series[1:]):
            merged[col] = ref_series
        else:
            for i, (source_name, s) in enumerate(col_series, start=1):
                prefix = source_name if source_name else str(i)
                merged[f"{prefix[:1]}._{col}"] = s

    return merged


def metadata_versus_df(data_path, gen_path, method):

    data_path = Path(data_path).expanduser()
    gen_path = Path(gen_path).expanduser()

    assert data_path.exists(), f"Data path {data_path} does not exist."
    assert gen_path.exists(), f"Generated path {gen_path} does not exist."

    logger.info(f"Data path: {data_path}")
    logger.info(f"Generated path: {gen_path}")

    data_name = extract_dataset_name(data_path)
    gen_name = extract_dataset_name(gen_path)

    if method == "lp":
        logger.info("Reading Label Powerset dataset meta.")

        data_meta = read_lp_dataset_meta(data_path)
        data_labels_df = build_label_df(
            data_meta["labels"],
            Labelset={i: ls for i, ls in enumerate(data_meta["labelsets"])},
        )
    else:
        raise NotImplementedError(f"ML method {method} not supported.")

    gen_labels = zip_labels(gen_path)
    gen_labels_df = build_label_df(gen_labels)

    metadata_df = [data_labels_df, gen_labels_df]
    metadata_source = ["original", "generated"]

    merged_df = merge_dataframes_with_duplicate_handling(metadata_df, metadata_source)
    merged_df.index.name = "Label"

    return merged_df, f"meta_{data_name}_{gen_name}"


####################################################################################################
# Formatting
####################################################################################################


def format_mean_std(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """
    Combine columns of the form `metric` and `metric_std`
    into a single formatted column: "mean ± std".

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing mean and *_std columns.
    decimals : int
        Number of decimals for formatting.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe with *_std columns removed.
    """
    df_fmt = df.copy()

    # Find all std columns
    std_cols = [col for col in df.columns if col.endswith("_std")]

    for std_col in std_cols:
        base_col = std_col[:-4]  # remove "_std"

        if base_col in df.columns:
            df_fmt[base_col] = (
                df[base_col].round(decimals).astype(str)
                + r" $\pm$ "
                + df[std_col].round(decimals).astype(str)
            )

    # Drop std columns
    df_fmt = df_fmt.drop(columns=std_cols, errors="ignore")

    return df_fmt


def format_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with original column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with formatted column names.
    """
    df = df.copy()
    df = df.rename(columns=lambda c: c.replace("_", " ").title())
    df = df.rename(columns={"Fid": "FID"})
    df = df.rename(
        columns={
            "Accuracy Per Class": "Accuracy",
            "Precision Per Class": "Precision",
            "Recall Per Class": "Recall",
        }
    )
    return df


####################################################################################################
# Saving LaTeX tables
####################################################################################################


def save_latex_table(df: pd.DataFrame, out_dir: str | Path, fname: str, **kwargs):
    """
    Save a DataFrame as a LaTeX table.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    out_dir : str | Path
        Directory where the LaTeX file will be saved.
    fname : str
        Name of the LaTeX file (without extension).
    """
    logger.info("Generating LaTeX table for DataFrame:")
    print(df)

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"{fname}.tex"

    styler = df.style.format(decimal=".", thousands=",", precision=3, **kwargs)
    latex = styler.to_latex(hrules=True)

    out.write_text(latex, encoding="utf-8")
    logger.info(f"Saved LaTeX table to {out.resolve()}")


####################################################################################################
####################################################################################################
####################################################################################################


@click.group()
def main():
    """
    Write results from a file given in input to a LaTeX table.
    
    Examples:

    \b
    # Generate 50000 images and save them as fid-tmp/*/*.png
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Compute dataset Inception reference statistics
    python evaluation.py inception-ref --data-path datasets/my-dataset.zip   

    \b
    # Calculate FID
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz   
    """


# @click.command()
# @click.option(
#     "-d",
#     "--data",
#     "--data-path",
#     "data_path",
#     type=click.Path(exists=True),
#     metavar="DIR|ZIP",
#     multiple=True,
#     help="Path to the original dataset.",
# )
# @click.option(
#     "--num-labels",
#     "num_labels",
#     type=int,
#     default=4,
#     help="Number of labels in the dataset.",
# )
# @click.option(
#     "--ml-method",
#     type=click.Choice(["br", "lp"]),
#     required=False,
#     help="Method used for fine-tuning the DINOv2 model (Binary Relevance or Label Powerset).",
# )
# @click.option(
#     "--output-path",
#     type=click.Path(exists=True),
#     default="./outputs/latex/",
#     help="Path to save the generated LaTeX tables.",
# )

####################################################################################################
# meta-vs
####################################################################################################


@main.command()
@click.option(
    "-d",
    "--data",
    "--data-path",
    "data_path",
    type=click.Path(exists=True),
    metavar="DIR|ZIP",
    help="Path to the original dataset (directory or zip file).",
)
@click.option(
    "-g",
    "--gen",
    "--gen-path",
    "gen_path",
    type=click.Path(exists=True),
    metavar="DIR|ZIP",
    help="Path to the generated dataset (directory or zip file).",
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["br", "lp"]),
    required=False,
    help="Method used for fine-tuning the DINOv2 model (Binary Relevance or Label Powerset).",
)
@click.option(
    "-o",
    "--out",
    "--output-path",
    "output_path",
    type=click.Path(exists=True),
    default="./outputs/latex/",
    help="Path to save the generated LaTeX tables.",
)
def meta_vs(data_path, gen_path, method, output_path):
    """
    Compare the original dataset with the generated one in terms of label distribution.

    Args:
        data_path (str | Path): Path to the original dataset (directory or zip file).
        gen_path (str | Path): Path to the generated dataset (directory or zip file).
        method (str): Method used for fine-tuning the DINOv2 model (Binary Relevance or Label Powerset).
        output_path (str | Path): Path to save the generated LaTeX tables.
    """

    df, name = metadata_versus_df(data_path, gen_path, method)
    df = format_mean_std(df, decimals=3)
    df = format_colnames(df)

    save_latex_table(
        df,
        out_dir=output_path,
        fname=name,
    )


####################################################################################################
# dino-eval
####################################################################################################


@main.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    help="Path to the evaluation file (JSONL format).",
)
@click.option(
    "--output-path",
    type=click.Path(exists=True),
    default="./outputs/latex/",
    help="Path to save the generated LaTeX tables.",
)
@click.option(
    "--meta-path",
    type=click.Path(exists=True),
    required=False,
    help="Path to the original dataset to extract metadata with.",
)
def dino_eval(input_path, output_path, meta_path):
    """
    Read the evaluation file and return a DataFrame.
    """

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    # Read evaluation
    with open(input_path, "r") as f:
        data = json.load(f)

    accuracy = data.pop("accuracy", None)
    logger.info(f"Accuracy: {accuracy:.3f}" if accuracy is not None else "No accuracy")

    # Build dataframe
    COLUMNS = [
        "accuracy_per_class",
        "precision_per_class",
        "recall_per_class",
    ]
    data = {k: v for k, v in data.items() if k in COLUMNS}
    df = pd.DataFrame.from_dict(data)
    df.index.name = "class"

    # Meta and fname
    name = None

    if meta_path is not None:
        # Get path to the data associated to the metrics input file
        for fname in ["generated_images.zip", "dataset.zip"]:
            if (input_path.parent / fname).exists():
                path = input_path.parent / fname
                break
        meta_path = Path(meta_path).expanduser()
        logger.info(f"Meta path: \n\t{path}\n{meta_path}\t")

        meta_df, name = metadata_versus_df(meta_path, path, method="lp")

        meta_df = format_mean_std(meta_df, decimals=3)
        meta_df = format_colnames(meta_df)
        save_latex_table(
            meta_df,
            out_dir=output_path,
            fname=name,
        )

    name = extract_dataset_name(input_path) if name is None else name
    name = name.replace("meta", "dino") if "meta" in name else f"dino_{name}"

    if "Labelset" in meta_df.columns:
        df["Labelset"] = meta_df["Labelset"]

    # Format DataFrame and write LaTeX table
    df = format_mean_std(df, decimals=3)
    df = format_colnames(df)

    save_latex_table(
        df,
        out_dir=output_path,
        fname=name,
    )


####################################################################################################
# evaluation
####################################################################################################


@main.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    help="Path to the evaluation file (JSONL format).",
)
@click.option(
    "--output-path",
    type=click.Path(exists=True),
    default="./outputs/latex/",
    help="Path to save the generated LaTeX tables.",
)
def evaluation(input_path, output_path):
    """
    Reads the evaluation file and returns a DataFrame.
    """

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    model_name = input_path.parent.name
    fname = f"eval-{model_name}"
    fname = fname.replace("-", "_")
    print(f"Model name: {model_name}")

    with open(input_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "class"

    # Format.
    df = format_mean_std(df, decimals=3)
    df = format_colnames(df)

    # Write into latex file.
    save_latex_table(df, out_dir=output_path, fname=fname)


if __name__ == "__main__":
    main()
