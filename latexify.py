"""This file aims to read results and convert them to LaTeX tables."""

import json
import logging
from collections import Counter
from pathlib import Path

import click
import pandas as pd

from datatools.multilabel import read_lp_dataset_meta
from datatools.utils import extract_dataset_name, read_dataset_meta, zip_labels

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
    counts = (
        dict(sorted(Counter(tuple(row) for row in labels).items(), key=lambda x: x[0]))
        if not isinstance(labels[0], int)
        else dict(sorted(Counter(labels).items()))
    )

    df = pd.DataFrame(counts.items(), columns=["Label", "Count"])
    df = df.set_index("Label")

    df["Proportion"] = df["Count"] / df["Count"].sum() * 100

    for col_name, mapping in kwargs.items():
        if mapping is not None:
            assert len(mapping) == len(df), f"Length mismatch for {col_name}."
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


def metadata_versus_df(data_path, gen_path):

    data_path = Path(data_path).expanduser()
    gen_path = Path(gen_path).expanduser()

    assert data_path.exists(), f"Data path {data_path} does not exist."
    assert gen_path.exists(), f"Generated path {gen_path} does not exist."

    logger.info(f"Data path: {data_path}")
    logger.info(f"Generated path: {gen_path}")

    data_name = extract_dataset_name(data_path)
    gen_name = extract_dataset_name(gen_path)

    data_meta = read_dataset_meta(data_path)
    gen_meta = read_dataset_meta(gen_path)

    data_labels_df = build_label_df(
        data_meta["labels"],
        Labelset=data_meta.get("labelsets", None),
    )
    gen_labels_df = build_label_df(
        gen_meta["labels"],
        Labelset=gen_meta.get("labelsets", None),
    )

    print(data_labels_df)  # FIX
    print(gen_labels_df)  # FIX

    metadata_df = [data_labels_df, gen_labels_df]
    metadata_source = ["original", "generated"]

    merged_df = merge_dataframes_with_duplicate_handling(metadata_df, metadata_source)
    merged_df.index.name = "Label"

    print(merged_df)  # FIX

    # Signed deltas: generated minus original.
    # if {"o._Count", "g._Count"}.issubset(merged_df.columns):
    #     merged_df["delta_count"] = merged_df["g._Count"] - merged_df["o._Count"]
    if {"o._Proportion", "g._Proportion"}.issubset(merged_df.columns):
        merged_df[r"$\Delta$ Proportion (\%)"] = (
            (merged_df["g._Proportion"] - merged_df["o._Proportion"])
            / merged_df["o._Proportion"]
            * 100
        )

    return merged_df, f"meta-{data_name}-{gen_name}"


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


def save_latex_table(df: pd.DataFrame, outdir: str | Path, fname: str, **kwargs):
    """
    Save a DataFrame as a LaTeX table.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    outdir : str | Path
        Directory where the LaTeX file will be saved.
    fname : str
        Name of the LaTeX file (without extension).
    """
    logger.info("Generating LaTeX table for DataFrame:")
    print(df)

    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    out = outdir / f"{fname}.tex"

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


####################################################################################################
# meta-vs
####################################################################################################


@main.command()
@click.option("--data", "-d",   help="Path to the original dataset (directory or zip file).", metavar="DIR|ZIP",  type=click.Path(exists=True))  # fmt: skip
@click.option("--gen", "-g",    help="Path to the generated dataset (directory or zip file).", metavar="DIR|ZIP", type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o", help="Path to save the generated LaTeX tables.", metavar="DIR",                   type=click.Path(), default="out-latexify")  # fmt: skip
def meta_vs(data, gen, outdir):
    """
    Compare the original dataset with the generated one in terms of label distribution.

    Args:
        data_path (str | Path): Path to the original dataset (directory or zip file).
        gen_path (str | Path): Path to the generated dataset (directory or zip file).
        method (str): Method used for fine-tuning the DINOv2 model (Binary Relevance or Label Powerset).
        output_path (str | Path): Path to save the generated LaTeX tables.
    """

    df, name = metadata_versus_df(data, gen)
    df = format_mean_std(df, decimals=3)
    df = format_colnames(df)

    if outdir is not None:
        outdir = Path(outdir).expanduser() / "meta-vs"
        outdir.mkdir(parents=True, exist_ok=True)
        save_latex_table(df, outdir=outdir, fname=name)

    return df, name


####################################################################################################
# dino-eval
####################################################################################################


@main.command()
@click.option("--eval", "-e",   help="Path to the evaluation file (JSONL format).", metavar="JSONL", type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o", help="Path to save the generated LaTeX tables.", metavar="DIR",      type=click.Path(exists=True), default="out-latexify")  # fmt: skip
def dino_eval(eval, outdir):
    """
    Read the evaluation file and return a DataFrame.
    """

    eval_path = Path(eval).expanduser()
    name = extract_dataset_name(eval_path, suffix=".jsonl")

    # Read evaluation
    with open(eval_path, "r") as f:
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

    # Format DataFrame and write LaTeX table
    df = format_mean_std(df, decimals=3)
    df = format_colnames(df)

    if outdir is not None:
        outdir = Path(outdir).expanduser() / "dino-eval"
        outdir.mkdir(parents=True, exist_ok=True)
        save_latex_table(df, outdir=outdir, fname=f"dino-{name}")

    return df, name


####################################################################################################
# evaluation
####################################################################################################


@main.command()
@click.option("--eval", "-e",   help="Path to the evaluation file (JSONL format).", metavar="JSONL", type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o", help="Path to save the generated LaTeX tables.", metavar="DIR",      type=click.Path(exists=True), default="out-latexify")  # fmt: skip
def gen_eval(eval, outdir):
    """
    Reads the evaluation file and returns a DataFrame.
    """

    eval_path = Path(eval).expanduser()

    name = extract_dataset_name(eval_path, depth=2, suffix=".jsonl")

    with open(eval_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "class"

    # Format DataFrame and write LaTeX table
    df = format_mean_std(df, decimals=3)
    df = format_colnames(df)

    if outdir is not None:
        outdir = Path(outdir).expanduser() / "gen-eval"
        outdir.mkdir(parents=True, exist_ok=True)
        save_latex_table(df, outdir=outdir, fname=f"eval-{name}")

    return df, name


####################################################################################################
# comparison
####################################################################################################


@main.command()
@click.option("--gen", "-g",    help="Path to the evaluation file (JSONL format) of generation.", metavar="JSONL",     type=click.Path(exists=True))  # fmt: skip
@click.option("--clf", "-c",    help="Path to the evaluation file (JSONL format) of classification.", metavar="JSONL", type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o", help="Path to save the generated LaTeX tables.", metavar="DIR",                        type=click.Path(exists=True), default="out-latexify")  # fmt: skip
def comparison(gen, clf, outdir):
    """
    Reads the evaluation file and returns a DataFrame.
    """

    gen_df, gen_name = gen_eval.callback(gen, None)
    clf_df, clf_name = dino_eval.callback(clf, None)

    assert gen_name == clf_name, "Dataset names do not match"
    name = gen_name

    # Filter gen_df
    gen_cols = ["Num Features", "FID", "P Inc", "P Dino"]
    gen_df = gen_df[[col for col in gen_cols if col in gen_df.columns]]
    gen_df.index = gen_df.index.astype(str)

    # Filter clf_df
    clf_cols = ["Recall", "Labelset"]
    clf_df = clf_df[[col for col in clf_cols if col in clf_df.columns]]
    clf_df.index = clf_df.index.astype(str)

    # Merge the two dataframes on the index (class)
    df = gen_df.merge(
        clf_df,
        left_index=True,
        right_index=True,
        how="outer",
    )
    df = pd.concat(
        [
            df.loc[["overall"]],
            df.drop("overall").sort_index(key=lambda x: x.astype(int)),
        ]
    )

    # Write into latex file.
    if outdir is not None:
        outdir = Path(outdir).expanduser() / "comparison"
        outdir.mkdir(parents=True, exist_ok=True)
        save_latex_table(df, outdir=outdir, fname=f"comp-{name}")

    return df, name


if __name__ == "__main__":
    main()
