"""This file aims to read results and convert them to LaTeX tables."""

import json
from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd

args = SimpleNamespace(
    eval_file="~/data/CelebA/edited/LP-50eb47c0-model3/evaluation.jsonl"
)

args.eval_file = Path(args.eval_file).expanduser()
print(f"Evaluation file path: {args.eval_file}")


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


@main.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    help="Path to the evaluation file (JSONL format).",
)
@click.option(
    "--output-path",
    type=click.Path(exists=True),
    help="Path to the evaluation file (JSONL format).",
)
def evaluation(input_path, output_path):
    """
    Reads the evaluation file and returns a DataFrame.
    """

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    model_name = input_path.parent.name
    fname = f"tab:eval-{model_name}"
    print(f"Model name: {model_name}")

    with open(input_path, "r") as f:
        data = json.load(f)

        import pandas as pd

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "class"

    # Column order.
    cols = [
        "fid",
        "P",
        "P_std",
        "R",
        "R_std",
        "D",
        "D_std",
        "C",
        "C_std",
        "num_features",
    ]
    df = df[cols]

    # Formatting for LaTeX.
    df_fmt = df.copy()

    # Combine mean ± std.
    for metric in ["P", "R", "D", "C"]:
        df_fmt[metric] = (
            df[metric].map(lambda x: f"{x:.3f}")
            + r" $\pm$ "
            + df[f"{metric}_std"].map(lambda x: f"{x:.3f}")
        )
    df_fmt = df_fmt.drop(columns=["P_std", "R_std", "D_std", "C_std"], errors="ignore")

    # Column names.
    df_fmt = df_fmt.rename(columns=lambda c: c.replace("_", " ").title()).rename(
        columns={"Fid": "FID"}
    )

    # Styler for LaTeX output.
    styler = (
        df_fmt.style.format(decimal=".", thousands=",", precision=2)
        # .hide(axis="index")
    )

    # Create LaTeX table.
    latex = styler.to_latex(
        hrules=True,
    )
    print(latex)

    # Write into latex file.
    out = Path(output_path) / f"{fname}.tex"
    out.write_text(latex)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
