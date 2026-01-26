# EDM Multi-Label

This repo explores conditional generative model evaluation with Multi-Label prediction.
This illustrative example is run with CelebA dataset and a selection of available attributes.

## Dataset

At the moment the project revolves around [CelebA dataset](https://arxiv.org/abs/1411.7766). Other datasets might be added later.

For automatic download, run:

[//]: # "FIX: UPDATE THIS, remove ALIgned"

```bash
uv run download.py --dir ~/data --aligned 1 --extension jpg
cd ~/data/CelebA
unzip img_align_celeba.zip
```

or [download dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) by hand.

## Multi-Label Classification with Label Powerset

For a given selection of labels among all available lables, defining a label space $\mathcal{L}$, the original dataset is split into non-overlapping classes, or label combinations, corresponding to the labelsets in the powerset of $\mathcal{L}$. Each subset is stored in a separate folder containing all images corresponding to that specific Labelset.

```bash
uv run build_subsets.py \
    --data-dir ~/data/CelebA \
    --img-dir img_align_celeba \
    --attr-file list_attr_celeba.txt \
    --out-dir powerset_partitions \
    --attrs Bangs Eyeglasses Male Smiling
```

With default parameters, data directory should look like:

```
~/data/CelebA
├── list_attr_celeba.txt
├── img_align_celeba.zip
├── img_align_celeba
└── powerset_partitions
```

```
~/data/CelebA/powerset_partitions
    ├── labelspace_hash
    │   ├── labelset1_hash
    │   │   ├── 000003.jpg
    │   │   └── 000007.jpg
    │   ├── labelset2_hash
    │   │   ├── 000001.jpg
    │   │   └── 000002.jpg
    │   ├── ...
    │   └── metadata.json
```

The dataset structure is built to be compatible with [EDM project](https://github.com/NVlabs/edm) and can be ingested directly using `dataset_tool.py`. More information on the tool in [`python dataset_tool.py --help`](https://github.com/NVlabs/edm/blob/main/docs/dataset-tool-help.txt).

```bash
uv run dataset_tool.py \
    --source ~/data/CelebA/powerset_partitions/<labelspace_hash> \
    --dest  ~/data/CelebA/edm-64x64/<labelspace_hash>.zip \
    --transform center-crop \
    --resolution 64x64
```

Alternatively (and equivalently) run:

```bash
git clone git@github.com:NVlabs/edm.git
cd edm
conda env create -f environment.yml -n edm
conda activate edm
python dataset_tool.py \
    --source ~/data/CelebA/powerset_partitions/<labelspace_hash> \
    --dest  ~/data/CelebA/edm-64x64/<labelspace_hash>.zip \
    --transform center-crop \
    --resolution 64x64
```

## Calculating FID

The Fréchet Inception Distance reference statistics for the dataset is computed with [EDM project](https://github.com/NVlabs/edm):

[//]: # "FIX: UPDATE THIS, do we add the file or let it in other project ?"

```bash
git clone git@github.com:NVlabs/edm.git
cd edm
conda env create -f environment.yml -n edm
conda activate edm
python fid.py ref --data datasets/my-dataset.zip --dest fid-refs/my-dataset.npz
```
