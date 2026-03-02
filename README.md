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

## Dataset Restructuring

For easier analysis, we restructure the dataset based on a selected subset of attributes. This restructuring consists of redefining the label space according to the chosen attributes.
We provide two alternative label transformation strategies:

### Label Powerset Transformation

Given a selection of labels from the available dataset attributes, we define a new label space $\mathcal{L}$.  
The dataset is then partitioned into **non-overlapping classes**, where each class corresponds to a unique combination of labels from the powerset of $\mathcal{L}$.

In other words, each distinct subset of selected labels defines a separate class, ensuring that each sample belongs to exactly one class.

### Multi-Hot Encoding

Alternatively, the selected labels can be encoded using **multi-hot encoding**.  
In this representation, each sample is associated with a binary vector indicating the presence or absence of each selected label.

This approach preserves the multi-label structure of the data instead of converting it into mutually exclusive classes.

### Command

```bash
uv run restructure_dataset.py \
    --data ./data/CelebA/img_align_celeba.zip \
    --labels ./data/CelebA/list_attr_celeba.txt \
    --selection Bangs --selection Eyeglasses --selection Male --selection Smiling \
    --method br \
    --outdir ./data/CelebA/edited/
```

With default parameters, data directory should look like:

```
./data/CelebA
├── list_attr_celeba.txt
├── img_align_celeba.zip
├── img_align_celeba
└── edited
    ├── BR-50eb47c0
    │   └── dataset_raw.zip
    ├── LP-50eb47c0
    │   └── dataset_raw.zip
    ...
```

The dataset structure is built to be compatible with [EDM project](https://github.com/NVlabs/edm) and can be ingested directly using `dataset_tool.py`. More information on the tool in [`python dataset_tool.py --help`](https://github.com/NVlabs/edm/blob/main/docs/dataset-tool-help.txt).

```bash
uv run dataset_tool.py \
    --source ./data/CelebA/edited/<dataset-name>/dataset_raw.zip \
    --dest  ./data/CelebA/edited/<dataset-name>/dataset.zip \
    --transform center-crop \
    --resolution 64x64
```

Alternatively (might not be equivalent after reworking) run:

```bash
git clone git@github.com:NVlabs/edm.git
cd edm
conda env create -f environment.yml -n edm
conda activate edm
python dataset_tool.py \
    --source ./data/CelebA/edited/<dataset-name>/dataset_raw.zip \
    --dest  ./data/CelebA/edited/<dataset-name>/dataset.zip \
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
