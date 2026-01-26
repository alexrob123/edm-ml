import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm

from training import dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

# CONFIG
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
NUM_CLASSES = 2
# NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data = "ffhq-64x64"  # Dataset name
# data = "afhqv2-64x64"
path_data = f"data/{data}"  # Path to the dataset
path_model = f"checkpoints/finetuned_dino_{data}.pth"  # Path to save the model

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")


# DATASETS
full_dataset = dataset.ImageFolderDataset(
    path=f"data/{data}.zip",
    # path="data/ffhqcond.zip",
    resolution=64,
    use_labels=True,
    max_size=None,
    xflip=False,
)
indices = torch.randperm(len(full_dataset))
split = int(0.9 * len(full_dataset))
train_indices, val_indices = indices[:split], indices[split:]

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# MODEL
model = AutoModelForImageClassification.from_pretrained(
    "facebook/dinov2-base", num_labels=NUM_CLASSES
)

model = nn.DataParallel(model)
print(model.module)
# Freeze DINOv2 backbone
for param in model.module.dinov2.parameters():
    param.requires_grad = False

# Unfreeze last 4 layers of the transformer
for layer in model.module.dinov2.encoder.layer[-4:]:
    for param in layer.parameters():
        param.requires_grad = True
model = model.to(DEVICE)

# TRAINING SETUP
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    [
        {"params": model.module.classifier.parameters(), "lr": 1e-3},
        {"params": model.module.dinov2.encoder.layer[-4:].parameters(), "lr": 1e-4},
    ],
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

progress_bar = tqdm.tqdm(range(NUM_EPOCHS), desc="Training Progress")
best_eval = 1e100
# TRAIN LOOP
for epoch in progress_bar:
    train_loss = 0
    progress_bar_train = tqdm.tqdm(
        train_loader, leave=False, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )
    for x, y in progress_bar_train:
        x, y = x.to(DEVICE).to(torch.float32) / 255.0, y.to(DEVICE)
        y = torch.argmax(y, dim=1)  # <- convert from one-hot to class index
        optimizer.zero_grad()
        inputs = processor(images=x, return_tensors="pt", do_rescale=False).to(DEVICE)
        outputs = model(**inputs)
        loss = criterion(outputs.logits, y)
        train_loss += loss.item() / x.size(0)
        loss.backward()
        optimizer.step()
        progress_bar_train.set_postfix({"Loss": loss.item() / x.size(0)})
    scheduler.step()
    total_val_loss = 0
    n_images = 0
    n_correct = 0
    for x, y in tqdm.tqdm(val_loader, leave=False, desc="Validation"):
        x, y = x.to(DEVICE).to(torch.float32) / 255.0, y.to(DEVICE)
        y = torch.argmax(y, dim=1)  # <- convert from one-hot to class index
        with torch.no_grad():
            inputs = processor(images=x, return_tensors="pt", do_rescale=False).to(
                DEVICE
            )
            outputs = model(**inputs)
            val_loss = criterion(outputs.logits, y)
            _, preds = torch.max(outputs.logits, 1)
            n_images += x.size(0)
            n_correct += (preds == y).sum().item()
        total_val_loss += val_loss.item() * x.size(0)
    accuracy = n_correct / n_images

    progress_bar.set_postfix(
        {
            "Acc": 100 * accuracy,
            "TLoss": train_loss / len(train_loader),
            "VLoss": total_val_loss / n_images,
        }
    )
    if total_val_loss / n_images < best_eval:
        best_eval = total_val_loss / n_images
        torch.save(model.module.state_dict(), path_model)
