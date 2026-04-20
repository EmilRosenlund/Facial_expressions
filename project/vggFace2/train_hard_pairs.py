import random


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np
import sys
import os
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import FER2013Dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return True, rank, world_size, local_rank, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0

def mixup_batch(inputs, labels, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0)).to(inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[idx]
    return mixed, labels, labels[idx], lam

# MLP head for classification
class MLPHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn4 = nn.BatchNorm1d(hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


# Combined model: VGGFace2 (InceptionResnetV1) backbone + MLP head
class VGGFace2WithMLP(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_rate=0.4):
        super().__init__()
        backbone = InceptionResnetV1(pretrained='vggface2')
        # Unfreeze last blocks: last 2 mixed_6 and all mixed_7 layers
        for name, param in backbone.named_parameters():
            param.requires_grad = False
        for name, param in backbone.named_parameters():
            if any([k in name for k in ["block8", "mixed_7a", "mixed_6a", "conv2d_4b", "conv2d_4a"]]):
                param.requires_grad = True
        self.dropout = nn.Dropout(dropout_rate)
        self.backbone = backbone
        self.head = MLPHead(512, hidden_size, num_classes, dropout_rate)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.head(x)
        return x

# Custom dataset for end-to-end training
class FER2013ImageDataset(Dataset):
    def __init__(self, split="train", transform=None, augment=False, stage2=False):
        self.dataset = FER2013Dataset(debug=False)
        self.transform = transform
        self.augment = augment
        self.samples = []
        self.stage2 = stage2
        for expression in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            data = self.dataset.load_data(split=split, expression=expression)
            for label, img in data:
                self.samples.append((img.copy(), label))
                # For training, add 3 augmentations per image
                if self.augment and split == "train":
                    for aug_img in self._augmentations(img.copy()):
                        self.samples.append((aug_img, label))
        

    def _augmentations(self, img):
        # Squeeze (random horizontal scaling)
        squeeze_factor = random.uniform(0.7, 1.0)
        squeeze_img = img.resize((int(img.width * squeeze_factor), img.height)).resize((img.width, img.height))
        # Rotate
        rotate_img = img.rotate(random.uniform(-20, 20))
        # Add Gaussian noise
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 10, np_img.shape)
        noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img)
        return [squeeze_img, rotate_img, noisy_img]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img.convert('RGB'))
        return img, label

def smooth_labels(labels, num_classes, smoothing=0.1):
    with torch.no_grad():
        confidence = 1.0 - smoothing
        label_shape = (labels.size(0), num_classes)
        smoothed = torch.full(label_shape, smoothing / (num_classes - 1), device=labels.device)
        smoothed.scatter_(1, labels.unsqueeze(1), confidence)
    return smoothed


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, rank=0, distributed=False, train_sampler=None):
    best_val_loss = float('inf')
    model.to(device)
    for epoch in range(num_epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        train_batches = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # MixUp augmentation
            mixed_inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels)
            outputs = model(mixed_inputs)
            # No log_softmax here, CrossEntropyLoss expects logits
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_batches += 1

        if distributed:
            train_stats = torch.tensor([total_loss, train_batches], dtype=torch.float64, device=device)
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            avg_train_loss = (train_stats[0] / train_stats[1].clamp(min=1.0)).item()
        else:
            avg_train_loss = total_loss / max(1, train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
                val_batches += 1

        if distributed:
            val_stats = torch.tensor([val_loss, val_batches], dtype=torch.float64, device=device)
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            avg_val_loss = (val_stats[0] / val_stats[1].clamp(min=1.0)).item()
        else:
            avg_val_loss = val_loss / max(1, val_batches)

        if is_main_process(rank):
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)

        # Save best checkpoint
        if avg_val_loss < best_val_loss and is_main_process(rank):
            best_val_loss = avg_val_loss
            print(f"New best model found at epoch {epoch+1} with val loss {avg_val_loss:.4f}. Saving model...v6")
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save(model_to_save.state_dict(), "best_model_v6.pth")

    if is_main_process(rank):
        save_path = "model.pth"
        model_to_save = model.module if isinstance(model, DDP) else model
        torch.save(model_to_save.state_dict(), save_path)


def train_arcface(model, arcface_head, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, rank=0, distributed=False, train_sampler=None, use_mixup=False):
    best_val_loss = float('inf')
    model.to(device)
    arcface_head.to(device)

    for epoch in range(num_epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        arcface_head.train()
        total_loss = 0.0
        train_batches = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_mixup:
                mixed_inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels)
                forward_inputs = mixed_inputs
            else:
                forward_inputs = inputs

            captured = {}
            model_ref = model.module if isinstance(model, DDP) else model

            def capture_fc5_input(_, fc5_input):
                captured["embeddings"] = fc5_input[0]

            hook = model_ref.head.fc5.register_forward_pre_hook(capture_fc5_input)
            _ = model(forward_inputs)
            hook.remove()

            embeddings = captured["embeddings"]
            if use_mixup:
                logits_a = arcface_head(embeddings, labels_a)
                logits_b = arcface_head(embeddings, labels_b)
                loss = lam * criterion(logits_a, labels_a) + (1 - lam) * criterion(logits_b, labels_b)
            else:
                logits = arcface_head(embeddings, labels)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_batches += 1

        if distributed:
            train_stats = torch.tensor([total_loss, train_batches], dtype=torch.float64, device=device)
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            avg_train_loss = (train_stats[0] / train_stats[1].clamp(min=1.0)).item()
        else:
            avg_train_loss = total_loss / max(1, train_batches)

        model.eval()
        arcface_head.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                captured = {}
                model_ref = model.module if isinstance(model, DDP) else model

                def capture_fc5_input(_, fc5_input):
                    captured["embeddings"] = fc5_input[0]

                hook = model_ref.head.fc5.register_forward_pre_hook(capture_fc5_input)
                _ = model(val_inputs)
                hook.remove()

                val_embeddings = captured["embeddings"]
                val_logits = arcface_head(val_embeddings, val_labels)
                loss = criterion(val_logits, val_labels)
                val_loss += loss.item()
                val_batches += 1

        if distributed:
            val_stats = torch.tensor([val_loss, val_batches], dtype=torch.float64, device=device)
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            avg_val_loss = (val_stats[0] / val_stats[1].clamp(min=1.0)).item()
        else:
            avg_val_loss = val_loss / max(1, val_batches)

        if is_main_process(rank):
            print(f'[ArcFace] Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss and is_main_process(rank):
            best_val_loss = avg_val_loss
            print(f"New best ArcFace model at epoch {epoch+1} with val loss {avg_val_loss:.4f}. Saving model...v6")
            model_to_save = model.module if isinstance(model, DDP) else model
            arcface_to_save = arcface_head.module if isinstance(arcface_head, DDP) else arcface_head
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "arcface_state_dict": arcface_to_save.state_dict(),
                },
                "best_model_v6_arcface.pth",
            )

    if is_main_process(rank):
        model_to_save = model.module if isinstance(model, DDP) else model
        arcface_to_save = arcface_head.module if isinstance(arcface_head, DDP) else arcface_head
        torch.save(
            {
                "model_state_dict": model_to_save.state_dict(),
                "arcface_state_dict": arcface_to_save.state_dict(),
            },
            "model_v6_arcface.pth",
        )


if __name__ == "__main__":
    stage1 = True
    stage2 = False





    hidden_size = 128
    num_classes = 7
    dropout_rate = 0.4
    random_seed = 42
    torch.manual_seed(random_seed)
    distributed, rank, world_size, local_rank, device = setup_distributed()
    if is_main_process(rank):
        print(f"Using device: {device} | distributed={distributed} | world_size={world_size}")
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    if is_main_process(rank):
        print("Data transforms defined.")
    # Datasets and loaders with 3x augmentation for training
    full_train_dataset = FER2013ImageDataset(split="train", transform=transform, augment=True)
    val_split = 0.2
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, sampler=val_sampler, num_workers=2, pin_memory=True)
    if is_main_process(rank):
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    # Model
    model = VGGFace2WithMLP(hidden_size, num_classes, dropout_rate)
    if is_main_process(rank):
        print(f"Model initialized. With VGGFace2 backbone and MLP head. and {sum(p.numel() for p in model.parameters())} parameters.")
    class_counts = torch.tensor([3995, 436, 4097, 7215, 4965, 4830, 3171], dtype=torch.float)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights.to(device))

        # Early backbone (frozen): lr=0, mid backbone: lr=1e-5, last block: lr=1e-4
    last_block_names = ["block8", "mixed_7a"]
    mid_block_names = ["mixed_6a", "conv2d_4b", "conv2d_4a"]
    # Group parameters
    last_block_params = [p for n, p in model.backbone.named_parameters() if any(k in n for k in last_block_names)]
    mid_block_params = [p for n, p in model.backbone.named_parameters() if any(k in n for k in mid_block_names) and not any(k in n for k in last_block_names)]
    frozen_param_names = [n for n, _ in model.backbone.named_parameters() if not any(k in n for k in (last_block_names + mid_block_names))]
    frozen_params = [p for n, p in model.backbone.named_parameters() if not any(k in n for k in (last_block_names + mid_block_names))]
    if is_main_process(rank):
        print("[DEBUG] Last block params unfrozen:")
        for n, p in model.backbone.named_parameters():
            if any(k in n for k in last_block_names):
                print("  ", n)
        print("[DEBUG] Mid block params unfrozen:")
        for n, p in model.backbone.named_parameters():
            if any(k in n for k in mid_block_names) and not any(k in n for k in last_block_names):
                print("  ", n)
        print("[DEBUG] Frozen block params:")
        for n, p in model.backbone.named_parameters():
            if n in frozen_param_names:
                print("  ", n)

    backbone_param_groups = [
        {"params": frozen_params, "lr": 0},
        {"params": mid_block_params, "lr": 1e-5},
        {"params": last_block_params, "lr": 1e-4},
    ]
    if distributed:
        model = DDP(model.to(device), device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)

    num_epochs = 300
    if stage1 == True:
        if is_main_process(rank):
            print("Starting Stage 1: Training from scratch")
        model_for_stage1 = model.module if isinstance(model, DDP) else model
        arcface_margin_head_stage1 = ArcMarginProduct(
            in_features=hidden_size // 4,
            out_features=num_classes,
            s=30.0,
            m=0.50,
        )
        if distributed:
            arcface_margin_head_stage1 = DDP(
                arcface_margin_head_stage1.to(device),
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )
        else:
            arcface_margin_head_stage1 = arcface_margin_head_stage1.to(device)

        optimizer_stage1 = optim.AdamW(
            backbone_param_groups + [
                {"params": model_for_stage1.head.parameters(), "lr": 1e-4},
                {"params": arcface_margin_head_stage1.parameters(), "lr": 5e-4},
            ],
            weight_decay=5e-4
        )
        scheduler_stage1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage1, mode='min', factor=0.5, patience=5)
        train_arcface(
            model,
            arcface_margin_head_stage1,
            train_loader,
            val_loader,
            criterion,
            optimizer_stage1,
            scheduler_stage1,
            num_epochs,
            device,
            rank=rank,
            distributed=distributed,
            train_sampler=train_sampler,
            use_mixup=True,
        )

    if is_main_process(rank):
        print("Training completed.")
    if stage2 == True:
        if is_main_process(rank):
            print("Starting Stage 2: Fine-tuning on hard pairs")
        epochs = 100
        model_for_stage2 = model.module if isinstance(model, DDP) else model
        model_for_stage2.dropout.p = 0.1
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0, weight=weights.to(device))
        arcface_margin_head = ArcMarginProduct(
            in_features=hidden_size // 4,
            out_features=num_classes,
            s=30.0,
            m=0.50,
        )
        if distributed:
            arcface_margin_head = DDP(
                arcface_margin_head.to(device),
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )
        else:
            arcface_margin_head = arcface_margin_head.to(device)

        backbone_param_groups = [
        {"params": frozen_params, "lr": 0},
        {"params": mid_block_params, "lr": 1e-5},
        {"params": last_block_params, "lr": 1e-4},
    ]
        optimizer = optim.SGD(backbone_param_groups + [
            {"params": model_for_stage2.head.parameters(), "lr": 1e-4},
            {"params": arcface_margin_head.parameters(), "lr": 5e-4},
        ], momentum=0.9, weight_decay=5e-4)
        scheduler_stage2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        with torch.no_grad():
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(torch.load("best_model_v5.pth", map_location=device))
        # Reuse the same Stage 1 split and only filter it for hard classes.
        hard_labels = {0, 2, 4, 6} # angry, fear, sad, neutral
        hard_train_indices = [
            idx for idx in train_dataset.indices
            if full_train_dataset.samples[idx][1] in hard_labels
        ]
        hard_val_indices = [
            idx for idx in val_dataset.indices
            if full_train_dataset.samples[idx][1] in hard_labels
        ]
        hard_train_dataset = torch.utils.data.Subset(full_train_dataset, hard_train_indices)
        hard_val_dataset = torch.utils.data.Subset(full_train_dataset, hard_val_indices)
        hard_train_sampler = DistributedSampler(hard_train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        hard_val_sampler = DistributedSampler(hard_val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        hard_train_loader = DataLoader(hard_train_dataset, batch_size=64, shuffle=(hard_train_sampler is None), sampler=hard_train_sampler, num_workers=2, pin_memory=True)
        hard_val_loader = DataLoader(hard_val_dataset, batch_size=64, shuffle=False, sampler=hard_val_sampler, num_workers=2, pin_memory=True)
        if is_main_process(rank):
            print(f"Hard pair train dataset size: {len(hard_train_dataset)}, Hard pair val dataset size: {len(hard_val_dataset)}")
        train_arcface(
            model,
            arcface_margin_head,
            hard_train_loader,
            hard_val_loader,
            criterion,
            optimizer,
            scheduler_stage2,
            epochs,
            device,
            rank=rank,
            distributed=distributed,
            train_sampler=hard_train_sampler,
        )
        if is_main_process(rank):
            print("Stage 2 training completed.")

    cleanup_distributed(distributed)