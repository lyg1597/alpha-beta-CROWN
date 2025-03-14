import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb
from gatenet import GateNet
from gatenet2 import GateNet2
# from loss import loss
from PIL import Image
import numpy as np

# Custom dataset for image and pose labels
class PoseDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.image_files = sorted(self.img_dir.glob("*.png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # Load image (no transformations needed since images are already 640x480)
        image = Image.open(img_path).convert("RGB")
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0,1]

        # Load labels from txt file (x y z roll pitch yaw)
        with open(label_path, "r") as f:
            label = torch.tensor([float(value) for value in f.readline().split()], dtype=torch.float32)

        return image, label

# Training function
def train_model(
        model,
        device,
        dataset,
        epochs=10,
        batch_size=4,
        learning_rate=1e-4,
        val_percent=0.1,
        save_checkpoint=True,
        checkpoint_dir="./checkpoints",  # Path to save checkpoints
        checkpoint_path=None,  # Path to load checkpoint from
        amp=False,
        gradient_clipping=1.0
):
    # Split dataset into training and validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Define optimizer, loss function, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    model.to(device)

    # Resume from checkpoint if provided
    start_epoch = 1
    global_step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1
        print(f"✅ Resuming training from epoch {starting_epoch}")

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize Weights & Biases (wandb)
    experiment = wandb.init(project='GateNet-Pose-Estimator', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_percent=val_percent)
    )

    logging.info("Starting training...")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        avg_val_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for images, true_poses in train_loader:
                images, true_poses = images.to(device, dtype=torch.float32), true_poses.to(device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    pred_poses = model(images)
                    loss_value = criterion(pred_poses, true_poses)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss_value).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss_value.item()

                # Log training loss
                experiment.log({'train loss': loss_value.item(), 'step': global_step, 'epoch': epoch})
                pbar.set_postfix(**{'loss (batch)': loss_value.item()})

                # Perform validation at intervals
                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and global_step % division_step == 0:
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_images, val_poses in val_loader:
                            val_images, val_poses = val_images.to(device), val_poses.to(device)
                            val_pred = model(val_images)
                            val_loss += criterion(val_pred, val_poses).item()

                    val_loss /= len(val_loader)
                    scheduler.step(val_loss)
                    logging.info(f'Validation Loss: {val_loss}')
                   
                    # Log validation loss
                    experiment.log({'validation loss': val_loss, 'epoch': epoch})
                    avg_val_loss = avg_val_loss + val_loss
            # avg_val_loss = avg_val_loss/

        # Save checkpoint in the specified directory
        if save_checkpoint and (avg_val_loss < best_val_loss):
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }
            torch.save(checkpoint, checkpoint_file)
            logging.info(f"Checkpoint saved at {checkpoint_file}, avg_val_loss: {avg_val_loss}")
            best_val_loss = avg_val_loss

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train GateNet for aircraft pose estimation")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img-dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--label-dir', type=str, required=True, help='Directory with pose labels')
    parser.add_argument('--save-checkpoint', action='store_true', default=True, help='Save model checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints", help='Directory to save checkpoints')  # NEW
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to resume training from a checkpoint')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision training')
    parser.add_argument('--avg_pool', action='store_true', default=True, help='Use avg pooling or max pooling. True for avg pooling.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset = PoseDataset(args.img_dir, args.label_dir)

    config = {
        'input_shape': (3, 1280, 960),
        'output_shape': (6,),  # X, Y, Z, yaw, pitch, roll
        'l2_weight_decay': 1e-4,
        'batch_norm_decay': 0.99,
        'batch_norm_epsilon': 1e-3
    }
    if args.avg_pool:
        model = GateNet2(config)
    else:
        model = GateNet(config)

    train_model(
        model,
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_checkpoint=args.save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,  # NEW: Custom directory
        checkpoint_path=args.checkpoint_path,
        amp=args.amp
    )

if __name__ == "__main__":
    main()