import argparse
import os
import torch
import numpy as np
from gatenet import GateNet
from gatenet2 import GateNet2
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_model(checkpoint_path, device, args):
    """Loads the trained GateNet model from a checkpoint."""
    config = {
        'input_shape': (3, 480, 640),
        'output_shape': (6,),
        'batch_norm_decay': 0.99,
        'batch_norm_epsilon': 1e-3
    }
    if args.avg_pool:
        model = GateNet2(config)
    else:
        model = GateNet(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """Loads and preprocesses the input image for the model."""
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def evaluate_model(model, test_image_dir, test_label_dir, device):
    """Evaluate model performance and generate error plots."""
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith(".png")])
   
    actual_values = []
    predicted_values = []
   
    for img_file in image_files:
        img_path = os.path.join(test_image_dir, img_file)
        label_path = os.path.join(test_label_dir, img_file.replace(".png", ".txt"))
       
        # Load ground truth
        with open(label_path, "r") as f:
            actual_pose = np.array([float(value) for value in f.readline().split()])

        # Preprocess image
        image = preprocess_image(img_path).to(device)

        # Run inference
        with torch.no_grad():
            predicted_pose = model(image).cpu().numpy().flatten()

        # Store values
        actual_values.append(actual_pose)
        predicted_values.append(predicted_pose)

    # Convert lists to numpy arrays
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    # Plot errors for each pose parameter
    labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    for i in range(6):
        sorted_indices = np.argsort(actual_values[:,i])
        sorted_actual = actual_values[sorted_indices,i]
        sorted_pred = predicted_values[sorted_indices,i]

        plt.figure(figsize=(8, 6))
        plt.plot(sorted_actual , sorted_pred, marker="o", linestyle="-", alpha=0.7, label="predicted {labels[i]}")
        plt.xlabel(f"Actual {labels[i]}")
        plt.ylabel(f"Predicted ({labels[i]}")
        plt.title(f"Prediction vs Actual {labels[i]}")
        plt.legend()
        plt.grid()
        plt.savefig(f"test_plotyt_{labels[i].lower()}.png")
        plt.close()

    print("✅ Error plots saved as 'error_plot_x.png', 'error_plot_y.png', etc.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GateNet on a test dataset.")
    parser.add_argument('--test-image-dir', type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument('--test-label-dir', type=str, required=True, help="Path to the folder containing test labels")
    parser.add_argument('--checkpoint-path', type=str, required=True, help="Path to the trained model checkpoint (.pth)")
    parser.add_argument('--avg_pool', action='store_true', default=True, help='Use avg pooling or max pooling. True for avg pooling.')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(args.checkpoint_path, device, args)

    # Run evaluation
    evaluate_model(model, args.test_image_dir, args.test_label_dir, device)

if __name__ == "__main__":
    main()
