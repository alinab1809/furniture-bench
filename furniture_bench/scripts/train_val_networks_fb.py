import copy
import random

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import h5py
import trimesh
from scipy.spatial.transform import Rotation as R

import os
import numpy as np

class ValNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 7 (pose1) + 7 (pose2) = 14
        self.mlp = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Assuming x is already concatenated (batch, 14)
        return self.mlp(x.float()).squeeze(-1)

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x shape: (Batch, 3, Num_Points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Global Max Pooling: The core of PointNet
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features).squeeze(-1)

class ValuationDataset(Dataset):
    def __init__(self, data_path, furniture, task="is_inserted"):
        base_poses = []
        moving_poses = []
        ys = []

        for item in furniture:
            path = data_path / f"{item}_rawpose_dataset.h5"
            if not os.path.exists(path):
                print(f"⚠️ Warning: File {path} not found. Skipping.")
                continue

            print(f"📦 Loading data from: {path}")
            with h5py.File(path, "r") as f:
                print(f.keys())
                for g_name in f.keys():
                    group = f[g_name]

                    # Load this demo's data
                    base = group["base_pose"][:]  # (T, 7)
                    moving = group["moving_pose"][:]  # (T, 7)
                    labels = group[task][:]  # (T,)

                    if task == "is_inserted":
                        labels_dist = group["is_inserted_dist"][:]

                        indices_to_use = labels == labels_dist
                        base = base[indices_to_use]
                        moving = moving[indices_to_use]
                        labels = labels[indices_to_use]
                    # --- REDUNDANCY CHECK ---
                    # 1. Combine into 14D vector
                    combined = np.hstack((base, moving))

                    # 2. Round to remove floating point jitter (e.g., 0.0001 precision)
                    # This treats nearly-identical poses as the same
                    combined_rounded = np.round(combined, decimals=4)

                    # 3. Find unique indices
                    # return_index=True gives us the first occurrence of each unique state
                    _, unique_indices = np.unique(combined_rounded, axis=0, return_index=True)

                    # Sort indices to maintain the temporal flow of the demo (optional but cleaner)
                    unique_indices.sort()

                    # 4. Filter data
                    base_poses.append(base[unique_indices])
                    moving_poses.append(moving[unique_indices])
                    ys.append(labels[unique_indices])

        # 2. Concatenate everything from all files into long arrays
        # This turns a list of (Steps, 7) arrays into one big (Total_Steps, 7) array
        all_base_poses = np.concatenate(base_poses, axis=0)
        all_moving_poses = np.concatenate(moving_poses, axis=0)
        all_y_labels = np.concatenate(ys, axis=0)

        # 3. Create the 14D input (N, 14) and labels
        self.x = torch.from_numpy(np.hstack((all_base_poses, all_moving_poses))).float()
        self.y = torch.from_numpy(all_y_labels).float()

        print(f"✅ Loaded total of {len(self.y)} frames from {len(furniture)} files.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": 0.99 if self.y[idx] == 1 else 0.01}


class PointCloudValuationDataset(Dataset):
    def __init__(self, data_dir, furniture_list, mesh_dir, n_points=1024):
        self.n_points = n_points

        all_base_poses, all_moving_poses, all_y = [], [], []
        self.sample_type = []  # To track which furniture each index belongs to

        # Initialize dictionaries to hold canonical points
        self.base_mesh_points = {}
        self.moving_mesh_points = {}

        for item in furniture_list:
            # 1. Load the Meshes for this furniture type
            if item == "lamp":
                self.base_mesh_points[item] = self._load_mesh(f"{mesh_dir}/lamp/lamp_base.obj")
                self.moving_mesh_points[item] = self._load_mesh(f"{mesh_dir}/lamp/lamp_bulb.obj")
            elif item == "one_leg":
                self.base_mesh_points[item] = self._load_mesh(f"{mesh_dir}/square_table/square_table_top.obj")
                self.moving_mesh_points[item] = self._load_mesh(f"{mesh_dir}/square_table/square_table_leg.obj")
            elif item == "stool":
                self.base_mesh_points[item] = self._load_mesh(f"{mesh_dir}/{item}/{item}_seat.obj")
                self.moving_mesh_points[item] = self._load_mesh(f"{mesh_dir}/{item}/{item}_leg1.obj")
            else:
                self.base_mesh_points[item] = self._load_mesh(f"{mesh_dir}/{item}/{item}_top.obj")
                self.moving_mesh_points[item] = self._load_mesh(f"{mesh_dir}/{item}/{item}_leg.obj")

            # 2. Load the Poses from H5
            file_path = data_dir / f"{item}_rawpose_dataset.h5"
            if not os.path.exists(file_path):
                print(f"⚠️ {file_path} not found. Skipping.")
                continue

            with h5py.File(file_path, "r") as f:
                for g_name in f.keys():
                    group = f[g_name]

                    # Load this demo's data
                    base = group["base_pose"][:]  # (T, 7)
                    moving = group["moving_pose"][:]  # (T, 7)
                    labels = group["is_inserted"][:]  # (T,)
                    labels_dist = group["is_inserted_dist"][:]

                    indices_to_use = labels == labels_dist
                    base = base[indices_to_use]
                    moving = moving[indices_to_use]
                    labels = labels[indices_to_use]

                    combined = np.hstack((base, moving))

                    # multiple demos with low randomness in start position -> we might have redundant samples
                    combined_rounded = np.round(combined, decimals=4)
                    _, unique_indices = np.unique(combined_rounded, axis=0, return_index=True)
                    unique_indices.sort()

                    all_base_poses.append(base[unique_indices])
                    all_moving_poses.append(moving[unique_indices])
                    all_y.append(labels[unique_indices])
                    self.sample_type.extend([item] * len(unique_indices))

        # 3. Final Concatenation
        self.base_poses = np.concatenate(all_base_poses, axis=0)
        self.moving_poses = np.concatenate(all_moving_poses, axis=0)
        self.y = np.concatenate(all_y, axis=0)
        self.sample_type = np.array(self.sample_type)
        print(f"✅ Loaded total of {len(self.y)} frames from {len(furniture_list)} files.")

    def __getitem__(self, idx):
        # Identify which furniture geometry to use
        f_type = self.sample_type[idx]

        # Get canonical points for this specific furniture
        p_base = self.base_mesh_points[f_type].clone()
        p_moving = self.moving_mesh_points[f_type].clone()

        # Transform them using the world pose at this index
        p_base = self.transform_points(p_base, self.base_poses[idx])
        p_moving = self.transform_points(p_moving, self.moving_poses[idx])

        # Combine into a single scene point cloud
        scene_pc = torch.cat([p_base, p_moving], dim=0)

        # Zero-center the cloud around the base for better generalization
        scene_pc -= p_base.mean(dim=0)

        # PointNet expects (Channel, N_Points) -> (3, 2048)
        scene_pc = scene_pc.transpose(0, 1)

        label = 0.99 if self.y[idx] == 1 else 0.01
        return {"x": scene_pc, "y": torch.tensor(label, dtype=torch.float32)}

    def _load_mesh(self, path):
        mesh = to_single_mesh(trimesh.load(path))
        # Sample points uniformly from the surface
        points = mesh.sample(self.n_points)
        return torch.from_numpy(points).float()

    def transform_points(self, points, pose):
        """ Applies the transformation P' = R*P + t """
        # pose is [x, y, z, qx, qy, qz, qw]
        t = torch.from_numpy(pose[:3]).float()
        q = pose[3:]
        rot_mat = torch.from_numpy(R.from_quat(q).as_matrix()).float()

        # (N, 3) @ (3, 3) + (3)
        return (points @ rot_mat.T) + t

    def __len__(self):
        return len(self.y)


def to_single_mesh(loaded_obj):
    """If trimesh loaded a Scene, merge it into one Mesh. Otherwise, return it."""
    if isinstance(loaded_obj, trimesh.Scene):
        # Concatenate all geometries in the scene into a single mesh
        return trimesh.util.concatenate([
            geom for geom in loaded_obj.geometry.values()
        ])
    return loaded_obj

def visualize_with_meshes(base_mesh_path, moving_mesh_path, base_pose, moving_pose, save_path):
    # 1. Load and force into Mesh objects
    base_mesh = to_single_mesh(trimesh.load(base_mesh_path))
    moving_mesh = to_single_mesh(trimesh.load(moving_mesh_path))

    # 2. Transformation Matrix Helper
    def get_matrix(pose):
        mat = np.eye(4)
        mat[:3, 3] = pose[:3]  # [x, y, z]
        # Scipy expects [x, y, z, w]. Ensure your dataset matches!
        mat[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        return mat

    # 3. Apply transformations
    base_mesh.apply_transform(get_matrix(base_pose))
    moving_mesh.apply_transform(get_matrix(moving_pose))

    # 4. Professional Muted Coloring
    # Blue for Base (RGBA)
    base_mesh.visual.face_colors = [150, 150, 220, 180]
    # Red for Moving (RGBA)
    moving_mesh.visual.face_colors = [220, 150, 150, 255]

    # 5. Export as GLB (for local 3D viewing)
    scene = trimesh.Scene([base_mesh, moving_mesh])
    export_name = save_path.replace(".png", ".glb")
    scene.export(export_name)
    print(f"✅ Exported 3D scene to {export_name}")

def visualize_model_predictions(model, dataloader, num_samples=3):
    model.eval()

    # Get one batch
    batch = next(iter(dataloader))
    inputs = batch["x"]
    labels = batch["y"]

    # Run Inference
    with torch.no_grad():
        preds = model(inputs)

    # Pick random indices
    indices = np.random.choice(len(inputs), num_samples, replace=False)

    for idx in range(len(inputs)):
        if (preds[idx] > 0.5 and not labels[idx] > 0.5) or (preds[idx] < 0.5 and not labels[idx] < 0.5) or idx in indices:
            data = inputs[idx].numpy()

            target = labels[idx].item()
            prediction = preds[idx].item()

            base_path = "~/code/furniture-bench/furniture_bench/assets/furniture/mesh/lamp/lamp_base.obj"
            moving_path = "~/code/furniture-bench/furniture_bench/assets/furniture/mesh/lamp/lamp_bulb.obj"
            visualize_with_meshes(base_path, moving_path, data[:7], data[7:14], f"check_alignment_{idx}_{prediction}_{target}.glb")

def train_single_network(data_path, network_name, network, furniture, pointcloud=False):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    if not pointcloud:
        dataset = ValuationDataset(data_path, furniture, task=network_name)
    else:
        mesh_dir = "~/code/furniture-bench/furniture_bench/assets/furniture/mesh"
        dataset = PointCloudValuationDataset(data_path, furniture, mesh_dir=mesh_dir)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # for reproducibility
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_indices = train_dataset.indices
    train_labels = dataset.y[train_indices].flatten()

    class_sample_count = torch.tensor([
        (train_labels < 0.5).sum(),
        (train_labels > 0.5).sum()
    ])
    print(f"Train class samples neg/pos: {class_sample_count}")

    weights = 1. / class_sample_count.float()
    train_samples_weights = torch.tensor([weights[1] if t > 0.5 else weights[0] for t in train_labels])

    train_sampler = WeightedRandomSampler(
        weights=train_samples_weights,
        num_samples=len(train_samples_weights),
        replacement=True
    )

    # 3. Hyperparameters
    # For to_platform (12 points), we use a tiny batch size.
    bs = 128
    lr = 4e-3 if not pointcloud else 1e-3

    epochs = 300 if not pointcloud else 150

    wandb.init(entity="alinaboehm", project="pretrain_valuation", config={"bs": bs, "lr": lr, "epochs": epochs, "pointcloud": pointcloud})
    furnitre_str = ""
    for item in sorted(furniture):
        furnitre_str += f"{item}_"
    wandb.run.name = f"fb_{furnitre_str}pc{pointcloud}_{network_name}_bs{bs}_lr{lr}"
    dir_name = "pc" if pointcloud else "mlp"
    checkpoint_dir = data_path / "checkpoints" / dir_name / furnitre_str
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Starting pretraining for {network_name}...")
    global_step = 0
    for epoch in range(epochs):
        network.train()
        epoch_loss = 0
        successes = {"all": 0, "1": 0, "0": 0}
        num_samples = {"all": 0, "1": 0, "0": 0}
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device).float()

            # Forward
            out = network(x)
            loss = loss_fn(out, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            preds = out > 0.5
            targets = y > 0.5

            # 2. Compare them and sum the matches
            # .item() converts the single-value tensor to a Python integer
            successes["all"] += (preds == targets).sum().item()
            num_samples["all"] += y.size(0)

            # 2. Positive Samples (y == 1)
            pos_mask = (targets == 1)
            num_pos = pos_mask.sum().item()
            if num_pos > 0:
                successes["1"] += (preds[pos_mask] == targets[pos_mask]).sum().item()
                num_samples["1"] += num_pos

            # 3. Negative Samples (y == 0)
            neg_mask = (targets == 0)
            num_neg = neg_mask.sum().item()
            if num_neg > 0:
                successes["0"] += (preds[neg_mask] == targets[neg_mask]).sum().item()
                num_samples["0"] += num_neg

        # --- Print Results at the end of epoch ---
        acc_all = successes["all"] / num_samples["all"] if num_samples["all"] > 0 else 0
        acc_pos = successes["1"] / num_samples["1"] if num_samples["1"] > 0 else 0
        acc_neg = successes["0"] / num_samples["0"] if num_samples["0"] > 0 else 0

        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch+1})
        wandb.log({"train_accuracy": acc_all, "epoch": epoch+1})
        wandb.log({"train_accuracy_pos": acc_pos, "epoch": epoch + 1})
        wandb.log({"train_accuracy_neg": acc_neg, "epoch": epoch + 1})

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")
            # Save checkpoint
            torch.save(network.state_dict(), checkpoint_dir / "latest.pt")

    network.eval()
    loss_eval = 0
    successes_eval = {"all": 0, "1": 0, "0": 0}
    num_samples_eval = {"all": 0, "1": 0, "0": 0}
    for batch in test_loader:
        with torch.no_grad():
            x = batch["x"].to(device)
            y = batch["y"].to(device).float()

            # Forward
            out = network(x)
            loss_eval += loss_fn(out, y)

            preds = out > 0.5
            targets = y > 0.5

            # 2. Compare them and sum the matches
            # .item() converts the single-value tensor to a Python integer
            successes_eval["all"] += (preds == targets).sum().item()
            num_samples_eval["all"] += y.size(0)
            pos_mask = (targets == 1)
            num_pos = pos_mask.sum().item()
            if num_pos > 0:
                successes_eval["1"] += (preds[pos_mask] == targets[pos_mask]).sum().item()
                num_samples_eval["1"] += num_pos

            neg_mask = (targets == 0)
            num_neg = neg_mask.sum().item()

            if num_neg > 0:
                successes_eval["0"] += (preds[neg_mask] == targets[neg_mask]).sum().item()
                num_samples_eval["0"] += num_neg

    acc_all = successes_eval["all"] / num_samples_eval["all"] if num_samples_eval["all"] > 0 else 0
    acc_pos = successes_eval["1"] / num_samples_eval["1"] if num_samples_eval["1"] > 0 else 0
    acc_neg = successes_eval["0"] / num_samples_eval["0"] if num_samples_eval["0"] > 0 else 0
    avg_loss = loss_eval / len(test_loader)
    wandb.log({"eval_loss": avg_loss})
    wandb.log({"eval_accuracy": acc_all, "epoch": epoch + 1})
    wandb.log({"eval_accuracy_pos": acc_pos, "epoch": epoch + 1})
    wandb.log({"eval_accuracy_neg": acc_neg, "epoch": epoch + 1})

    visualize_model_predictions(network, test_loader, num_samples=6)

    print(f"Final: Epoch {epoch + 1} | Loss: {avg_loss:.6f}")
    # Save checkpoint
    torch.save(network.state_dict(), checkpoint_dir / "latest.pt")
    print(f"Finished {network_name} pretraining.")
    wandb.finish()


def inspect_dataset(file_path, demo_idx=0, num_samples=25):
    # In your inspect_dataset function
    with h5py.File(file_path, "r") as f:
        print([k for k in f.keys()])
        demo = f[f"demo_{demo_idx}"]
        indices = np.random.choice(len(demo["is_inserted"]), (num_samples,), replace=False)
        indices.sort()
        # Load data
        imgs = demo["color_image2"][indices]
        labels = demo["is_inserted"][indices]
        labels_dist = demo["is_inserted_dist"][indices]
        # We'll calculate distance on the fly or load it if you saved it
        m_poses = demo["moving_pose"][:num_samples]
        b_poses = demo["base_pose"][:num_samples]

        print(f"Inspecting {num_samples} frames from {demo_idx}...")

        canvas_items = []
        for i in range(num_samples):
            img = imgs[i].copy()

            # Ensure BGR for OpenCV saving
            if img.shape[0] == 3:  # If (C, H, W)
                img = img.transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw Label and Index on the image
            label_color = (0, 255, 0) if labels[i] == 1 else (0, 0, 255)
            status = f"{labels[i]} / {labels_dist[i]}"

            cv2.putText(img, f"ID:{i} {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

            canvas_items.append(img)

        # Tile the images into a 5x5 grid
        grid_size = int(np.sqrt(num_samples))
        rows = []
        for r in range(grid_size):
            row = np.hstack(canvas_items[r * grid_size: (r + 1) * grid_size])
            rows.append(row)

        grid = np.vstack(rows)
        output_name = f"inspect_demo{demo_idx}.jpg"
        cv2.imwrite(output_name, grid)
        print(f"✅ Verification grid saved to: {output_name}")


# Usage:
# inspect_dataset("lamp_dataset.h5", demo_idx=0)

if __name__ == "__main__":
    from pathlib import Path
    import sys

    # inspect_dataset("models/pretrain/fb/data/lamp_rawpose_dataset.h5", demo_idx=0)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pc", action="store_true")
    args = parser.parse_args()
    pointcloud = args.pc
    if pointcloud:
        network = PointNet()
    else:
        network = ValNet()
    furniture = ["lamp"]
    train_single_network(Path("./"), network_name="is_screwed_in", network=network, furniture=furniture, pointcloud=pointcloud)
