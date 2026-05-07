import copy
import random
import shutil

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
    def __init__(self, task="is_inserted"):
        super().__init__()
        # Input: 7 (pose1) + 7 (pose2) = 14
        if task == "is_in_corner":
            in_dim = 28
        else:
            in_dim = 14
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
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
    def __init__(self, task="is_inserted"):
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
            path = data_path / f"{item}_rawpose_dataset_1.h5"
            if not os.path.exists(path):
                print(f"⚠️ Warning: File {path} not found. Skipping.")
                continue

            print(f"📦 Loading data from: {path}")
            with h5py.File(path, "r") as f:
                for g_name in f.keys():
                    group = f[g_name]

                    # Load this demo's data
                    base = group["base_pose"][:]  # (T, 7)
                    if task == "is_in_corner":
                        moving = group["obstacle_pos"][:]
                    else:
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
    def __init__(self, data_dir, furniture_list, mesh_dir, n_points=1024, task="is_inserted"):
        self.n_points = n_points
        self.task = task  # Store the task name

        all_base_poses, all_moving_poses, all_y = [], [], []
        self.sample_type = []

        self.base_mesh_points = {}
        self.moving_mesh_points = {}  # For "is_inserted", this is the bulb/leg

        # Specifically for the corner task
        self.obstacle_front_points = None
        self.obstacle_side_points = None

        # Pre-load obstacle meshes if needed for the task
        if self.task == "is_in_corner":
            self.obstacle_front_points = self._load_mesh(f"{mesh_dir}/obstacle_front.obj")
            self.obstacle_side_points = self._load_mesh(f"{mesh_dir}/obstacle_side.obj")

        for item in furniture_list:
            # 1. Load the Meshes (Base meshes are always needed)
            if item == "lamp":
                self.base_mesh_points[item] = self._load_mesh(f"{mesh_dir}/lamp/lamp_base.obj")
                if self.task != "is_in_corner":
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
                continue

            with h5py.File(file_path, "r") as f:
                for g_name in f.keys():
                    group = f[g_name]
                    base = group["base_pose"][:]

                    # Logic switch for input types
                    if self.task == "is_in_corner":
                        # Use obstacle_pos (21 dims)
                        moving = group["obstacle_pos"][:]
                        labels = group[self.task][:]
                    else:
                        # Standard moving_pose (7 dims)
                        moving = group["moving_pose"][:]
                        labels = group[self.task][:]
                        if self.task == "is_inserted":
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

        self.base_poses = np.concatenate(all_base_poses, axis=0)
        self.moving_poses = np.concatenate(all_moving_poses, axis=0)  # Can be (N, 7) or (N, 21)
        self.y = np.concatenate(all_y, axis=0)
        print(self.y.shape)
        self.sample_type = np.array(self.sample_type)

    def __getitem__(self, idx):
        f_type = self.sample_type[idx]
        p_base = self.base_mesh_points[f_type].clone()
        p_base = self.transform_points(p_base, self.base_poses[idx])

        if self.task == "is_in_corner":
            # 21-dim pose: [front_pose(7), left_pose(7), right_pose(7)]
            full_pose = self.moving_poses[idx]

            # Transform Front Obstacle
            p_front = self.obstacle_front_points.clone()
            p_front = self.transform_points(p_front, full_pose[0:7])

            # Transform Left Obstacle (Side mesh)
            p_left = self.obstacle_side_points.clone()
            p_left = self.transform_points(p_left, full_pose[7:14])

            # Transform Right Obstacle (Side mesh)
            p_right = self.obstacle_side_points.clone()
            p_right = self.transform_points(p_right, full_pose[14:21])

            # Combine all 3 obstacles into one "moving" cloud
            p_moving = torch.cat([p_front, p_left, p_right], dim=0)

            # Subsample to keep point cloud size consistent (optional)
            # If PointNet expects 2048 total, and base is 1024, p_moving should be 1024
            indices = torch.randperm(len(p_moving))[:self.n_points]
            p_moving = p_moving[indices]
        else:
            # Standard single mesh logic
            p_moving = self.moving_mesh_points[f_type].clone()
            p_moving = self.transform_points(p_moving, self.moving_poses[idx])

        # Combine into a single scene point cloud
        scene_pc = torch.cat([p_base, p_moving], dim=0)
        scene_pc -= p_base.mean(dim=0)  # Zero-center
        scene_pc = scene_pc.transpose(0, 1)

        label = 0.99 if self.y[idx] == 1 else 0.01
        return {"x": scene_pc, "y": label}

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


def visualize_model_predictions(model, dataloader, num_samples=3, device=torch.device("cpu"), pc=False):
    model.eval()
    model = model.to(device)

    # 1. Get Batch
    batch = next(iter(dataloader))
    inputs = batch["x"].to(device)  # Can be Poses (B, 14/28) or PointCloud (B, 3, N)
    labels = batch["y"].to(device)

    # 2. Run Inference
    with torch.no_grad():
        preds = model(inputs)

    # 3. Pick random indices
    print(len(inputs))
    indices = np.random.choice(len(inputs), num_samples, replace=False)
    visualized = 0
    for idx in range(len(inputs)):
        if visualized > num_samples + 5:
            break
        # Check for misclassifications or random samples
        if (preds[idx] > 0.5 and not labels[idx] > 0.5) or \
                (preds[idx] < 0.5 and not labels[idx] < 0.5) or idx in indices:
            visualized += 1
            data = inputs[idx].cpu().numpy()
            target = labels[idx].cpu().item()
            prediction = preds[idx].cpu().item()
            save_name = f"{idx}_{prediction:.2f}_{target}.glb"

            if pc:
                # --- POINT CLOUD VISUALIZATION ---
                # PointNet input is usually (3, N), trimesh needs (N, 3)
                if data.shape[0] == 3:
                    data = data.T

                pcd = trimesh.PointCloud(vertices=data)
                # Apply a default color to the point cloud
                pcd.colors = [100, 100, 250, 255]
                pcd.export(f"pc_check_{save_name}")
                print(f"✅ Exported PointCloud to pc_check_{save_name}")

            else:
                # --- MESH / POSE VISUALIZATION ---
                base_path = "~/code/furniture-bench/furniture_bench/assets/furniture/mesh/lamp/lamp_base.obj"

                if data.shape[0] == 28:
                    # Obstacle Bench Logic
                    obs_front = "~/code/furniture-bench/furniture_bench/assets/furniture/mesh/obstacle_front.obj"
                    obs_side = "~/code/furniture-bench/furniture_bench/assets/furniture/mesh/obstacle_side.obj"

                    moving_paths = [obs_front, obs_side, obs_side]
                    moving_poses = [data[7:14], data[14:21], data[21:28]]
                    mode_prefix = "obstacle_base"
                else:
                    # Default Lamp/Furniture Logic
                    bulb_path = "~/code/furniture-bench/furniture_bench/assets/furniture/mesh/lamp/lamp_bulb.obj"
                    moving_paths = [bulb_path]
                    moving_poses = [data[7:14]]
                    mode_prefix = "lamp_bulb"

                visualize_with_meshes(
                    base_path,
                    moving_paths,
                    data[:7],
                    moving_poses,
                    f"{mode_prefix}_{save_name}"
                )


def visualize_with_meshes(base_mesh_path, moving_mesh_paths, base_pose, moving_poses, save_path):
    def get_matrix(pose):
        mat = np.eye(4)
        mat[:3, 3] = pose[:3]
        mat[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        return mat

    base_mesh = to_single_mesh(trimesh.load(base_mesh_path))
    base_mesh.apply_transform(get_matrix(base_pose))
    base_mesh.visual.face_colors = [150, 150, 220, 150]  # Muted Blue

    all_meshes = [base_mesh]

    # 2. Process all moving parts (Bulb or 3 Obstacles)
    colors = [
        [220, 150, 150, 255],  # Soft Red
        [150, 220, 150, 255],  # Soft Green
        [220, 220, 150, 255]  # Soft Yellow
    ]

    for i, (m_path, m_pose) in enumerate(zip(moving_mesh_paths, moving_poses)):
        m_mesh = to_single_mesh(trimesh.load(m_path))
        m_mesh.apply_transform(get_matrix(m_pose))
        # Assign unique color if multiple obstacles, else standard red
        m_mesh.visual.face_colors = colors[i % len(colors)]
        all_meshes.append(m_mesh)

    # 3. Export Scene
    scene = trimesh.Scene(all_meshes)
    scene.export(save_path)
    print(f"✅ Exported 3D scene to {save_path}")


def to_single_mesh(loaded_geometry):
    """Helper to ensure we have a single mesh object from trimesh.load"""
    if isinstance(loaded_geometry, trimesh.Scene):
        return trimesh.util.concatenate([
            geom for geom in loaded_geometry.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ])
    return loaded_geometry

def train_single_network(data_path, network_name, network, furniture, pointcloud=False):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    if not pointcloud:
        dataset = ValuationDataset(data_path, furniture, task=network_name)
    else:
        mesh_dir = "~/code/optionatari/furniture-bench/furniture_bench/assets/furniture/mesh"
        dataset = PointCloudValuationDataset(data_path, furniture, mesh_dir=mesh_dir, task=network_name)

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
    bs = 128
    lr = 5e-3 if not pointcloud else 1e-3

    epochs = 250 if not pointcloud else 50

    wandb.init(entity="alinaboehm", project="pretrain_valuation", config={"bs": bs, "lr": lr, "epochs": epochs, "pointcloud": pointcloud})
    furnitre_str = ""
    for item in sorted(furniture):
        furnitre_str += f"{item}_"
    wandb.run.name = f"fb_{furnitre_str}pc{pointcloud}_{network_name}_bs{bs}_lr{lr}"
    dir_name = "pc" if pointcloud else "mlp"
    checkpoint_dir = data_path / "checkpoints" / dir_name / furnitre_str / network_name
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

    visualize_model_predictions(network, test_loader, num_samples=6, device=device, pc=pointcloud)

    print(f"Final: Epoch {epoch + 1} | Loss: {avg_loss:.6f}")
    # Save checkpoint
    if os.path.exists(checkpoint_dir / "latest.pt"):
        os.rename(checkpoint_dir / "latest.pt", checkpoint_dir / "old_latest.pt")
    print(os.listdir(checkpoint_dir))
    torch.save(network.state_dict(), checkpoint_dir / "latest.pt")
    print(f"Finished {network_name} pretraining.")
    wandb.finish()
    shutil.rmtree("wandb/")

def eval_only(data_path, network, network_name):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    if not pointcloud:
        dataset = ValuationDataset(data_path, furniture, task=network_name)
    else:
        mesh_dir = "~/code/optionatari/furniture-bench/furniture_bench/assets/furniture/mesh"
        dataset = PointCloudValuationDataset(data_path, furniture, mesh_dir=mesh_dir, task=network_name)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # for reproducibility
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    bs = 128

    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
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
    print(acc_all, acc_pos, acc_neg)
    print(num_samples_eval, num_samples_eval["1"])
    visualize_model_predictions(network, test_loader, num_samples=10, device=device, pc=pointcloud)

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
    task = "is_screwed_in"
    if pointcloud:
        network = PointNet()
    else:
        network = ValNet(task)
    furniture = ["lamp"]
    train_single_network(Path("./"), network_name=task, network=network, furniture=furniture, pointcloud=pointcloud)

    state_dict = torch.load("checkpoints/pc/lamp_/is_screwed_in/latest.pt")
    network.load_state_dict(state_dict)
    eval_only(Path("./"), network, task)
