import h5py
import isaacgym
import torch
import numpy as np
import gym
from pathlib import Path
from furniture_bench.envs.initialization_mode import Randomness
import furniture_bench.controllers.control_utils as C
import sys
import os
import cv2
from scipy.spatial.transform import Rotation as R

class PoseDataCollector:
    def __init__(self, furniture="lamp", data_path="_rawpose_dataset_screw.h5"):
        data_path = furniture + data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.furniture = furniture
        self.env = gym.make(
            "FurnitureSimFull-v0",
            furniture=furniture,
            num_envs=1,
            headless=True,
            np_step_out=False,
            channel_first=False,  # Set to False to get (H, W, 3) for easy saving
            randonmess=Randomness.LOW,
            record=True
        )
        self.data_path = data_path

    def get_transformation_matrix(self, pose):
        """Converts [x, y, z, qx, qy, qz, qw] to 4x4 matrix."""
        mat = np.eye(4)
        mat[:3, 3] = pose[:3]
        # SciPy expects [x, y, z, w] which is the FurnitureBench default
        mat[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        return mat

    def collect_dataset(self, num_demos=10):
        # Get the target relative pose from the furniture definition
        # This is the 'Golden Truth' for a successful insertion
        target_rel_poses = self.env.unwrapped.furniture.assembled_rel_poses

        with h5py.File(self.data_path, "w") as f:
            demo = 0
            num_failed_demos = 0
            while demo < num_demos:
                print(f"Starting Demo {demo + 1}/{num_demos}")
                demo_key = f"demo_{demo}"
                group = f.create_group(demo_key)

                base_pose_ds = group.create_dataset("base_pose",
                                                     shape=(0, 7),
                                                     maxshape=(None,7),
                                                     dtype='float32', chunks=(1,7))
                moving_pose_ds = group.create_dataset("moving_pose",
                                                     shape=(0, 7),
                                                     maxshape=(None,7),
                                                     dtype='float32', chunks=(1,7))

                # img_ds1 = group.create_dataset("color_image1",
                #                               shape=(0, 224, 224, 3),
                #                               maxshape=(None, 224, 224, 3),
                #                               dtype='uint8', chunks=(1, 224, 224, 3))
                # img_ds2 = group.create_dataset("color_image2",
                #                                shape=(0, 224, 224, 3),
                #                                maxshape=(None, 224, 224, 3),
                #                                dtype='uint8', chunks=(1, 224, 224, 3))

                inserted_label_ds = group.create_dataset("is_inserted",
                                                shape=(0,),
                                                maxshape=(None,),
                                                dtype='int32', chunks=(1,))

                inserted_label_dist_ds = group.create_dataset("is_inserted_dist",
                                                         shape=(0,),
                                                         maxshape=(None,),
                                                         dtype='int32', chunks=(1,))

                corner_label_ds = group.create_dataset("is_in_corner",
                                                shape=(0,),
                                                maxshape=(None,),
                                                dtype='int32', chunks=(1,))

                screwed_label_ds = group.create_dataset("is_screwed_in",
                                                shape=(0,),
                                                maxshape=(None,),
                                                dtype='int32', chunks=(1,))

                obs = self.env.reset()
                done = False
                step_idx = 0
                failed_validation = False

                unwrapped = self.env.unwrapped
                if self.furniture in ["lamp", "round_table"]:
                    # for isinserted, we only look at the bulb and we don't want to track the hood once bulb assembly is done
                    part_idx1, part_idx2 = unwrapped.furniture.should_be_assembled[unwrapped.assemble_idx]
                moved_to_corner = False
                screwed_label = 0

                while not done:
                    # 1. Get action from scripted agent
                    action, skill_complete = unwrapped.get_assembly_action()

                    # 2. Get Part Indices for the current task
                    # We use unwrapped to access the core FurnitureBench variables
                    if self.furniture not in ["lamp", "round_table"]:
                        part_idx1, part_idx2 = unwrapped.furniture.should_be_assembled[unwrapped.assemble_idx]

                    # 3. Get RAW Simulator Poses (Option A - No AprilTag transform)
                    raw_poses, _ = unwrapped._get_parts_poses(sim_coord=True)
                    raw_poses = raw_poses.squeeze().cpu().numpy()

                    dim = 7
                    base_state = raw_poses[part_idx1 * dim: (part_idx1 + 1) * dim]
                    moving_state = raw_poses[part_idx2 * dim: (part_idx2 + 1) * dim]

                    # 4. Calculate GEOMETRIC Label
                    T_base = self.get_transformation_matrix(base_state)
                    T_moving = self.get_transformation_matrix(moving_state)
                    T_current_rel = np.linalg.inv(T_base) @ T_moving

                    # 3. Check distance against every possible valid pose in the list
                    target_mats_list = target_rel_poses[(part_idx1, part_idx2)]
                    part_idx1, part_idx2 = self.furniture.should_be_assembled[self.assemble_idx]
                    distances = []
                    for target_mat in target_mats_list:
                        # Ensure the target is a numpy array (in case it's a list or tensor)
                        if isinstance(target_mat, torch.Tensor):
                            target_mat = target_mat.cpu().numpy()
                        elif isinstance(target_mat, list):
                            target_mat = np.array(target_mat)

                        # Calculate distance to this specific hole
                        d = np.linalg.norm(T_current_rel[:3, 3] - target_mat[:3, 3])
                        distances.append(d)

                    # 4. The final label is 1 if we are close to ANY of the holes
                    min_dist = min(distances)
                    label_dist = 1 if min_dist < 0.02 else 0

                    moving_part = self.env.furniture.parts[part_idx2]

                    current_state = moving_part._state

                    # "screw" is a dummy state when the screwing is done - we should switch from isinserted to isscrewedin there
                    inserted_label = 1 if ("screw" in current_state or "pre_grasp" in current_state and current_state != "screw") else 0

                    if inserted_label == 1 and label_dist != 1:
                        # agent says it is inserted but distance values say otherwise
                        print(
                            f"❌ Label mismatch at step {step_idx} (State: {current_state}, Dist: {min_dist:.4f}m). Restarting demo...")
                        failed_validation = True
                        num_failed_demos += 1
                        if num_failed_demos > 14:
                            if self.furniture == "stool":
                                print("failed too many times, exiting ", num_failed_demos)
                                self.env.close()
                                sys.exit()
                        break
                    elif label_dist == 1 and "insert" in current_state:
                        inserted_label = 1

                    # 5. Store current observation and calculated states
                    base_pose_ds.resize((step_idx + 1, 7))
                    base_pose_ds[step_idx] = base_state
                    moving_pose_ds.resize((step_idx + 1, 7))
                    moving_pose_ds[step_idx] = moving_state

                    # img_ds1.resize((step_idx + 1, 224, 224, 3))
                    # img_ds1[step_idx] = obs["color_image1"].cpu().numpy().copy()
                    # img_ds2.resize((step_idx + 1, 224, 224, 3))
                    # img_ds2[step_idx] = obs["color_image2"].cpu().numpy().copy()

                    inserted_label_ds.resize((step_idx + 1,))
                    inserted_label_ds[step_idx] = inserted_label

                    inserted_label_dist_ds.resize((step_idx + 1,))
                    inserted_label_dist_ds[step_idx] = label_dist

                    corner_label = 0
                    if not moved_to_corner:
                        if self.env.furniture.parts[part_idx1].pre_assemble_done:
                            corner_label = 1
                            moved_to_corner = True

                    corner_label_ds.resize((step_idx + 1,))
                    corner_label_ds[step_idx] = corner_label

                    part1_pose = C.to_homogeneous(
                        self.rb_states[part_idx1][0][:3],
                        C.quat2mat(self.rb_states[part_idx1][0][3:7]),
                    )
                    part2_pose = C.to_homogeneous(
                        self.rb_states[part_idx2][0][:3],
                        C.quat2mat(self.rb_states[part_idx1][0][3:7]),
                    )
                    rel_pose = torch.linalg.inv(part1_pose) @ part2_pose
                    assembled_rel_poses = self.furniture.assembled_rel_poses[(part_idx1, part_idx2)]
                    if self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses):
                        screwed_label = 1

                    screwed_label_ds.resize((step_idx + 1,))
                    screwed_label_ds[step_idx] = screwed_label

                    if screwed_label == 1:
                        # inserted should now be False even if the distance is low
                        inserted_label_dist_ds[step_idx] = 0

                    obs, rew, done, info = self.env.step(action)
                    step_idx += 1


                if failed_validation:
                    del f[demo_key]
                    continue
                else:
                    num_failed_demos = 0
                    print(f"✅ Demo {demo + 1} completed successfully with {step_idx} frames.")
                    demo += 1  # Move to the next demo

        print(f"Full Dataset saved to {self.data_path}")

    def collect_dual_view_samples(self, num_samples=10):
        save_dir = "verification_dual_view"
        os.makedirs(save_dir, exist_ok=True)

        obs = self.env.reset()
        done = False
        samples_saved = 0
        is_inserted_latch = False

        print(f"Collecting {num_samples} dual-view samples...")

        while not done and samples_saved < num_samples:
            action, _ = self.env.get_assembly_action()

            # --- State & Label Logic ---
            part_idx1, part_idx2 = self.env.furniture.should_be_assembled[self.env.assemble_idx]
            moving_part = self.env.furniture.parts[part_idx2]
            current_state = moving_part._state

            if current_state in ["insert", "insert_release"] or "screw" in current_state or "pre_grasp" in current_state:
                is_inserted_latch = True
            if current_state == "done" or self.env.furniture.all_assembled():
                is_inserted_latch = False

            label = 1 if is_inserted_latch else 0

            # --- Environment Step ---
            obs, rew, done, info = self.env.step(action)

            # --- Visual Capture Logic ---
            # Capture at intervals OR exactly when state changes
            if self.env.env_steps % 60 == 0:
                img1 = obs.get('color_image1')
                img2 = obs.get('color_image2')

                if img1 is not None and img2 is not None:
                    # Convert from Tensor to Numpy if necessary
                    if isinstance(img1, torch.Tensor):
                        img1 = img1.squeeze().cpu().numpy()
                        img2 = img2.squeeze().cpu().numpy()
                    else:
                        img1 = np.squeeze(img1)
                        img2 = np.squeeze(img2)

                    # Ensure RGB -> BGR for OpenCV
                    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

                    # Combine images side-by-side
                    # Note: They must have the same height.
                    combined_img = np.hstack((img1_bgr, img2_bgr))
                    print(combined_img.shape)

                    # Overlay status text on the combined frame
                    color = (255, 0, 0)
                    cv2.putText(combined_img, f"State: {current_state}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
                    cv2.putText(combined_img, f"Label: {label}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

                    # Add labels for which camera is which
                    cv2.putText(combined_img, "Camera 1 (Wrist)", (20, img1.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(combined_img, "Camera 2 (Side)", (img1.shape[1] + 20, img2.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    filename = f"step_{self.env.env_steps.item()}_L{label}_{current_state}.png"
                    cv2.imwrite(os.path.join(save_dir, filename), combined_img)
                    samples_saved += 1

        print(f"Saved {samples_saved} dual-view images to {save_dir}")


import multiprocessing


def collect_furniture(furniture_name):
    # This creates the class, runs the env, and then the function exits,
    # killing the local process and freeing the PhysX Foundation.
    print(f"--- Starting collection for {furniture_name} ---")
    PoseDataCollector(furniture=furniture_name)
    print(f"--- Finished {furniture_name} ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture_name", type=str, required=True)
    parser.add_argument("--num_demos", type=int, default=10)
    args = parser.parse_args()
    collector = PoseDataCollector(furniture=args.furniture_name)
    collector.collect_dataset(num_demos=args.num_demos)
