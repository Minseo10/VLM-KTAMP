import genesis as gs
import time
import numpy as np
from math import radians
import torch
import os
import subprocess
import open3d as o3d  # For creating and visualizing point clouds
import json
import re
from pathlib import Path


PI = np.pi
COLOR_MAP = {
    'red':    [1, 0, 0, 1],
    'green':  [0, 1, 0, 1],
    'blue':   [0, 0, 1, 1],
    'white':  [1, 1, 1, 1],
    'brown':  [0.396, 0.263, 0.129, 1],
    'grey':   [0.5, 0.5, 0.5, 1],
    'yellow': [1, 1, 0, 1],
    'cyan':   [0, 1, 1, 1],
    'magenta': [1, 0, 1, 1],
    'apricot': [0.698, 0.514, 0.302]
}

class DualArm:
    def __init__(self, vis_sim=False):

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.4, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.9),
                camera_fov=40,
                max_FPS=200,
            ),
            show_viewer=vis_sim,
            show_FPS = False,
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=2,  # for more stable grasping contact
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = False, # visualize the coordinate frame of `world` at its origin
                world_frame_size = 1.0, # length of the world frame in meter
                show_link_frame  = False, # do not visualize coordinate frames of entity links
                plane_reflection = True, # turn on plane reflection
                segmentation_level = 'entity',
                ambient_light = (0.1, 0.1, 0.1),
                lights = [
                    {"type": "directional", "dir": (1.5, 0, -1), "color": (1.0, 1.0, 1.0), "intensity": 6.0},
                ]
            ),
            renderer = gs.renderers.Rasterizer(), # by default
            # renderer=gs.renderers.RayTracer(  # type: ignore
            #     env_surface=gs.surfaces.Emission(
            #         emissive_texture=gs.textures.ImageTexture(
            #             image_path="textures/indoor_bright.png",
            #         ),
            #     ),
            #     env_radius=15.0,
            #     env_euler=(0, 0, 180),
            #     lights=[
            #         {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (10.0, 10.0, 10.0)},
            #     ],
            # ),
        )
        self.EE_FRAMES = {
            'left': 'left_gripper_tool0',  # l_gripper_palm_link | l_gripper_tool_frame
            'right': 'right_gripper_tool0',  # r_gripper_palm_link | r_gripper_tool_frame
        }

        self.object_dict = {}
        # self.region_dict = {"r1": [[-0.30, -0.525], [-0.1875, -0.4125], [0.81, 0.82]], "r2": [[-0.30, -0.525], [-0.4125, -0.6375], [0.81, 0.82]], "r3": [[-0.30, -0.525], [-0.6375, -0.78], [0.81, 0.82]], "r4": [[-0.57, -0.80], [-0.1875, -0.4125], [0.81, 0.82]], "r5": [[-0.57, -0.80], [-0.4125, -0.6375], [0.81, 0.82]], "r6": [[-0.57, -0.80], [-0.6375, -0.78], [0.81, 0.82]]}
        self.region_dict = {"table": [[-0.40, -0.77], [-0.10, -0.80], [0.86, 0.87]]}

        # Define joints indices
        self.left_arm = np.array([0, 2, 4, 6, 8, 10])
        self.right_arm = np.array([1, 3, 5, 7, 9, 11])
        self.left_fingers = np.array([16, 22, 28, 17, 23, 29, 18, 24, 30])
        self.right_fingers = np.array([19, 25, 31, 20, 26, 32, 21, 27, 33])
        self.left_palm = np.array([12, 13])
        self.right_palm = np.array([14, 15])

        self.left_gripper = np.concatenate((
            self.left_fingers, self.left_palm
        ))
        self.right_gripper = np.concatenate((
            self.right_fingers, self.right_palm
        ))

        # Define control gains
        # TODO: optimal control gains
        self.kp_arm = np.array([4500, 4500, 3500, 2000, 2000, 2000])
        # self.kd_arm = 2 * 7 * np.sqrt(self.kp_arm)
        self.kd_arm = np.array([450, 450, 350, 200, 200, 200])
        self.kp_gripper = np.array([100] * 11)
        # self.kd_gripper = 2 * 7 * np.sqrt(self.kp_gripper)
        self.kd_gripper = np.array([10] * 11)
        self.force_arm = np.concatenate([np.tile([87, 87, 87, 12, 12, 12], 2), [100] * 22])

        # add default entities (robot, plane, table, camera)
        # using ray tracer
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # using rasterizer
        # self.plane = self.scene.add_entity(
        #     gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        # )

        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file="/home/user/robot_ws/src/dual_ur_robotiq/dual_ur_robotiq_description/urdf/dual_ur_robotiq_genesis.urdf",
                fixed=True,
                merge_fixed_links=False,
            ),
            # vis_mode="collision",
            # visualize_contact=True,
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="/home/user/Genesis/genesis/assets/meshes/table/Desk_OBJ.obj",
                pos=(-0.61, 0.0, 0.58),  # height 0.7
                euler=(90, 0, 90),
                scale=(0.0045, 0.0029, 0.0025),
                fixed=True,
            ),
            # vis_mode = "collision",
        )
        self.object_dict["table"] = self.table

        self.left_wall = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(1.55, 0.1, 2.0),
                pos=(-0.125, -2, 1.0),
                fixed=True,
                visualization=False
            ),
        )

        self.right_wall = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(1.55, 0.1, 2.0),
                pos=(-0.125, 2, 1.0),
                fixed=True,
                visualization=False
            ),
        )

        self.back_wall = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.1, 3.4, 2.0),
                pos=(1.5, 0.0, 1.0),
                fixed=True,
                visualization=False
            ),
        )

        self.cam = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.08, 0.13, 0.63),
                pos=(-0.9, -0.33, 1.025),
                fixed=True,
                visualization=False
            ),
        )


        # cameras
        self.cam_front = self.scene.add_camera(
            res=(640, 480),
            pos=(-2.4, 0.0, 1.5),
            lookat=(0.0, 0.0, 0.9),
            fov=40,
            GUI=False,
        )
        self.cam_top = self.scene.add_camera(
            res=(640, 480),
            pos=(-0.8, 0.0, 3.0),
            lookat=(-0.5, 0.0, 0.0),
            fov=40,
            GUI=False,
        )
        self.cam_left = self.scene.add_camera(
            res=(640, 480),
            pos=(-0.5, -2.0, 1.5),
            lookat=(-0.5, 0.0, 0.9),
            fov=40,
            GUI=False,
        )
        self.cam_right = self.scene.add_camera(
            res=(640, 480),
            pos=(-0.5, 2.0, 1.5),
            lookat=(-0.5, 0.0, 0.9),
            fov=40,
            GUI=False,
        )

    def annotate_image(self, filename, node_name):

        try:
            from PIL import Image, ImageDraw, ImageFont

            im = Image.open(filename).convert("RGBA")
            W, H = im.size
            overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # 자동 폰트 크기(이미지 짧은 변의 5%)
            font_size = max(18, int(min(W, H) * 0.05))
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            text = str(node_name)

            # 텍스트 크기 측정
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            margin = max(16, font_size // 3)
            x = W - tw - margin
            y = margin

            # 반투명 검정 배경 박스
            pad = max(8, font_size // 4)
            draw.rectangle(
                [(x - pad, y - pad), (x + tw + pad, y + th + pad)],
                fill=(0, 0, 0, 170)  # alpha ↑ 더 진하게 보이게
            )

            # 굵은 외곽선(블랙) + 본문(화이트)
            outline = max(2, font_size // 10)
            for dx, dy in [(-outline, 0), (outline, 0), (0, -outline), (0, outline),
                           (-outline, -outline), (outline, -outline), (-outline, outline), (outline, outline)]:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

            out = Image.alpha_composite(im, overlay).convert("RGB")
            out.save(filename, "JPEG")
        except Exception as e:
            print(f"Warning: Pillow annotate failed: {e}")
        return


    def capture_screenshot(self, screenshot_dir, problem_name, node_name):
        try:
            filename = f'{screenshot_dir}/{problem_name}/{node_name}.jpg'
            command = ['import', '-window', 'Genesis 0.2.1', filename]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error capturing screenshot: {e}")
        except FileNotFoundError:
            print("Error: 'import' command not found. Install ImageMagick first.")

        return filename

    def initialize_scene(self, problem):
        pass

    def set_control_gains(self):
        self.robot.set_dofs_kp(
            np.concatenate((self.kp_arm, self.kp_arm, self.kp_gripper, self.kp_gripper)),
            np.concatenate((self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )
        self.robot.set_dofs_kv(
            np.concatenate((self.kd_arm, self.kd_arm, self.kd_gripper, self.kd_gripper)),
            np.concatenate((self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )
        self.robot.set_dofs_force_range(
            -self.force_arm,
            self.force_arm,
            np.concatenate((self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )

    def open_gripper(self, left=True, object=None):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        self.robot.control_dofs_position(np.array([0.0] * 9 + [-0.15708, 0.15708]), gripper_dofs)

        for i in range(100):
            self.scene.step()

    # TODO: close gripper with pinch mode
    def close_gripper(self, left=True, object=None):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        # robot.control_dofs_position(np.array([0.55, 0.0, -0.5236, 0.55, 0.0, -0.5323, 0.55, 0.0, -0.5323, -0.15708, 0.15708]), gripper_dofs)
        self.robot.control_dofs_position(np.array([0.78, 0.0, -0.5, 0.78, 0.0, -0.5, 0.78, 0.0, -0.5, -0.15708, 0.15708]), gripper_dofs)
        # robot.control_dofs_force(np.array([2, 2, 2] * 3 + [-2, 2]), gripper_dofs)

        for i in range(100):
            self.scene.step()

    def control_gripper_contact(self, left=True, desired_force=10.0, pinch=True):
        finger_links = [32, 40, 46, 33, 41, 47, 34, 42, 48] if left else [36, 43, 49, 37, 44, 50, 38, 45, 51]
        finger_dofs = self.left_fingers if left else self.right_fingers
        dt = np.zeros(9)

        if pinch:
            while True:
                contact_force = self.robot.get_links_net_contact_force()[finger_links].cpu().numpy()
                print("net contact: ", contact_force)
                forces_size = np.linalg.norm(contact_force, axis=1)
                print('forces_size', forces_size)
                # Finger forces split
                finger_1_force = forces_size[:3]
                finger_2_force = forces_size[3:6]
                finger_middle_force = forces_size[6:9]

                # Helper function to decide control direction
                def update_dt(forces, dt_indices):
                    if np.all(forces < desired_force):
                        dt[dt_indices[0]] = 0.05  # Close
                        dt[dt_indices[1]] = 0.0
                        dt[dt_indices[2]] = -0.05  # Open
                    elif forces[0] >= desired_force and np.all(forces[1:] < desired_force):
                        dt[dt_indices[0]] = 0.0
                        dt[dt_indices[1]] = 0.05
                        dt[dt_indices[2]] = 0.0
                    elif forces[1] >= desired_force > forces[2]:
                        dt[dt_indices[0]] = 0.0
                        dt[dt_indices[1]] = 0.0
                        dt[dt_indices[2]] = 0.05
                    elif forces[2] >= desired_force:
                        dt[dt_indices[0]] = 0.0
                        dt[dt_indices[1]] = 0.0
                        dt[dt_indices[2]] = 0.05   # ???
                        return True  # Force threshold met
                    return False

                # Update control for each finger
                stop_finger_1 = update_dt(finger_1_force, [0, 1, 2])
                stop_finger_2 = update_dt(finger_2_force, [3, 4, 5])
                stop_finger_middle = update_dt(finger_middle_force, [6, 7, 8])

                # Break the loop if all fingers have sufficient contact force
                if stop_finger_1 and stop_finger_2 and stop_finger_middle:
                    break

                # Control joints
                qpos = self.robot.get_dofs_position(finger_dofs).cpu().numpy()
                new_qpos = qpos + dt
                self.robot.control_dofs_position(new_qpos, finger_dofs)

                # Step the simulation
                self.scene.step()

    def pinch_gripper(self, left=True):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        self.robot.control_dofs_position(np.array([0.05, 0.0, -0.05, 0.05, 0.0, -0.05, 0.05, 0.0, -0.05, -0.15708, 0.15708]), gripper_dofs)

        for i in range(100):
            self.scene.step()

    def ik(self, pose, left=True):
        arm_dofs = self.left_arm if left else self.right_arm
        end_effector = self.robot.get_link(self.EE_FRAMES['left']) if left else self.robot.get_link(self.EE_FRAMES['right'])
        qpos = self.robot.inverse_kinematics(
            link=end_effector,
            pos=pose[:3],  # np array [x, y, z] (in meters)
            quat=pose[3:],  # np array [x,y,z,w] (normalized quaternion)
        )
        return qpos

    def motion_planning(self, qpos, left=True, holding=False, planner="RRTConnect", held_entity=None):
        arm_dofs = self.left_arm if left else self.right_arm

        if holding:
            path = self.robot.plan_path(qpos, planner=planner, ignore_collision=True, held_entity=held_entity)
        else:
            path = self.robot.plan_path(qpos, planner=planner, held_entity=held_entity)

        return path

    def move(self, path, n=None, take_screenshot=False, action_name=None, count_start=0):
        # Directory to save screenshots
        screenshot_dir = "../experiments/blocksworld/screenshots"
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        count = count_start
        if take_screenshot and len(path) > 1:
            screenshot_indices = set(
                round(i * (len(path) - 1) / (n - 1))
                for i in range(n)
            )
        else:
            screenshot_indices = set()

        for i, waypoint in enumerate(path):
            self.robot.control_dofs_position(waypoint)
            self.scene.step()

            if i in screenshot_indices and take_screenshot:
                self.capture_screenshot(screenshot_dir, action_name, count)
                count += 1

        # allow robot to reach the last waypoint
        for i in range(100):
            self.scene.step()

    def current_ee_pose(self, left=True):
        if left:
            position = self.robot.get_links_pos()[26].tolist()
            orientation = self.robot.get_links_quat()[26].tolist()
        else:
            position = self.robot.get_links_pos()[30].tolist()
            orientation = self.robot.get_links_quat()[30].tolist()
        ee_pose = [position[0], position[1], position[2], orientation[1], orientation[2], orientation[3],
                   orientation[0]]  # x y z qx qy qz qw
        # print("ee_pose", ee_pose)
        return ee_pose

    def generate_point_cloud(self):
        """
        Generate a 3D point cloud from rgb and depth images.
        """

        rgb, depth, _, _ = self.cam_0.render(rgb=True, depth=True, segmentation=False, normal=False)

        h, w = depth.shape
        fx, fy, cx, cy = self.cam_0.intrinsics[0][0], self.cam_0.intrinsics[1][1], self.cam_0.intrinsics[0][2], self.cam_0.intrinsics[1][2]

        # Create a grid of pixel coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x.flatten()
        y = y.flatten()

        # Project depth into 3D space
        z = depth.flatten()
        x_3d = (x - cx) * z / fx
        y_3d = (y - cy) * z / fy

        # Mask out invalid points
        valid_mask = z > 0
        x_3d = x_3d[valid_mask]
        y_3d = y_3d[valid_mask]
        z_3d = z[valid_mask]
        rgb = rgb.reshape(-1, 3)[valid_mask]

        # Combine XYZ and RGB into a point cloud
        points = np.column_stack((x_3d, y_3d, z_3d, rgb))
        return points

    def save_pcd_file(self, points, save_dir="./pointclouds"):
        """
        Save a point cloud to a .pcd file using a custom ASCII format.

        Parameters:
            points: np.ndarray of shape (N, 6) where each row is [x, y, z, r, g, b].
                    r, g, b are assumed to be in the range [0, 255].
            filename: Name of the output .pcd file.
            save_dir: Directory to save the file.

        Returns:
            pcd_file_path: The full path to the saved .pcd file.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        pcd_file_path = os.path.join(save_dir, f"output_{ts}.pcd")

        # Number of points
        N = points.shape[0]

        # Build PCD header string (PCD v0.7 format, ASCII)
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F U\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {N}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {N}\n"
            "DATA ascii\n"
        )

        # Open the file and write header and point data
        with open(pcd_file_path, "w") as f:
            f.write(header)
            for row in points:
                x, y, z, r, g, b = row
                # Convert r, g, b to integers and pack them into a single unsigned int.
                # The typical encoding: (r << 16) | (g << 8) | b.
                rgb_int = (int(round(r)) << 16) | (int(round(g)) << 8) | int(round(b))
                f.write(f"{x:.7f} {y:.7f} {z:.7f} {rgb_int}\n")

        # print(f"Saved point cloud to: {pcd_file_path}")
        return pcd_file_path


    def _safe_plan(self, qpos_goal, qpos_start=None, planner="RRTConnect", ignore_collision=False, only_left=False, only_right=False, held_entity=None):
        try:
            current_qpos = self.robot.get_qpos().detach()
            right_arm_fix = torch.tensor(
                [radians(-138), radians(-108), radians(-96), radians(-34), radians(-56), radians(-87)],
                dtype=current_qpos.dtype,
                device=current_qpos.device
            )

            final_traj = []

            if qpos_start is not None:
                traj = self.robot.plan_path(qpos_goal=qpos_goal, qpos_start=qpos_start,
                                       planner=planner, ignore_collision=ignore_collision)  # TODO: disabled held_entity for pr2 (just in case)
            else:
                traj = self.robot.plan_path(qpos_goal=qpos_goal,
                                       planner=planner, ignore_collision=ignore_collision)
            if traj is None or (hasattr(traj, "__len__") and len(traj) == 0):
                return False, "empty_or_none_trajectory"

            if only_left:
                freeze_list = list(self.left_gripper) + list(self.right_gripper)
                for i in range(len(traj)):
                    ti = traj[i]
                    idx = torch.as_tensor(freeze_list, dtype=torch.long, device=ti.device)
                    cq = current_qpos.to(device=ti.device, dtype=ti.dtype)
                    ti[idx] = cq.index_select(0, idx)
                    ti[self.right_arm] = right_arm_fix
                    final_traj.append(ti)
            elif only_right:
                freeze_list = list(self.left_gripper) + list(self.left_arm) + list(self.right_gripper)
                for i in range(len(traj)):
                    ti = traj[i]
                    idx = torch.as_tensor(freeze_list, dtype=torch.long, device=ti.device)
                    cq = current_qpos.to(device=ti.device, dtype=ti.dtype)
                    ti[idx] = cq.index_select(0, idx)
                    final_traj.append(ti)
            else:
                final_traj = traj
            return True, final_traj
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def save_snapshot4(self, save_to_dir, node_name):
        self.cam_front.save_snapshot(save_to_filename=f"{save_to_dir}/{node_name}_front")
        self.cam_top.save_snapshot(save_to_filename=f"{save_to_dir}/{node_name}_top")
        self.cam_left.save_snapshot(save_to_filename=f"{save_to_dir}/{node_name}_left")
        self.cam_right.save_snapshot(save_to_filename=f"{save_to_dir}/{node_name}_right")

        file_path_list = [f"{save_to_dir}/{node_name}_front_rgb.png", f"{save_to_dir}/{node_name}_top_rgb.png",
                          f"{save_to_dir}/{node_name}_left_rgb.png", f"{save_to_dir}/{node_name}_right_rgb.png"]
        for file_path in file_path_list:
            self.annotate_image(file_path, node_name)

        return file_path_list

def base_color(name: str) -> str:
    n = name.lower()
    return re.sub(r'\d+$', '', n)

def to_wxyz(q_xyzw):
    # PyBullet: (x, y, z, w) → Genesis: (w, x, y, z)
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])


def start_sim(json_path, method, prob_num, prob_idx, trial, repeat, num_distractor=0, vis_sim=False):
    # load json file and bring entry
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for e in meta:
        if e.get("num") == prob_num and e.get("index") == prob_idx and e.get("trial") == trial:
            entry = e
            break
    blocks_info = entry["objects"]

    # Initialize
    gs.init(backend=gs.gpu)

    # Initialize DualArm
    dual_arm = DualArm(vis_sim=vis_sim)

    # add object
    dual_arm.object_dict = getattr(dual_arm, "object_dict", {})
    for i, (name, info) in enumerate(blocks_info.items()):
        size = tuple(info["size"])  # (sx, sy, sz)
        pos0 = tuple([0.40, 0.80 + 0.10 * i, info["size"][2] / 2 + 0.01])
        col = COLOR_MAP.get(base_color(name), COLOR_MAP["grey"])

        ent = dual_arm.scene.add_entity(
            morph=gs.morphs.Box(
                size=size,
                pos=pos0,
            ),
            surface=gs.surfaces.Rough(
                color=col,
            ),
        )
        dual_arm.object_dict[name] = ent

    # Build scene
    dual_arm.scene.build()

    dual_arm.cam_front.start_recording()

    # Set control gains
    dual_arm.set_control_gains()

    # Move to start pose
    # TODO: same in real-world?
    left_arm_joints = np.array([radians(138), radians(-72), radians(96), radians(-146), radians(56), radians(3)])
    right_arm_joints = np.array([radians(-138), radians(-108), radians(-96), radians(-34), radians(-56), radians(-87)])
    init_qpos = np.zeros(34)
    init_qpos[dual_arm.left_arm] = left_arm_joints
    init_qpos[dual_arm.right_arm] = right_arm_joints
    init_path = dual_arm.motion_planning(init_qpos, planner="RRTConnect")
    dual_arm.move(init_path)

    # Pinch mode
    dual_arm.pinch_gripper(True)

    # set pose of the blocks
    for name, info in blocks_info.items():
        ent = dual_arm.object_dict[name]
        pos = np.array(info["pose"]["position"], dtype=float)
        q_xyzw = info["pose"]["quaternion"]
        q_wxyz = np.array(to_wxyz(q_xyzw), dtype=float)

        ent.set_pos(pos)
        ent.set_quat(q_wxyz)

    dual_arm.left_wall.set_pos(np.array([-0.125, -1.65, 1.0]))
    dual_arm.right_wall.set_pos(np.array([-0.125, 1.6, 1.0]))
    dual_arm.back_wall.set_pos(np.array([0.7, 0.0, 1.0]))

    for i in range(10):
        dual_arm.scene.step()

    screenshot_dir = Path(f"../experiments/blocksworld_pr/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = f"../experiments/blocksworld_pr/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}"
    # file_path = pr2.capture_screenshot(screenshot_dir=str(screenshot_dir), node_name="node0")
    file_path_list = dual_arm.save_snapshot4(screenshot_dir, node_name="node0")
    return dual_arm, file_path_list

