import genesis as gs
import time
import numpy as np
import torch
import os
import subprocess
from utils import *
import json
import re
from pathlib import Path
# import video_recorder


PI = np.pi
COLOR_MAP = {
    'red':    [1, 0, 0, 1],
    'green':  [0, 1, 0, 1],
    'blue':   [0, 0, 1, 1],
    'white':  [1, 1, 1, 1],
    'brown':  [0.698, 0.514, 0.302, 1],
    'grey':   [0.5, 0.5, 0.5, 1],
    'yellow': [1, 1, 0, 1],
    'cyan':   [0, 1, 1, 1],
    'magenta': [1, 0, 1, 1],
}

class PR2:
    def __init__(self, vis_sim=False):

        # Special configurations
        self.TOP_HOLDING_LEFT_ARM = [0.67717021, -0.34313199, 1.2, -1.46688405, 1.24223229, -1.95442826, 2.22254125]
        self.SIDE_HOLDING_LEFT_ARM = [0.39277395, 0.33330058, 0., -1.52238431, 2.72170996, -1.21946936, -2.98914779]
        self.REST_LEFT_ARM = [2.13539289, 1.29629967, 3.74999698, -0.15000005, 10000., -0.10000004, 10000.]
        self.REST_RIGHT_ARM = [-2.13539289, 1.29629967, 0, 0, 0., 0, 0.]
        self.WIDE_LEFT_ARM = [1.5806603449288885, -0.14239066980481405, 1.4484623937179126, -1.4851759349218694,
                         1.3911839347271555,
                         -1.6531320011389408, -2.978586584568441]
        self.CENTER_LEFT_ARM = [-0.07133691252641006, -0.052973836083405494, 1.5741805775919033, -1.4481146328076862,
                           1.571782540186805, -1.4891468812835686, -9.413338322697955]
        self.STRAIGHT_LEFT_ARM = np.zeros(7)
        self.COMPACT_LEFT_ARM = [PI / 4, 0., PI / 2, -5 * PI / 8, PI / 2, -PI / 2,
                            5 * PI / 8]

        # COMPACT_LEFT_ARM = [PI/4, 0., PI/2, -5*PI/8, -PI/2, -PI/2, 3*PI/8] # More inward
        # COMPACT_LEFT_ARM = [1*PI/8, 0., PI/2, -4*PI/8, -PI/2, -PI/2, 3*PI/8-PI/2] # Most inward

        self.CLEAR_LEFT_ARM = [PI / 2, 0., PI / 2, -PI / 2, PI / 2, -PI / 2, 0.]
        # WIDE_RIGHT_ARM = [-1.3175723551150083, -0.09536552225976803, -1.396727055561703, -1.4433371993320296,
        #                   -1.5334243909312468, -1.7298129320065025, 6.230244924007009]

        self.DEFAULT_CONFIG = {
            'top': self.TOP_HOLDING_LEFT_ARM,
            'side': self.SIDE_HOLDING_LEFT_ARM,
        }
        self.EE_FRAMES = {
            'left': 'l_gripper_tool_frame',  # l_gripper_palm_link | l_gripper_tool_frame
            'right': 'r_gripper_tool_frame',  # r_gripper_palm_link | r_gripper_tool_frame
        }

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.4, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.9),
                camera_fov=40,
                max_FPS=200,
                run_in_thread=True,
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
                    {"type": "directional", "dir": (-1.5, 0, -1), "color": (1.0, 1.0, 1.0), "intensity": 6.0},
                ]
            ),
            renderer = gs.renderers.Rasterizer(), # by default
        )
        self.object_dict = {}
        self.region_dict = {"table": [[0.35, 0.70], [0.05, 0.65], [0.86, 0.87]]}

        # Define joints indices
        self.n_dofs = 30
        self.torso = np.array([0, 1])
        self.left_arm = np.array([3, 5, 7, 9, 11, 13, 15])
        self.right_arm = np.array([2, 4, 6, 8, 10, 12, 14])
        self.left_gripper = np.array([19, 20, 21, 25, 26, 27, 29])
        self.right_gripper = np.array([16, 17, 18, 22, 23, 24, 28])

        # Define control gains
        self.kp_torso = np.array([4500, 4500])
        self.kd_torso = np.array([10000, 10000])
        # self.force_torso = np.array([1000, 1000])

        self.kp_left_arm = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000])
        self.kd_left_arm = np.array([450, 450, 350, 350, 200, 200, 200])
        self.kp_right_arm = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000])
        self.kd_right_arm = np.array([450, 450, 350, 350, 200, 200, 200])
        self.kp_gripper = np.array([100] * 7)
        self.kd_gripper = np.array([10] * 7)
        self.force_arm_gripper = np.concatenate([np.tile([87, 87, 87, 87, 12, 12, 12], 2), [100] * 14])

        # add default entities (robot, plane, table, camera)
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file="assets/pr2_description/pr2_modified.urdf",
                pos=(0.0, 0.0, 0.0),
                fixed=True,
                merge_fixed_links=False,
            ),
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="assets/table/Desk_OBJ.obj",
                pos=(0.55, 0.0, 0.59),  # height 0.71
                euler=(90, 0, 90),
                scale=(0.0037, 0.0024, 0.0025),
                fixed=True,
            ),
        )
        self.object_dict["table"] = self.table.idx

        # cameras
        self.cam_front = self.scene.add_camera(
            res=(640, 480),
            pos=(2.4, 0.0, 1.5),
            lookat=(0.0, 0.0, 0.9),
            fov=40,
            GUI=False,
        )
        self.cam_top = self.scene.add_camera(
            res=(640, 480),
            pos=(0.5, 0.0, 3.0),
            lookat=(0.5, 0.0, 0.0),
            fov=40,
            GUI=False,
        )
        self.cam_left = self.scene.add_camera(
            res=(640, 480),
            pos=(0.4, -2.0, 1.5),
            lookat=(0.4, 0.0, 0.9),
            fov=40,
            GUI=False,
        )
        self.cam_right = self.scene.add_camera(
            res=(640, 480),
            pos=(0.4, 2.0, 1.5),
            lookat=(0.4, 0.0, 0.9),
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

            font_size = max(18, int(min(W, H) * 0.05))
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            text = str(node_name)

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            margin = max(16, font_size // 3)
            x = W - tw - margin
            y = margin

            pad = max(8, font_size // 4)
            draw.rectangle(
                [(x - pad, y - pad), (x + tw + pad, y + th + pad)],
                fill=(0, 0, 0, 170)
            )

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

    def capture_screenshot(self, screenshot_dir, node_name):
        try:
            filename = f'{screenshot_dir}/{node_name}.jpg'
            command = ['import', '-window', 'Genesis 0.2.1', filename]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error capturing screenshot: {e}")
        except FileNotFoundError:
            print("Error: 'import' command not found. Install ImageMagick first.")

        self.annotate_image(filename, node_name)

        return filename

    def set_control_gains(self):
        self.robot.set_dofs_kp(
            np.concatenate((self.kp_torso, self.kp_left_arm, self.kp_right_arm, self.kp_gripper, self.kp_gripper)),
            np.concatenate((self.torso, self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )
        self.robot.set_dofs_kv(
            np.concatenate((self.kd_torso, self.kd_left_arm, self.kd_right_arm, self.kd_gripper, self.kd_gripper)),
            np.concatenate((self.torso, self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )
        self.robot.set_dofs_force_range(
            -self.force_arm_gripper,
            self.force_arm_gripper,
            np.concatenate((self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )

    # gripper control fix
    def open_gripper(self, left=True, object=None):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        self.robot.control_dofs_position(np.array([0.548] * 7), gripper_dofs)

        for i in range(100):
            self.scene.step()

    def close_gripper(self, left=True, object=None):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        self.robot.control_dofs_position(np.array([0.0] * 7), gripper_dofs)

        for i in range(100):
            self.scene.step()

    def ik(self, pose, left=True):
        end_effector = self.robot.get_link(self.EE_FRAMES['left']) if left else self.robot.get_link(
            self.EE_FRAMES['right'])
        qpos = self.robot.inverse_kinematics(
            link=end_effector,
            pos=pose[:3],  # np array [x, y, z] (in meters)
            quat=pose[3:],  # np array [x,y,z,w] (normalized quaternion)
            dofs_idx_local=self.left_arm
        )
        return qpos

    def motion_planning(self, qpos, left=True, holding=False, planner="RRTConnect", ee_link_name=None, with_entity=None):
        arm_dofs = self.left_arm if left else self.right_arm

        if holding:
            path = self.robot.plan_path_ompl(qpos, planner=planner, ignore_collision=True, ee_link_name=ee_link_name, with_entity=with_entity)
        else:
            path = self.robot.plan_path_ompl(qpos, planner=planner, ee_link_name=ee_link_name, with_entity=with_entity)

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

    # TODO: ee link idx
    def current_ee_pose(self, left=True):
        if left:
            position = self.robot.get_link(self.EE_FRAMES['left']).get_pos()
            orientation = self.robot.get_link(self.EE_FRAMES['left']).get_quat()
        else:
            position = self.robot.get_link(self.EE_FRAMES['right']).get_pos()
            orientation = self.robot.get_link(self.EE_FRAMES['right']).get_quat()
        ee_pose = [position[0].item(), position[1].item(), position[2].item(), orientation[1].item(), orientation[2].item(), orientation[3].item(),
                    orientation[0].item()]  # x y z qx qy qz qw
        print("ee_pose", ee_pose)
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

    def safe_plan(self, qpos_goal, qpos_start=None, planner="RRTConnect", ignore_collision=False, only_left=False, only_right=False, ee_link_name=None, with_entity=None):
        try:
            current_qpos = self.robot.get_qpos().detach()
            right_arm_fix = torch.tensor(
                self.REST_RIGHT_ARM,
                dtype=current_qpos.dtype,
                device=current_qpos.device
            )
            torso_fix = torch.tensor(
                [0.5, 0.0],
                dtype=current_qpos.dtype,
                device=current_qpos.device
            )
            final_traj = []

            if qpos_start is not None:
                traj = self.robot.plan_path_ompl(qpos_goal=qpos_goal, qpos_start=qpos_start,
                                       planner=planner, ignore_collision=ignore_collision)
            else:
                traj = self.robot.plan_path_ompl(qpos_goal=qpos_goal,
                                       planner=planner, ignore_collision=ignore_collision)
            if traj is None or (hasattr(traj, "__len__") and len(traj) == 0):
                return False, "empty_or_none_trajectory"

            if only_left:
                freeze_list = list(self.torso) + list(self.left_gripper) + list(self.right_arm) + list(self.right_gripper)
                for i in range(len(traj)):
                    ti = traj[i]
                    idx = torch.as_tensor(freeze_list, dtype=torch.long, device=ti.device)
                    cq = current_qpos.to(device=ti.device, dtype=ti.dtype)
                    ti[idx] = cq.index_select(0, idx)
                    ti[self.right_arm] = right_arm_fix
                    ti[self.torso] = torso_fix
                    final_traj.append(ti)
            elif only_right:
                freeze_list = list(self.torso) + list(self.left_gripper) + list(self.left_arm) + list(self.right_gripper)
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

    def save_snapshot4(self, save_to_dir, node_name, world=None):
        self.cam_front.save_snapshot(save_to_filename=f"{save_to_dir}/{node_name}_front")
        self.cam_top.save_snapshot(save_to_filename=f"{save_to_dir}/{node_name}_top")
        self.cam_left.save_snapshot(save_to_filename=f"{save_to_dir}/{node_name}_left")
        self.cam_right.save_snapshot(save_to_filename=f"{save_to_dir}/{node_name}_right")

        file_path_list = [f"{save_to_dir}/{node_name}_front_rgb.jpg", f"{save_to_dir}/{node_name}_top_rgb.jpg",
                          f"{save_to_dir}/{node_name}_left_rgb.jpg", f"{save_to_dir}/{node_name}_right_rgb.jpg"]
        for file_path in file_path_list:
            self.annotate_image(file_path, node_name)

        return file_path_list


def base_color(name: str) -> str:
    n = name.lower()
    return re.sub(r'\d+$', '', n)

def to_wxyz(q_xyzw):
    # PyBullet: (x, y, z, w) → Genesis: (w, x, y, z)
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])

def start_sim(json_path, method, prob_num, prob_idx, trial, repeat, vis_sim=False):
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
    pr2 = PR2(vis_sim=vis_sim)

    # add object
    pr2.object_dict = getattr(pr2, "object_dict", {})
    for i, (name, info) in enumerate(blocks_info.items()):
        size = tuple(info["size"])  # (sx, sy, sz)
        pos0 = tuple([0.40, 0.80 + 0.10 * i, info["size"][2] / 2 + 0.01])
        col = COLOR_MAP.get(base_color(name), COLOR_MAP["grey"])

        ent = pr2.scene.add_entity(
            morph=gs.morphs.Box(
                size=size,
                pos=pos0,
            ),
            surface=gs.surfaces.Rough(
                color=col,
            ),
        )
        pr2.object_dict[name] = ent.idx

    # Build the scene
    pr2.scene.build()

    pr2.cam_front.start_recording()

    # Set control gains
    pr2.set_control_gains()

    link = pr2.robot.get_link(pr2.EE_FRAMES['left'])

    # move to start pose
    left_arm_joints = np.array(pr2.DEFAULT_CONFIG['top'])
    right_arm_joints = np.array(pr2.REST_RIGHT_ARM)
    gripper_joints = np.array([0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.8])

    init_qpos = np.zeros(30)
    init_qpos[pr2.torso] = np.array([2.0, 0.0])
    init_path = pr2.motion_planning(init_qpos, planner="RRTConnect")
    pr2.move(init_path)

    init_qpos[pr2.left_arm] = left_arm_joints
    init_qpos[pr2.left_gripper] = gripper_joints
    init_path = pr2.motion_planning(init_qpos, planner="RRTConnect")
    pr2.move(init_path)

    init_qpos[pr2.right_arm] = right_arm_joints
    init_path = pr2.motion_planning(init_qpos, planner="RRTConnect")
    pr2.move(init_path)

    init_qpos[pr2.torso] = np.array([0.5, 0.0])
    init_path = pr2.motion_planning(init_qpos, planner="RRTConnect")
    pr2.move(init_path)

    # to fix the torso & right arm
    pr2.kd_torso = np.array([10000000, 10000000])
    pr2.kd_arm =  np.array([10000] * 7)
    pr2.set_control_gains()

    # set pose of the blocks
    for name, info in blocks_info.items():
        idx = pr2.object_dict[name]
        ent = pr2.scene.entities[idx]
        pos = np.array(info["pose"]["position"], dtype=float)
        q_xyzw = info["pose"]["quaternion"]
        q_wxyz = np.array(to_wxyz(q_xyzw), dtype=float)

        ent.set_pos(pos)
        ent.set_quat(q_wxyz)

    for i in range(10):
        pr2.scene.step()

    screenshot_dir = Path(f"../experiments/blocksworld_pr/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = f"../experiments/blocksworld_pr/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}"
    # file_path = pr2.capture_screenshot(screenshot_dir=str(screenshot_dir), node_name="node0")
    file_path_list = pr2.save_snapshot4(screenshot_dir, node_name="node0")

    return pr2, file_path_list
