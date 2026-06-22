import genesis as gs
import time
import numpy as np
from math import radians
import torch
import os
import subprocess
import open3d as o3d  # For creating and visualizing point clouds
from scipy.spatial.transform import Rotation as R

import math
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
    'brown':  [0.396, 0.263, 0.129, 1],
    'grey':   [0.5, 0.5, 0.5, 1],
    'yellow': [1, 1, 0, 1],
    'cyan':   [0, 1, 1, 1],
    'magenta': [1, 0, 1, 1],
}

# collision bitmasks
PLANE = 0b00001  # bit 0
HOOK  = 0b00010  # bit 1
ROBOT = 0b00100  # bit 2
MAIN  = 0b01000  # bit 3
GHOST = 0b10000  # bit 4
ALL   = 0b11111


def annotate_image(filename, node_name):
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


class FRANKA:
    def __init__(self, show_viewer=True, record_video=False):

        self.EE_FRAMES = {
            'ee': 'panda_grasptarget'
        }
        self.DEFAULT_CONFIG = {
            'top': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'side': [-0.0806406098426434, -1.6722951504174777, 0.07069076842695393, -2.7449419709102822, 0.08184716251979611, 1.7516337599063168, 0.7849295270972781],
        }

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=40,
                max_FPS=200,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
            sim_options=gs.options.SimOptions(
                dt=0.01,
                # substeps=2,  # for more stable grasping contact
            ),
            rigid_options=gs.options.RigidOptions(
                # enable_self_collision=False,
                # box_box_detection=False,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                world_frame_size = 1.0,
                show_link_frame  = False,
                plane_reflection = False,
                segmentation_level = 'entity',
                ambient_light = (0.1, 0.1, 0.1),
                lights = [
                    {"type": "directional", "dir": (-2.0, 0.0, -1), "color": (1.0, 1.0, 1.0), "intensity": 6.0},
                ]
            ),
            renderer = gs.renderers.Rasterizer(), # by default
        )

        self.object_dict = {}
        self.region_dict = {"table": [[0.0, 0.7], [-0.7, 0.7], [0.07, 0.08]]}
        self.attach_dict = {}

        # Define joints indices
        self.n_dofs = 9
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)

        # Define control gains
        self.kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
        self.kd = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
        self.force = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])

        # add default entities
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(
                contype=PLANE,
                conaffinity=ALL,
            ),
        )

        self.robot = self.scene.add_entity(
            # gs.morphs.MJCF(
            #     file="/home/minseo/develop/Genesis/genesis/assets/xml/franka_emika_panda/panda.xml",
            # ),
            gs.morphs.URDF(
                file="/home/minseo/develop/Genesis/genesis/assets/urdf/panda_bullet/panda.urdf",
                fixed=True,
                merge_fixed_links=False,
            )
        )

        # TODO: eye on hand camera
        self.T = np.eye(4)
        self.T[:3, :3] = np.array([
            # [1, 0, 0],
            # [0, -1, 0],
            # [0, 0, -1]
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        self.T[:3, 3] = np.array([0.1, 0.0, 0.0])
        self.cam_arm = self.scene.add_camera(GUI=False, fov=60, res=(1280, 960))

        self.cam_left = self.scene.add_camera(
            res=(640, 480),
            pos=(0.0, 3.0, 0.8),
            lookat=(0.0, 0.0, 0.6),
            fov=40,
            GUI=False,
        )

        # self.cam_top = self.scene.add_camera(
        #     res=(640, 480),
        #     pos=(0.0, 0.05, 2.3),
        #     lookat=(0.0, 0.0, 0.0),
        #     fov=60,
        #     GUI=False,
        # )
        
        # TODO
        self.cam_front = self.scene.add_camera(
            res=(1280, 960),
            pos=(3.0, 0.0, 0.8),
            lookat=(0.0, 0.0, 0.6),
            fov=40,
            GUI=False,
        )

        self.cam_right = self.scene.add_camera(
            res=(640, 480),
            pos=(0.0, -3.0, 0.8),
            lookat=(0.0, 0.0, 0.6),
            fov=40,
            GUI=False,
        )

        self.cam_record = self.scene.add_camera(
            res=(1280, 960),
            pos=(2.0, 0.0, 1.5),
            lookat=(0.0, 0.0, 0.5),
            fov=60,
            GUI=True,
        )
        self.record_video = record_video

        self.debug_objects = []  # for find_dice domain

    def get_image(self, camera=None, segment=False):
        """
        Capture image from a Genesis camera and return as CameraImage object.
        
        Args:
            camera: Genesis camera object (default: self.cam_arm for eye-on-hand)
            segment: If True, include segmentation mask
            
        Returns:
            CameraImage object with rgb, depth, segmentation, camera_pose, and camera_matrix
        """
        from utils.utils import CameraImage, transformation_to_pose, Euler
        
        camera = camera or self.cam_arm
        
        # Render camera and get images
        rgb_arr, depth_arr, seg_arr, _ = camera.render(
            rgb=True,
            depth=True,
            segmentation=segment,
            normal=False,
            colorize_seg=False
        )
        
        # Get camera pose in world frame
        camera_transform = camera.transform  # 4x4 numpy array
        camera_pos, camera_quat = transformation_to_pose(camera_transform)
        # camera_quat is (qx, qy, qz, qw) from transformation_to_pose
        camera_pose = tuple(list(camera_pos) + list(camera_quat))
        
        # Get camera intrinsics
        # Genesis camera.intrinsics is a 3x3 matrix
        # Extract focal length and principal point
        camera_matrix = camera.intrinsics  # 3x3 numpy array
        
        # Extract segmentation if requested
        segmented = None
        if segment and seg_arr is not None:
            # seg_arr contains entity/link indices, reshape and store
            segmented = seg_arr.astype(np.uint32)
        
        return CameraImage(
            rgbPixels=rgb_arr,
            depthPixels=depth_arr,
            segmentationMaskBuffer=segmented,
            camera_pose=camera_pose,
            camera_matrix=camera_matrix,
        )

    def capture_screenshot(self, screenshot_dir, node_name):
        try:
            filename = f'{screenshot_dir}/{node_name}.jpg'
            command = ['import', '-window', 'Genesis 0.2.1', filename]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error capturing screenshot: {e}")
        except FileNotFoundError:
            print("Error: 'import' command not found. Install ImageMagick first.")

        annotate_image(filename, node_name)

        return filename

    def set_control_gains(self):
        self.robot.set_dofs_kp(self.kp)
        self.robot.set_dofs_kv(self.kd)
        self.robot.set_dofs_force_range(-self.force, self.force)

    def open_gripper(self, object_name=None, attach=True):
        self.robot.control_dofs_position(np.array([0.54, 0.54]), self.fingers_dof)
        for i in range(100):
            self.scene.step()
            if self.record_video:   
                self.render_cameras()

        if attach:
            rigid = self.scene.sim.rigid_solver
            if object_name == "hook":
                link_obj = self.scene.entities[self.object_dict[object_name]].get_link("hook_link").idx
            else:
                link_obj = self.scene.entities[self.object_dict[object_name]].get_link("box_baselink").idx
            link_franka = self.robot.get_link(self.EE_FRAMES['ee']).idx
            rigid.delete_weld_constraint(link_obj, link_franka)
            
    def close_gripper(self, object_name=None, attach=True):
        self.robot.control_dofs_force(np.array([-0.5, -0.5]), self.fingers_dof)
        for i in range(100):
            self.scene.step()
            if self.record_video:
                self.render_cameras()

        if attach:
            rigid = self.scene.sim.rigid_solver
            if object_name == "hook":
                link_obj = self.scene.entities[self.object_dict[object_name]].get_link("hook_link").idx
            else:
                link_obj = self.scene.entities[self.object_dict[object_name]].get_link("box_baselink").idx
            link_franka = self.robot.get_link(self.EE_FRAMES['ee']).idx
            rigid.add_weld_constraint(link_obj, link_franka)

    def ik(self, pose, left=True):
        end_effector = self.robot.get_link(self.EE_FRAMES['ee'])
        qpos = self.robot.inverse_kinematics(
            link=end_effector,
            pos=pose[:3],  # np array [x, y, z] (in meters)
            quat=pose[3:],  # np array [x, y, z, w] (normalized quaternion)
        )
        return qpos

    def motion_planning(self, qpos, left=True, holding=False, planner="RRTConnect", ee_link_name=None, with_entity=None):
        if holding:
            path = self.robot.plan_path_ompl(qpos, planner=planner, ignore_collision=True, ee_link_name=ee_link_name, with_entity=with_entity)
        else:
            path = self.robot.plan_path_ompl(qpos, planner=planner,)
        return path

    def move(self, path, n=None, take_screenshot=False, action_name=None, count_start=0):
        # Directory to save screenshots
        screenshot_dir = "../experiments/tool_use/screenshots"
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
            if self.record_video:
                self.render_cameras()
            if i in screenshot_indices and take_screenshot:
                self.capture_screenshot(screenshot_dir, action_name, count)
                count += 1

        # allow robot to reach the last waypoint
        for i in range(100):
            self.scene.step()
            if self.record_video:
                self.render_cameras()


    def start_recording(self, save_dir="./"):
        self.recording_save_dir = save_dir
        if hasattr(self.scene, 'visualizer') and self.scene.visualizer is not None:
            for camera in self.scene.visualizer.cameras:
                camera.start_recording()

    def stop_recording(self, video_name="video"):
        if hasattr(self.scene, 'visualizer') and self.scene.visualizer is not None:
            for camera in self.scene.visualizer.cameras:
                camera.stop_recording(
                    save_to_filename=f"{self.recording_save_dir}/{video_name}.mp4",
                    fps=100
                )

    def render_cameras(self):
        try:
            if hasattr(self.scene, 'visualizer') and self.scene.visualizer is not None:
                for camera in self.scene.visualizer.cameras:
                    camera.render()
        except Exception as e:
            # Silently ignore rendering errors to avoid breaking the main workflow
            pass

    def _warmup_snapshot_context(self, warmup_renders=2):
        # Warm up camera contexts before readback to reduce transient EGL failures.
        for _ in range(max(1, int(warmup_renders))):
            try:
                self.render_cameras()
            except Exception:
                pass

    def _save_camera_snapshot_with_retry(self, camera, save_prefix, max_retries=4):
        last_error = None
        retries = max(1, int(max_retries))

        for _ in range(retries):
            try:
                # A direct render right before save helps ensure the camera context is current.
                try:
                    camera.render()
                except Exception:
                    pass

                camera.save_snapshot(save_to_filename=save_prefix)
                out_path = f"{save_prefix}_rgb.jpg"
                if os.path.exists(out_path):
                    return out_path, None
            except Exception as e:
                last_error = e
                # Try to re-bind context for the next attempt.
                self._warmup_snapshot_context(warmup_renders=1)
                time.sleep(0.05)

        return None, last_error

    def plan_workspace_motion(self, ee_waypoints, max_attempts=20):
        ee_link = self.robot.get_link(self.EE_FRAMES['ee'])

        arm_conf = self.ik(ee_waypoints[0])
        arm_waypoints = [arm_conf]

        for ee_pose in ee_waypoints:
            arm_conf = self.ik(ee_pose)
            arm_waypoints.append(arm_conf)
        
        return arm_waypoints

    def current_ee_pose(self, draw=False):
        position = self.robot.get_link(self.EE_FRAMES['ee']).get_pos()
        orientation = self.robot.get_link(self.EE_FRAMES['ee']).get_quat()
        ee_pose = [position[0].item(), position[1].item(), position[2].item(), orientation[1].item(), orientation[2].item(), orientation[3].item(),
                    orientation[0].item()]  # x y z qx qy qz qw
        print("ee_pose", ee_pose)

        if draw:
            x, y, z, qx, qy, qz, qw = ee_pose
            T = np.eye(4, dtype=float)
            T[:3, :3] = R.from_quat(quat=[qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [x, y, z]
            mesh1 = self.scene.draw_debug_frame(T=T, axis_length=0.12, origin_size=0.02, axis_radius=0.005)

        return ee_pose

    def safe_plan(self, qpos_goal, qpos_start=None, planner="RRTConnect", ignore_collision=False, only_left=False, only_right=False, ee_link_name=None, with_entity=None):
        try:
            current_qpos = self.robot.get_qpos().detach()
            final_traj = []

            if qpos_start is not None:
                traj = self.robot.plan_path_ompl(qpos_goal=qpos_goal, qpos_start=qpos_start, planner=planner, ignore_collision=ignore_collision, ee_link_name=ee_link_name, with_entity=with_entity)
            else:
                traj = self.robot.plan_path_ompl(qpos_goal=qpos_goal, planner=planner, ignore_collision=ignore_collision, ee_link_name=ee_link_name, with_entity=with_entity)
            if traj is None or (hasattr(traj, "__len__") and len(traj) == 0):
                return False, "empty_or_none_trajectory"

            # fix the finger positions
            freeze_list = list(self.fingers_dof)
            for i in range(len(traj)):
                ti = traj[i]
                idx = torch.as_tensor(freeze_list, dtype=torch.long, device=ti.device)
                cq = current_qpos.to(device=ti.device, dtype=ti.dtype)
                ti[idx] = cq.index_select(0, idx)
                final_traj.append(ti)

            return True, traj
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def save_snapshot4(self, save_to_dir, node_name, world=None):
        os.makedirs(save_to_dir, exist_ok=True)

        # Move ghost blocks out of the way temporarily to avoid them showing up in the snapshots
        if world is not None and hasattr(world, 'ghosts'):
            for name, idx in self.object_dict.items():
                if name != "hook":
                    for ghost_idx in world.ghosts[idx]:
                        ghost = self.scene.entities[ghost_idx]
                        current_pos = ghost.get_pos()
                        ghost.set_pos((current_pos[0]+3.0, current_pos[1], current_pos[2]))

        self._warmup_snapshot_context(warmup_renders=2)

        front_path, front_err = self._save_camera_snapshot_with_retry(
            self.cam_front,
            f"{save_to_dir}/{node_name}_front",
            max_retries=4,
        )


        arm_path, arm_err = self._save_camera_snapshot_with_retry(
            self.cam_arm,
            f"{save_to_dir}/{node_name}_arm",
            max_retries=4,
        )

        left_path, left_err = self._save_camera_snapshot_with_retry(
            self.cam_left,
            f"{save_to_dir}/{node_name}_left",
            max_retries=4,
        )

        right_path, right_err = self._save_camera_snapshot_with_retry(
            self.cam_right,
            f"{save_to_dir}/{node_name}_right",
            max_retries=4,
        )

        # file_path_list = [f"{save_to_dir}/{node_name}_front_rgb.jpg", f"{save_to_dir}/{node_name}_top_rgb.jpg",
        #                   f"{save_to_dir}/{node_name}_left_rgb.jpg", f"{save_to_dir}/{node_name}_right_rgb.jpg"]
        file_path_list = [front_path, left_path, right_path]
        for file_path in file_path_list:
            annotate_image(file_path, node_name)

        if world is not None and hasattr(world, 'ghosts'):
            for name, idx in self.object_dict.items():
                if name != "hook":
                    for ghost_idx in world.ghosts[idx]:
                        ghost = self.scene.entities[ghost_idx]
                        current_pos = ghost.get_pos()
                        ghost.set_pos((current_pos[0]-3.0, current_pos[1], current_pos[2]))

        return file_path_list


def base_color(name: str) -> str:
    n = name.lower()
    return re.sub(r'\d+$', '', n)

def to_wxyz(q_xyzw):
    # PyBullet: (x, y, z, w) → Genesis: (w, x, y, z)
    return q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]


def get_pose(entity):
    pos = entity.get_pos()  # base-link pos (torch)
    quat = entity.get_quat()  # base-link quat (torch) wxyz
    # Convert tensor scalars to Python floats
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
        quat = quat.cpu().numpy()
    return [float(pos[0]), float(pos[1]), float(pos[2]), 
            float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]  # x y z qx qy qz qw


def set_pose(entity, pose):
    entity.set_pos(pose[:3])
    entity.set_quat(pose[3:])


def main():
    # Initialize
    gs.init(backend=gs.gpu, precision="32")
    franka = FRANKA(record_video=True)

    hook = franka.scene.add_entity(
        morph=gs.morphs.URDF(
            file="/home/minseo/Genesis/genesis/assets/urdf/hook/hook.urdf",
            pos=(0.1, -0.35, 0.10),
            euler=(0.0, 0.0, -30.0),
            merge_fixed_links=False,
        )
    )

    # Build the scene
    franka.scene.build()
    franka.start_recording()

    # Set control gains
    franka.set_control_gains()

    # attach mounted camera
    franka.cam_arm.attach(franka.robot.get_link(franka.EE_FRAMES['ee']), franka.T)

    franka.save_snapshot4("./", "node0", belief=None, world=None)

    # move to start pose
    arm_joints = np.array(franka.DEFAULT_CONFIG['side'])
    gripper_joints = np.array([0.54, 0.54])
    init_qpos = np.zeros(9)
    init_qpos[franka.motors_dof] = arm_joints
    init_qpos[franka.fingers_dof] = gripper_joints
    init_path = franka.motion_planning(init_qpos, planner="RRTConnect")
    franka.move(init_path)

    for i in range(1000):
        franka.scene.step()
        if franka.record_video:
            franka.render_cameras()

    # check ee frame
    init_pose = franka.current_ee_pose(draw=True)

    ee_pose_traj = [[0.10000019893050194, -0.3500019460916519, 0.10589226722711284, -3.6385247668761454e-18, 0.004566964756682248, 0.9999895713620774, 7.966969345876937e-16],
                    [0.10000019893050194, -0.35000194609165175, 0.010892267227112812, -3.6385247668761454e-18, 0.004566964756682248, 0.9999895713620774, 7.966969345876937e-16],
                    [0.10000019893050194, -0.35000194609165214, 0.2558922672271129, -3.6385247668761454e-18, 0.004566964756682248, 0.9999895713620774, 7.966969345876937e-16]]
    valid, traj = franka.safe_plan(qpos_goal=franka.ik(ee_pose_traj[0]), planner="RRTConnect", only_left=True, with_entity=hook, ee_link_name=franka.EE_FRAMES['ee'])
    franka.move(traj, take_screenshot=False)
    valid, traj = franka.safe_plan(qpos_goal=franka.ik(ee_pose_traj[1]), planner="RRTConnect", only_left=True, with_entity=hook,
                     ee_link_name=franka.EE_FRAMES['ee'])
    franka.move(traj, take_screenshot=False)
    valid, traj = franka.safe_plan(qpos_goal=franka.ik(ee_pose_traj[2]), planner="RRTConnect", only_left=True, with_entity=hook,
                     ee_link_name=franka.EE_FRAMES['ee'])
    franka.move(traj, take_screenshot=False)

    
    franka.stop_recording()


if __name__ == "__main__":
    json_path = f"/home/minseo/develop/pomdp_llm/experiments/tool_use/problem/problems_meta.json"
    # start_sim(json_path, "pomdp_llm", 3, 1, 1, 1, 12)
    main()
