import genesis as gs
import numpy as np
import os
import subprocess
from utils import *
import json
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

class KUKA:
    def __init__(self):

        # Special configurations

        self.EE_FRAMES = {
            'ee': 'iiwa_link_ee_kuka'
        }
        self.DEFAULT_CONFIG = {
            'top': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'side': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, 2.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=200,
                run_in_thread=True,
            ),
            show_viewer=True,
            show_FPS = False,
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=2,  # for more stable grasping contact
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # box_box_detection=True,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = False, # visualize the coordinate frame of `world` at its origin
                world_frame_size = 1.0, # length of the world frame in meter
                show_link_frame  = False, # do not visualize coordinate frames of entity links
                plane_reflection = True, # turn on plane reflection
                segmentation_level = 'entity',
                ambient_light = (0.1, 0.1, 0.1),
                lights = [
                    {"type": "directional", "dir": (0, -2.0, -1), "color": (1.0, 1.0, 1.0), "intensity": 6.0},
                ]
            ),
            renderer = gs.renderers.Rasterizer(), # by default
        )
        self.object_dict = {}
        self.region_dict = {"table": [[-0.7, 0.7], [-0.7, 0.7], [0.22, 0.23]]}  # TODO: z area
        self.attach_dict = {}

        # Define joints indices
        self.n_dofs = 7

        # Define control gains
        self.kp = np.array([3500, 3500, 3000, 3000, 1500, 1500, 1500])
        self.kd = np.array([350, 350, 300, 300, 150, 150, 150])
        self.force = np.array([87, 87, 87, 87, 12, 12, 12])

        # add default entities
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file="assets/kuka_iiwa/model_free_base.urdf",
                pos=(0.0, 0.0, 0.0),
                fixed=True,
                merge_fixed_links=False,
            ),
        )

        self.sink = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.25, 0.25, 0.05),
                pos=(-0.5, 0.0, 0.025),
                collision=True,
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                color=(0.6, 0.1, 0.1),
            ),
        )
        self.object_dict["mysink"] = self.sink

        self.stove = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.25, 0.25, 0.05),
                pos=(0.5, 0.0, 0.025),
                collision=True,
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                color=(0.1, 0.1, 0.6),
            ),
        )
        self.object_dict["mystove"] = self.stove

        # cameras
        self.cam_front = self.scene.add_camera(
            res=(640, 480),
            pos=(0.0, 2.2, 1.0),
            lookat=(0.0, 0.0, 0.5),
            fov=40,
            GUI=False,
        )
        self.cam_top = self.scene.add_camera(
            res=(640, 480),
            pos=(0.0, 0.05, 2.3),
            lookat=(0.0, 0.0, 0.0),
            fov=40,
            GUI=False,
        )
        self.cam_left = self.scene.add_camera(
            res=(640, 480),
            pos=(1.0, 0.0, 1.5),
            lookat=(0.5, 0.0, 0.3),
            fov=40,
            GUI=False,
        )
        self.cam_right = self.scene.add_camera(
            res=(640, 480),
            pos=(-1.0, 0.0, 1.5),
            lookat=(-0.5, 0.0, 0.3),
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
        self.robot.set_dofs_kp(self.kp)
        self.robot.set_dofs_kv(self.kd)
        self.robot.set_dofs_force_range(-self.force, self.force)

    # detach
    def open_gripper(self, left=True, object=None):
        rigid = self.scene.sim.rigid_solver
        link_obj = np.array([object.get_link("box_baselink").idx], dtype=gs.np_int)
        link_kuka = np.array([self.robot.get_link(self.EE_FRAMES['ee']).idx], dtype=gs.np_int)
        rigid.delete_weld_constraint(link_obj, link_kuka)

    # attach
    def close_gripper(self, left=True, object=None):
        rigid = self.scene.sim.rigid_solver
        link_obj = np.array([object.get_link("box_baselink").idx], dtype=gs.np_int)
        link_kuka = np.array([self.robot.get_link(self.EE_FRAMES['ee']).idx], dtype=gs.np_int)
        rigid.add_weld_constraint(link_obj, link_kuka)

    def detach_constraint(self, object1, object2):
        rigid = self.scene.sim.rigid_solver
        link_obj1 = np.array([object1.get_link("box_baselink").idx], dtype=gs.np_int)
        link_obj2 = np.array([object2.get_link("box_baselink").idx], dtype=gs.np_int)
        rigid.delete_weld_constraint(link_obj1, link_obj2)

    def attach_constraint(self, object1, object2):
        rigid = self.scene.sim.rigid_solver
        link_obj1 = np.array([object1.get_link("box_baselink").idx], dtype=gs.np_int)
        link_obj2 = np.array([object2.get_link("box_baselink").idx], dtype=gs.np_int)
        rigid.add_weld_constraint(link_obj1, link_obj2)

    def ik(self, pose, left=True):
        end_effector = self.robot.get_link(self.EE_FRAMES['ee'])
        qpos = self.robot.inverse_kinematics(
            link=end_effector,
            pos=pose[:3],  # np array [x, y, z] (in meters)
            quat=pose[3:],  # np array [x,y,z,w] (normalized quaternion)
            # dofs_idx_local=self.left_arm
        )
        return qpos

    def motion_planning(self, qpos, left=True, holding=False, planner="RRTConnect", held_entity=None):
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
            screenshot_indices = set()  # 스크린샷 찍지 않을 때

        # take_screenshot = True 이면 n등분 해서 사진 찍기
        for i, waypoint in enumerate(path):
            self.robot.control_dofs_position(waypoint)
            self.scene.step()
            # n등분 지점이라면 스크린샷 촬영
            if i in screenshot_indices and take_screenshot:
                self.capture_screenshot(screenshot_dir, action_name, count)
                count += 1

        # allow robot to reach the last waypoint
        for i in range(100):
            self.scene.step()

    def current_ee_pose(self, left=True):
        position = self.robot.get_link(self.EE_FRAMES['ee']).get_pos()
        orientation = self.robot.get_link(self.EE_FRAMES['ee']).get_quat()
        ee_pose = [position[0].item(), position[1].item(), position[2].item(), orientation[1].item(), orientation[2].item(), orientation[3].item(),
                    orientation[0].item()]  # x y z qx qy qz qw
        print("ee_pose", ee_pose)
        return ee_pose

    def safe_plan(self, qpos_goal, qpos_start=None, planner="RRTConnect", ignore_collision=False, only_left=False, only_right=False, held_entity=None):
        try:
            if qpos_start is not None:
                traj = self.robot.plan_path(qpos_goal=qpos_goal, qpos_start=qpos_start,
                                       planner=planner, ignore_collision=ignore_collision, held_entity=held_entity)
            else:
                traj = self.robot.plan_path(qpos_goal=qpos_goal,
                                       planner=planner, ignore_collision=ignore_collision, held_entity=held_entity)
            if traj is None or (hasattr(traj, "__len__") and len(traj) == 0):
                return False, "empty_or_none_trajectory"

            return True, traj
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


def get_color(name: str) -> str:
    n = name.lower()
    color = "grey"
    if n == "celery":
        color = "green"
    elif n == "radish":
        color = "blue"
    elif n == "bacon":
        color = "magenta"
    elif n == "egg":
        color = "yellow"
    elif n == "chicken":
        color = "brown"
    elif n == "apple":
        color = "red"
    return color

def to_wxyz(q_xyzw):
    # PyBullet: (x, y, z, w) → Genesis: (w, x, y, z)
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])

def start_sim(json_path, method, prob_num, prob_idx, trial, repeat, num_distractor=0):
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
    kuka = KUKA()

    # add foods
    kuka.object_dict = getattr(kuka, "object_dict", {})
    for i, (name, info) in enumerate(blocks_info.items()):
        size = tuple(info["size"])  # (sx, sy, sz)
        pos0 = tuple([0.40, 0.80 + 0.10 * i, info["size"][2] / 2 + 0.01])
        col = COLOR_MAP.get(get_color(name), COLOR_MAP["grey"])

        ent = kuka.scene.add_entity(
            morph=gs.morphs.Box(
                size=size,
                pos=pos0,
                collision=True,
            ),
            surface=gs.surfaces.Rough(
                color=col,
            ),
        )
        kuka.object_dict[name] = ent

    # add distractors
    xs = [-0.15, 0.0, 0.15]  # fixed
    ys = [-0.4, 0.4, -0.6, 0.6]
    i=1
    for yy in ys:
        for xx in xs:
            if i > num_distractor:
                break
            ent = kuka.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(0.06, 0.06, 0.10),
                    pos=(xx, yy, 0.05),
                    fixed=True,
                    collision=True,
                ),
                surface=gs.surfaces.Rough(
                    color=(0.5, 0.5, 0.5, 1),
                ),
            )
            kuka.object_dict[f"dis{i}"] = ent
            i += 1

    # Build the scene
    kuka.scene.build()
    kuka.cam_front.start_recording()

    # Set control gains
    kuka.set_control_gains()
    link = kuka.robot.get_link(kuka.EE_FRAMES['ee'])

    # set pose of the blocks
    for name, info in blocks_info.items():
        ent = kuka.object_dict[name]
        pos = np.array(info["pose"]["position"], dtype=float)
        q_xyzw = info["pose"]["quaternion"]
        q_wxyz = np.array(to_wxyz(q_xyzw), dtype=float)

        ent.set_pos(pos)
        ent.set_quat(q_wxyz)

    for i in range(10):
        kuka.scene.step()

    screenshot_dir = Path(f"../experiments/kitchen/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = f"../experiments/kitchen/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}"
    file_path_list = kuka.save_snapshot4(screenshot_dir, node_name="node0")

    return kuka, file_path_list
