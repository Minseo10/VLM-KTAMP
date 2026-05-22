# from tf.transformations import quaternion_matrix
from attr import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
from typing import Any, Tuple, Optional
import math
import json
from typing import Union, List
import torch
import genesis as gs
import itertools


## 3D transformation functions
PointType = Tuple[float, float, float]
EulerType = PointType
QuatType = Tuple[float, float, float, float]
PoseType = Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]
_EPS = np.finfo(float).eps * 4.0


def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])


def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])


def Pose(point: Optional[PointType] = None, euler: Optional[EulerType] = None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return np.concatenate([point, quat_from_euler(euler)])


@dataclass
class AABB:
    lower: list
    upper: list


@dataclass
class Pixel:
    row: int
    column: int


@dataclass
class OOBB:
    aabb: AABB
    pose: Pose


@dataclass
class CameraImage:
    rgbPixels: Any
    depthPixels: Any
    segmentationMaskBuffer: Any
    camera_pose: Pose
    camera_matrix: Any


def dimensions_from_camera_image(camera_image: CameraImage):
    assert camera_image.rgbPixels.shape[:2] == camera_image.depthPixels.shape[:2]
    return camera_image.rgbPixels.shape[1], camera_image.rgbPixels.shape[0]


def tensor_to_aabb(aabb_tensor):
    aabb_np = aabb_tensor.detach().cpu().numpy() if torch.is_tensor(aabb_tensor) else aabb_tensor
    
    if aabb_np.ndim == 2 and aabb_np.shape == (2, 3):
        # Single AABB case: shape (2, 3)
        lower = aabb_np[0].tolist()
        upper = aabb_np[1].tolist()
        return AABB(lower, upper)
    elif aabb_np.ndim == 3 and aabb_np.shape[1:] == (2, 3):
        # Multiple AABB case: shape (n_envs, 2, 3)
        aabbs = []
        for i in range(aabb_np.shape[0]):
            lower = aabb_np[i, 0].tolist()
            upper = aabb_np[i, 1].tolist()
            aabbs.append(AABB(lower, upper))
        return aabbs
    else:
        raise ValueError(f"Expected tensor shape (2, 3) or (n_envs, 2, 3), got {aabb_np.shape}")

def pose7_to_T44(pose7):
    from tf.transformations import quaternion_matrix

    """
    Converts quaternion to transformation matrix
    pose7  : [x, y, z, qw, qx, qy, qz]  (list·tuple·ndarray)
    return : 4×4 homogeneous transform (numpy.ndarray)
    """
    x, y, z, qw, qx, qy, qz = pose7
    T = quaternion_matrix([qx, qy, qz, qw])
    T[0:3, 3] = [x, y, z]
    return T


def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        # Extract position and quaternion, handling tuple return from previous multiplyTransforms
        if isinstance(pose, tuple) and len(pose) == 2:
            # pose is (position, quaternion) from previous multiplyTransforms call
            pos1, quat1 = pose
        else:
            # pose is 7-element list [x,y,z,qx,qy,qz,qw]
            pos1, quat1 = pose[0:3], pose[3:7]
        
        if isinstance(next_pose, tuple) and len(next_pose) == 2:
            pos2, quat2 = next_pose
        else:
            pos2, quat2 = next_pose[0:3], next_pose[3:7]
        
        # Call multiplyTransforms and convert result to 7-element list for consistency
        result_pos, result_quat = p.multiplyTransforms(pos1, quat1, pos2, quat2)
        pose = list(result_pos) + list(result_quat)
    return pose


def get_difference(p1, p2):
    return np.array(list(p2)) - np.array(list(p1))


def get_distance(p1, p2, **kwargs):
    return get_length(get_difference(p1, p2), **kwargs)


def clip(value, min_value=-np.inf, max_value=+np.inf):
    return min(max(min_value, value), max_value)


def quat_angle_between(quat0, quat1):
    delta = p.getDifferenceQuaternion(quat0, quat1)
    d = clip(delta[-1], min_value=-1.0, max_value=1.0)
    angle = math.acos(d)
    return angle


def get_pose_distance(pose1, pose2):
    pos1, quat1 = pose1[0:3], pose1[3:]
    pos2, quat2 = pose2[0:3], pose2[3:]
    pos_distance = get_distance(pos1, pos2)
    ori_distance = quat_angle_between(quat1, quat2)
    return pos_distance, ori_distance


def convex_combination(x, y, w=0.5):
    return (1 - w) * np.array(x) + w * np.array(y)


def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def unit_point():
    return (0.0, 0.0, 0.0)


def unit_quat():
    return quat_from_euler([0, 0, 0])  # [X,Y,Z,W]


def unit_pose():
    return np.array(list(unit_point()) + list(unit_quat()))


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quat_combination(quat1, quat2, fraction=0.5):
    # return p.getQuaternionSlerp(quat1, quat2, interpolationFraction=fraction)
    return quaternion_slerp(quat1, quat2, fraction)


def interpolate_poses(pose1, pose2, pos_step_size=0.01, ori_step_size=np.pi / 16):
    pos1, quat1 = pose1[0:3], pose1[3:]
    pos2, quat2 = pose2[0:3], pose2[3:]
    num_steps = max(
        2,
        int(
            math.ceil(
                max(
                    np.divide(
                        get_pose_distance(pose1, pose2), [pos_step_size, ori_step_size]
                    )
                )
            )
        ),
    )
    yield pose1
    for w in np.linspace(0, 1, num=num_steps, endpoint=True)[1:-1]:
        pos = convex_combination(pos1, pos2, w=w)
        quat = quat_combination(quat1, quat2, fraction=w)
        yield (pos, quat)
    yield pose2


def invert_pose(pose):
    """
    Args:
        pose: tuple (position, quaternion)
            - position: iterable of 3 floats (x, y, z)
            - quaternion: iterable of 4 floats (qx, qy, qz, qw)  # x,y,z,w ordering

    Returns:
        inverse_pose: (inv_position, inv_quaternion) with same structure
    """
    pos, quat = pose[0:3], pose[3:]
    # unpack
    x, y, z = pos
    qx, qy, qz, qw = quat

    # build a Rotation, note scipy expects [x, y, z, w]
    rot = R.from_quat([qx, qy, qz, qw])
    # inverse rotation
    rot_inv = rot.inv()

    # inverse translation = -R_inv * original_translation
    inv_pos = rot_inv.apply([-x, -y, -z])

    # inverse quaternion
    qx_i, qy_i, qz_i, qw_i = rot_inv.as_quat()  # still [x,y,z,w]

    return np.array([inv_pos[0], inv_pos[1], inv_pos[2], qx_i, qy_i, qz_i, qw_i])


def get_approach_vector_from_pose(pose, robot):
    """
    place_pose: [x, y, z, qx, qy, qz, qw]
    returns: unit approach vector in world frame as [vx, vy, vz]
    """
    qx, qy, qz, qw = pose[3:]
    rot = R.from_quat([qx, qy, qz, qw])

    if robot=="kuka" or robot=="franka":
        approach = rot.apply([0, 0, 1]).tolist()
    elif robot=="pr2" or robot=="dual_arm":
        approach = rot.apply([1, 0, 0]).tolist()
    return approach


def xyzw_to_wxyz(pose_xyzw):
    """
    Converts pose from xyzw to wxyz
    pose_xyzw: [x, y, z, qx, qy, qz, qw] (list)
    return: [x, y, z, qw, qx, qy, qz] (list)
    """
    pose_wxyz = [pose_xyzw[0], pose_xyzw[1], pose_xyzw[2], pose_xyzw[6], pose_xyzw[3], pose_xyzw[4], pose_xyzw[5]]
    return pose_wxyz


def posewxyz_to_T44(pose7):
    """
    Converts quaternion to transformation matrix
    pose7  : [x, y, z, qw, qx, qy, qz]  (list·tuple·ndarray)
    return : 4×4 homogeneous transform (numpy.ndarray)
    """
    x, y, z, qw, qx, qy, qz = pose7
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat(quat=[qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def quat_sim_to_cur(sim_wrapper, left):
    # choose orientation components from discrete set {-0.5, 0.5}
    link_name = sim_wrapper.EE_FRAMES['left'] if left else sim_wrapper.EE_FRAMES['right']  # dependable on robot
    quat_tensor = sim_wrapper.robot.get_link(link_name).get_quat()  # wxyz
    current_quat = quat_tensor.cpu().detach().numpy()

    # 8 candidates (qx,qy,qz ∈ {-0.5,0.5}, qw=0.5)
    candidates = [
        np.array([qx0, qy0, qz0, 0.5])
        for qx0 in (-0.5, 0.5)
        for qy0 in (-0.5, 0.5)
        for qz0 in (-0.5, 0.5)
    ]
    cur = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])

    # selected candidate which has the highest similarity to current quaternion
    best = max(candidates, key=lambda q: abs(np.dot(cur, q)))
    qx, qy, qz, qw = best.tolist()
    return qx, qy, qz, qw


def get_length(vec, norm=2):
    return np.linalg.norm(vec, ord=norm)


def get_unit_vector(vec):
    norm = get_length(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm


def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler)


def point_from_pose(pose):
    return pose[0:3]


def invert(pose):
    point, quat = pose[0:3], pose[3:]
    return p.invertTransform(point, quat)


def tform_point(affine, point):
    return point_from_pose(multiply(affine, Pose(point=point)))


def tform_points(affine, points):
    return [tform_point(affine, p) for p in points]


def pixel_from_ray(camera_matrix, ray):
    return camera_matrix.dot(np.array(ray) / ray[2])[:2]


def pixel_from_point(camera_matrix, point_camera, width, height):
    px, py = pixel_from_ray(camera_matrix, point_camera)
    if (0 <= px < width) and (0 <= py < height):
        r, c = np.floor([py, px]).astype(int)
        return Pixel(r, c)
    return None


def aabb_from_points(points):
    return AABB(np.min(points, axis=0), np.max(points, axis=0))


def get_aabb_vertices(aabb: AABB):
    d = len(aabb.lower)
    return [
        tuple([aabb.lower, aabb.upper][i[k]][k] for k in range(d))
        for i in itertools.product(range(2), repeat=d)
    ]


def safe_zip(sequence1, sequence2):
    sequence1, sequence2 = list(sequence1), list(sequence2)
    assert len(sequence1) == len(sequence2)
    return list(zip(sequence1, sequence2))


def draw_oobb(oobb: OOBB, color=(0, 0, 1, 1), sim_wrapper=None):
    handles = []
    debug_box = sim_wrapper.__annotations__scene.draw_debug_box(
        bounds=[oobb.aabb.lower, oobb.aabb.upper],
        color=color,
        wireframe=False,
        wireframe_radius=0.005,
    )
    handles.append(debug_box)
    return handles


def transformation_to_pose(trans):
    matrix = np.array(trans)

    # Extract the rotation matrix (top-left 3x3)
    rotation_matrix = matrix[:3, :3]

    # Extract the translation vector (first three elements of the fourth column)
    translation_vector = matrix[:3, 3]

    # Convert the rotation matrix to a quaternion
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return (translation_vector, quaternion)


# json utils
def minify_json(data_or_path: Union[dict, list, str]) -> str:

    if isinstance(data_or_path, (dict, list)):
        obj = data_or_path
    else:
        with open(data_or_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def chunk_text(s: str, max_chars: int = 12000) -> List[str]:

    return [s[i:i+max_chars] for i in range(0, len(s), max_chars)]


def _to_torch_vec(x):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=gs.tc_float, device=gs.device).contiguous()
    return torch.as_tensor(x, dtype=gs.tc_float, device=gs.device).contiguous()


def _as_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _json_safe(obj: Any):
    """Convert common non-JSON-native objects (e.g., numpy arrays/scalars)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)

    # numpy arrays and numpy scalar values without hard dependency on numpy
    if hasattr(obj, "tolist"):
        try:
            return _json_safe(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    return obj


# kinodynamicTAMP utils
def check_place_pose_collision(
    entity,
    pose_xyzw,                      # [x, y, z, qx, qy, qz, qw]
    *,
    restore=True,
    ignore_entities=None,           # [entity_a, entity_b, ...]
):
    """
    Temporarily move entity to candidate pose and check collision.
    Args:
        entity: Genesis Entity object
        pose_xyzw: world pose [x y z qx qy qz qw]
        ignore_entities: entities to ignore collisions with (table, support, robot, etc.)
    Returns:
        is_free: True if no collision excluding ignore_entities
        raw_pairs: raw result from entity.detect_collision()
        kept_pairs: collision pairs after filtering ignored entities
    """
    pose_xyzw = list(pose_xyzw)
    pose_wxyz = xyzw_to_wxyz(pose_xyzw)
    pos = _to_torch_vec(pose_wxyz[:3])
    quat = _to_torch_vec(pose_wxyz[3:])

    # Backup current pose
    old_pos  = entity.get_pos()   # base-link pos (torch)
    old_quat = entity.get_quat()  # base-link quat (torch)

    # Temporarily move to candidate pose
    entity.set_pos(pos)
    entity.set_quat(quat)

    try:
        # Raw collision pairs (Nx2, each value is geom id)
        raw_pairs = entity.detect_collision()
        raw_pairs_np = _as_numpy(raw_pairs)
        if raw_pairs_np.size == 0:
            return True, raw_pairs_np, raw_pairs_np

        # Filter ignored entities
        kept_mask = np.ones(len(raw_pairs_np), dtype=bool)
        if ignore_entities:
            # Geometry range of this entity
            e_lo, e_hi = int(entity.geom_start), int(entity.geom_end)

            # Pre-collect geometry ranges of ignored entities
            ign_ranges = [(int(ign.geom_start), int(ign.geom_end)) for ign in ignore_entities]

            def _in_range(gid, lo, hi):
                return (gid >= lo) and (gid < hi)

            for i, (a, b) in enumerate(raw_pairs_np):
                # One of (a,b) is always in entity geom range
                # Skip if other side is in ignore range
                if _in_range(a, e_lo, e_hi) and not _in_range(b, e_lo, e_hi):
                    other_id = int(b)
                elif _in_range(b, e_lo, e_hi) and not _in_range(a, e_lo, e_hi):
                    other_id = int(a)
                else:
                    # Both in entity range or both outside (rare)
                    other_id = None

                if other_id is not None:
                    for lo, hi in ign_ranges:
                        if _in_range(other_id, lo, hi):
                            kept_mask[i] = False
                            break

        kept_pairs = raw_pairs_np[kept_mask]
        is_free = (len(kept_pairs) == 0)
        return bool(is_free), raw_pairs_np, kept_pairs

    finally:
        if restore:
            entity.set_pos(old_pos, zero_velocity=True)
            entity.set_quat(old_quat, zero_velocity=True)


def append_error(continuous_params: dict, feedback: str, sep: str=" | "):
    """Append feedback to continuous_params['error'] string."""
    if not feedback:
        return
    prev = continuous_params.get("error")
    if not prev:
        continuous_params["error"] = str(feedback)
    else:
        fb = str(feedback)
        if fb not in prev:
            continuous_params["error"] = f"{prev}{sep}{fb}"
    return continuous_params


def check_on(sim_wrapper, obj, underobj, tol=0.06):
    aabb_obj = sim_wrapper.scene.entities[sim_wrapper.object_dict[obj]].get_AABB().tolist()
    min_bb_obj = [v for v in aabb_obj[0]]
    max_bb_obj = [v for v in aabb_obj[1]]
    if underobj == 'mytable':
        if (
                min_bb_obj[0] > -0.7 - tol
                and max_bb_obj[0] < 0.7 + tol
                and min_bb_obj[1] > -0.7 - tol
                and max_bb_obj[1] < 0.7 + tol
                and min_bb_obj[2] > 0.0 - tol
        ):
            return True
        return False
    else:
        aabb_underobj = sim_wrapper.scene.entities[sim_wrapper.object_dict[underobj]].get_AABB().tolist()
        min_bb_underobj = [v for v in aabb_underobj[0]]
        max_bb_underobj = [v for v in aabb_underobj[1]]
        if (
                min_bb_obj[0] > min_bb_underobj[0] - tol
                and max_bb_obj[0] < max_bb_underobj[0] + tol
                and min_bb_obj[1] > min_bb_underobj[1] - tol
                and max_bb_obj[1] < max_bb_underobj[1] + tol
                and min_bb_obj[2] > max_bb_underobj[2] - tol
                and min_bb_obj[2] < max_bb_underobj[2] + tol
        ):
            return True
        return False


def check_on_table(robot_name, sim_wrapper, obj, tol=0.05):
    aabb_obj = sim_wrapper.scene.entities[sim_wrapper.object_dict[obj]].get_AABB().tolist()
    min_bb_obj = [v for v in aabb_obj[0]]
    max_bb_obj = [v for v in aabb_obj[1]]

    if robot_name == 'pr2':
        table = [[0.35, 0.75], [-0.65, 0.65], [0.71, 0.72]]
    elif robot_name == 'dual_arm':
        table = [[-0.775, -0.30], [-0.83, -0.0], [0.71, 0.72]]
    elif robot_name == 'franka':
        table = [[-0.7, 0.7], [-0.7, 0.7], [0.0, 0.5]]

    if (
            min_bb_obj[0] > table[0][0] - tol
            and max_bb_obj[0] < table[0][1] + tol
            and min_bb_obj[1] > table[1][0] - tol
            and max_bb_obj[1] < table[1][1] + tol
            and min_bb_obj[2] > table[2][0] - tol
    ):
        return True
    return False


def check_holding(sim_wrapper, obj, tol=0.06):
    aabb = sim_wrapper.scene.entities[sim_wrapper.object_dict[obj]].get_AABB().tolist()
    min_pt = np.array(aabb[0])
    max_pt = np.array(aabb[1])
    center = (min_pt + max_pt) / 2

    current_ee_pose = sim_wrapper.current_ee_pose()
    pos = np.array(current_ee_pose[:3])

    if(
        -tol< center[0] - pos[0] < tol
        and -tol< center[1] - pos[1] < tol
        and -tol< center[2] - pos[2] < tol
    ):
        return True
    return False

