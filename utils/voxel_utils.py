import os
import time
from itertools import product

import numpy as np
import pybullet as p
from utils.utils import *

# import tampura_environments.panda_utils.pb_utils as pbu

MAX_TEXTURE_WIDTH = 418  # max square dimension
MAX_PIXEL_VALUE = 2**8 - 1
MAX_LINKS = 125  # Max links seems to be 126



class VoxelGrid(object):
    def __init__(
        self,
        resolutions,
        default=bool,
        world_from_grid=unit_pose(),  # [0,0,0,0,0,0,1]
        aabb=None,
        color=(1, 0, 0, 0.5),
        **kwargs,
    ):
        # def __init__(self, sizes, centers, pose=unit_pose()):
        # TODO: defaultdict
        # assert len(sizes) == len(centers)
        assert callable(default)
        self.resolutions = resolutions
        self.default = default
        self.value_from_voxel = {}
        self.world_from_grid = world_from_grid
        self.aabb = aabb  # TODO: apply  ###
        self.color = color
        self.occupied_points = []
        self.occupied_voxel_points = []
        self.cloud_handles = []

        # self.bodies = None
        # TODO: store voxels more intelligently spatially


    def get_frontier(self):
        twod = self.project2d()
        (
            xs,
            ys,
            _,
        ) = zip(*twod)

        twod_array = np.zeros((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
        for i, j, _ in twod:
            twod_array[i - min(xs), j - min(ys)] = 1

        frontiers = []
        for x in range(twod_array.shape[0]):
            for y in range(twod_array.shape[1]):
                frontier = False
                for diffx in [-1, 1]:
                    for diffy in [-1, 1]:
                        if (
                            twod_array[x, y] == 1
                            and x + diffx >= 0
                            and x + diffx < twod_array.shape[0]
                            and y + diffy >= 0
                            and y + diffy < twod_array.shape[1]
                        ):
                            if twod_array[x + diffx, y + diffy] == 0:
                                frontier = True
                if frontier:
                    point = self.pose_from_voxel((x, y, 1))[0]
                    frontiers.append((point[0], point[1]))

        return frontiers

    @property
    def occupied(self):  # TODO: get_occupied
        return sorted(self.value_from_voxel)

    def __iter__(self):
        return iter(self.value_from_voxel)

    def __len__(self):
        return len(self.value_from_voxel)

    def copy(self):  # TODO: deepcopy
        new_grid = VoxelGrid(
            self.resolutions, self.default, self.world_from_grid, self.aabb, self.color
        )
        new_grid.value_from_voxel = dict(self.value_from_voxel)
        return new_grid

    def to_grid(self, point_world):
        return tform_point(invert(self.world_from_grid), point_world)

    def to_world(self, point_grid):  ###
        return tform_point(self.world_from_grid, point_grid)

    def voxel_from_point(self, point):
        point_grid = self.to_grid(point)
        return tuple(np.floor(np.divide(point_grid, self.resolutions)).astype(int))

    # def voxels_from_aabb_grid(self, aabb):
    #    voxel_lower, voxel_upper = map(self.voxel_from_point, aabb)
    #    return map(tuple, product(*[range(l, u + 1) for l, u in safe_zip(voxel_lower, voxel_upper)]))
    def voxels_from_aabb(self, aabb):  ###
        aabb = aabb_from_points(
            [self.voxel_from_point(point) for point in get_aabb_vertices(aabb)]
        )
        return map(
            tuple,
            product(
                *[range(l, u + 1) for l, u in safe_zip(aabb.lower, aabb.upper)]
            ),
        )

    def occupied_voxels_from_aabb(self, aabb):
        vis_points = np.array(self.occupied_points)
        vis_voxels = np.array(self.occupied_voxel_points)
        vis_idx = np.all(
            (aabb.lower <= vis_points) & (vis_points <= aabb.upper), axis=1
        )
        return list([tuple(vp) for vp in vis_voxels[vis_idx]])

    def occupied_voxels_points_from_aabb(self, aabb):
        vis_points = np.array(self.occupied_points)
        vis_idx = np.all(
            (aabb.lower <= vis_points) & (vis_points <= aabb.upper), axis=1
        )
        return vis_points[vis_idx]

    # Grid coordinate frame
    def lower_from_voxel(self, voxel):
        return np.multiply(voxel, self.resolutions)  # self.to_world(

    def center_from_voxel(self, voxel):  ###
        return self.lower_from_voxel(np.array(voxel) + 0.5)

    def upper_from_voxel(self, voxel):
        return self.lower_from_voxel(np.array(voxel) + 1.0)

    def aabb_from_voxel(self, voxel):
        return AABB(self.lower_from_voxel(voxel), self.upper_from_voxel(voxel))

    def ray_trace(self, start_cell, goal_point):
        # TODO: finish adapting
        if self.is_occupied(start_cell):
            return [], False
        goal_cell = self.get_index(goal_point)
        start_point = self.get_center(start_cell)
        unit = goal_point - start_point
        unit /= np.linalg.norm(unit)
        direction = (unit / np.abs(unit)).astype(int)

        path = []
        current_point = start_point
        current_cell = start_cell
        while current_cell != goal_cell:
            path.append(current_cell)
            min_k, min_t = None, float("inf")
            for k, sign in enumerate(direction):
                next_point = (
                    self.get_min(current_cell)
                    if sign < 0
                    else self.get_max(current_cell)
                )
                t = ((next_point - current_point) / direction)[k]
                assert t > 0
                if (t != 0) and (t < min_t):
                    min_k, min_t = k, t
            assert min_k is not None
            current_point += min_t * unit
            current_cell = np.array(current_cell, dtype=int)
            current_cell[min_k] += direction[min_k]
            current_cell = tuple(current_cell)
            if self.is_occupied(current_cell):
                return path, False
        return path, True

    # World coordinate frame
    def pose_from_voxel(self, voxel):
        pose_grid = Pose(self.center_from_voxel(voxel))
        return multiply(self.world_from_grid, pose_grid)

    def vertices_from_voxel(self, voxel):
        return list(
            map(self.to_world, get_aabb_vertices(self.aabb_from_voxel(voxel)))
        )

    def contains(self, voxel):  # TODO: operator versions
        return voxel in self.value_from_voxel

    def get_value(self, voxel):
        assert self.contains(voxel)
        return self.value_from_voxel[voxel]

    def set_value(self, voxel, value):
        # TODO: remove if value == default
        self.value_from_voxel[voxel] = value

    def remove_value(self, voxel): ###
        if self.contains(voxel):
            self.value_from_voxel.pop(voxel)  # TODO: return instead?

    is_occupied = contains  ###

    def set_occupied(self, voxel):
        if self.is_occupied(voxel):
            return False
        self.set_value(voxel, value=self.default())
        self.occupied_points.append(list(self.center_from_voxel(voxel)))
        self.occupied_voxel_points.append(voxel)
        return True

    def set_free(self, voxel): ### Removes voxels only from data structures, not visualization
        if not self.is_occupied(voxel):
            return False
        self.remove_value(voxel)
        idx = self.occupied_points.index(list(self.center_from_voxel(voxel)))
        self.occupied_points.remove(self.occupied_points[idx])
        self.occupied_voxel_points.remove(self.occupied_voxel_points[idx])
        return True

    def get_neighbors(self, index):
        for i in range(len(index)):
            direction = np.zeros(len(index), dtype=int)
            for n in (-1, +1):
                direction[i] = n
                yield tuple(np.array(index) + direction)

    def get_clusters(self, voxels=None):
        if voxels is None:
            voxels = self.occupied
        clusters = []
        assigned = set()

        def dfs(current):
            if (current in assigned) or (not self.is_occupied(current)):
                return []
            cluster = [current]
            assigned.add(current)
            for neighbor in self.get_neighbors(current):
                cluster.extend(dfs(neighbor))
            return cluster

        for voxel in voxels:
            cluster = dfs(voxel)
            if cluster:
                clusters.append(cluster)
        return clusters

    def add_point(self, point):
        self.set_occupied(self.voxel_from_point(point))

    def add_aabb(self, aabb):
        for voxel in self.voxels_from_aabb(aabb):
            self.set_occupied(voxel)

    def draw_voxel(self, voxel, color=None, sim_wrapper=None):
        if color is None:
            color = self.color
        aabb = self.aabb_from_voxel(voxel)
        return draw_oobb(
            OOBB(aabb, self.world_from_grid), color=color, sim_wrapper=sim_wrapper
        )

    def create_intervals(self): ###
        voxel_heights = {}
        for i, j, k in self.occupied:
            voxel_heights.setdefault((i, j), set()).add(k)
        voxel_intervals = []
        for i, j in voxel_heights:
            heights = sorted(voxel_heights[i, j])
            start = last = heights[0]
            for k in heights[1:]:
                if k == last + 1:
                    last = k
                else:
                    interval = (start, last)
                    voxel_intervals.append((i, j, interval))
                    start = last = k
            interval = (start, last)
            voxel_intervals.append((i, j, interval))

        return voxel_intervals

    def draw_intervals(self, sim_wrapper): ###
        for debug_box in sim_wrapper.debug_objects:
            sim_wrapper.scene.clear_debug_object(debug_box)

        handles = []
        for i, j, (k1, k2) in self.create_intervals():
            voxels = [(i, j, k1), (i, j, k2)]
            aabb = aabb_from_points(
                [
                    extrema
                    for voxel in voxels
                    for extrema in [
                        self.aabb_from_voxel(voxel).lower,
                        self.aabb_from_voxel(voxel).upper,
                    ]
                ]
            )
            handles.extend(
                draw_oobb(
                    OOBB(aabb, self.world_from_grid),
                    color=[self.color.red, self.color.green, self.color.blue],
                    sim_wrapper=sim_wrapper,
                )
            )
        return handles

    def project2d(self):
        # TODO: combine adjacent voxels into larger lines
        # TODO: greedy algorithm that combines lines/boxes
        # TODO: combine intervals
        tallest_voxel = {}
        for i, j, k in self.occupied:
            tallest_voxel[i, j] = max(k, tallest_voxel.get((i, j), k))
        return {(i, j, k) for (i, j), k in tallest_voxel.items()}
