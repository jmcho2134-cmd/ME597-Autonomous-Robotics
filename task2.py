#!/usr/bin/env python3

import math
import time
from collections import deque
from heapq import heappush, heappop
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan


# =========================
# Parameters
# =========================
wall_buffer_m   = 0.15   # collision inflation [m]
allow_diag      = True   # allow diagonal moves
lookahead_m     = 0.45
waypoint_tol_m  = 0.22

# Velocity limits
max_lin_vel = 0.2
max_ang_vel = 1.2

# Controller gains / thresholds
kp_lin   = 1.3
kp_yaw   = 1.6
ki_yaw   = 0.0
kd_yaw   = 0.08

face_goal_rad   = 0.7
goal_tol_m      = 0.05

# ===== Cost-based A* (stay away from walls) =====
USE_COST_ASTAR     = True
UNKNOWN_COST       = 0.0     # unknown cell penalty
OBSTACLE_COST_GAIN = 10.0
OBSTACLE_SAFE_DIST = 0.65

# ===== LaserScan-based dynamic obstacles =====
LASER_FRONT_DEG       = 45.0    # forward range
LASER_OBS_DIST        = 1.0     # distance to recognize as obstacle
PATH_OBS_HIT_DIST     = 0.65     # subpath replanning threshold
DYN_OBS_INFLATION_RAD = 0.15    # obstacle inflation radius

# Static wall filter threshold:
# In raycasting, if the difference between the map wall distance and the LaserScan distance
# is less than or equal to this value, it is considered a static wall [m].
STATIC_HIT_TOL = 0.04

# Dynamic obstacle candidates within this distance of a static wall are ignored
# (since cost-based wall avoidance already handles that area).
DYN_IGNORE_NEAR_STATIC = max(OBSTACLE_SAFE_DIST - 0.05, 0.0)


# =========================
# Utilities
# =========================
def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# ============ Map Processor ===============
class MapProcessor:
    def __init__(self):
        self.grid: Optional[List[int]] = None      # static occupancy [-1,0..100]
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.inflated: Optional[List[int]] = None
        self.obs_dist_cells: Optional[List[float]] = None
        self.dynamic_occ: Optional[List[int]] = None
        self.ready = False

    def update_from_msg(self, msg: OccupancyGrid, inflation_radius_m: float = wall_buffer_m):
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.grid = list(msg.data)
        self.dynamic_occ = None

        # Compute static distance map + collision inflation.
        self.inflated = self._inflate_grid(self.grid, self.width, self.height, inflation_radius_m)
        self.ready = True

    def _inflate_grid(self, data: List[int], w: int, h: int, inflation_radius_m: float) -> List[int]:
        if w == 0 or h == 0 or data is None:
            self.obs_dist_cells = None
            return []

        occ_static = [1 if (v > 0) else 0 for v in data]

        q = deque()
        dist = [math.inf] * (w * h)

        for idx, v in enumerate(occ_static):
            if v == 1:
                dist[idx] = 0.0
                q.append(idx)

        nbrs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            i = q.popleft()
            x = i % w
            y = i // w
            for dx, dy in nbrs4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    j = ny * w + nx
                    if dist[j] == math.inf:
                        dist[j] = dist[i] + 1
                        q.append(j)

        self.obs_dist_cells = dist

        # Collision inflation: unknown + near-obstacle cells.
        cells = int(math.ceil(inflation_radius_m / max(self.resolution, 1e-6)))
        inflated = [0] * (w * h)
        for i, d in enumerate(dist):
            v = data[i]
            if v < 0:
                # Unknown region is always blocked.
                inflated[i] = 1
            elif d <= cells:
                inflated[i] = 1
            else:
                inflated[i] = 0

        # Apply dynamic obstacle layer (if exists).
        if self.dynamic_occ is not None:
            for i, dv in enumerate(self.dynamic_occ):
                if dv != 0:
                    inflated[i] = 1

        return inflated

    # Inflated map publishing for RViz debugging.
    def to_inflated_occgrid(self, stamp) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'

        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.resolution = self.resolution
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        if (not self.ready) or self.inflated is None or self.width == 0 or self.height == 0:
            msg.data = [0] * (self.width * self.height)
        else:
            msg.data = [100 if v != 0 else 0 for v in self.inflated]

        return msg

    # Map conversions
    def pose_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        gx = int((x_m - self.origin_x) / self.resolution + 0.5)
        gy = int((y_m - self.origin_y) / self.resolution + 0.5)
        return gx, gy

    def grid_to_pose(self, gx: int, gy: int) -> Tuple[float, float]:
        x = gx * self.resolution + self.origin_x
        y = gy * self.resolution + self.origin_y
        return x, y

    def in_bounds(self, gx: int, gy: int) -> bool:
        return (0 <= gx < self.width) and (0 <= gy < self.height)

    def is_free(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return False
        idx = gy * self.width + gx
        return self.inflated[idx] == 0

    # ----- Dynamic obstacle layer -----
    def add_dynamic_obstacle_world(self, x_m: float, y_m: float, radius_m: float):
        if not self.ready or self.width == 0 or self.height == 0:
            return

        gx, gy = self.pose_to_grid(x_m, y_m)
        if not self.in_bounds(gx, gy):
            return

        if self.dynamic_occ is None:
            self.dynamic_occ = [0] * (self.width * self.height)
        if self.grid is None:
            return

        cells = int(math.ceil(radius_m / max(self.resolution, 1e-6)))
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                nx, ny = gx + dx, gy + dy
                if not self.in_bounds(nx, ny):
                    continue
                if math.hypot(dx, dy) * self.resolution <= radius_m:
                    idx = ny * self.width + nx
                    # Dynamic layer mark
                    self.dynamic_occ[idx] = 1
                    # Also mark static grid as occupied (if it was free/unknown)
                    if self.grid[idx] <= 0:   # 0 or -1
                        self.grid[idx] = 100  # occupied

        # Recompute distance map + inflation.
        self.inflated = self._inflate_grid(
            self.grid,
            self.width,
            self.height,
            wall_buffer_m
        )

    # ----- Distance to static obstacle (world) -----
    def obstacle_dist_m_world(self, x_m: float, y_m: float) -> float:
        if self.obs_dist_cells is None:
            return 1e9
        gx, gy = self.pose_to_grid(x_m, y_m)
        if not self.in_bounds(gx, gy):
            return 1e9
        idx = gy * self.width + gx
        d_cells = self.obs_dist_cells[idx]
        if d_cells == math.inf:
            return 1e9
        return d_cells * self.resolution

    # ----- Raycast distance to static obstacle along a beam -----
    def raycast_to_static_obstacle(self,
                                   x0: float, y0: float,
                                   phi: float,
                                   max_range: float) -> Optional[float]:
        if self.grid is None or not self.ready:
            return None

        step = max(self.resolution * 0.5, 0.02)
        s = 0.0
        while s <= max_range:
            x = x0 + s * math.cos(phi)
            y = y0 + s * math.sin(phi)
            gx, gy = self.pose_to_grid(x, y)
            if not self.in_bounds(gx, gy):
                break
            idx = gy * self.width + gx
            if self.grid[idx] > 0:
                return s
            s += step
        return None


# ============ A* Algorithm (with cost map) ===============
class AStar:
    def __init__(self, mp: MapProcessor, use_8_conn: bool = allow_diag):
        self.mp = mp
        if use_8_conn:
            self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]
            self.step_cost = [1, 1, 1, 1,
                              math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]
        else:
            self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self.step_cost = [1, 1, 1, 1]

    def h(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

    def _cell_cost(self, idx: int) -> float:
        if not USE_COST_ASTAR:
            return 0.0
        if self.mp.grid is None or self.mp.obs_dist_cells is None:
            return 0.0

        gval = self.mp.grid[idx]
        dist_cells = self.mp.obs_dist_cells[idx]
        dist_m = dist_cells * self.mp.resolution if dist_cells < math.inf else 1e9

        cost = 0.0
        # unknown cell penalty
        if gval == -1:
            cost += UNKNOWN_COST

        # wall close penalty
        if dist_m < OBSTACLE_SAFE_DIST:
            cost += OBSTACLE_COST_GAIN * (OBSTACLE_SAFE_DIST - dist_m) / OBSTACLE_SAFE_DIST

        return cost

    def plan(self, start_g: Tuple[int, int], goal_g: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        if not self.mp.in_bounds(*start_g) or not self.mp.in_bounds(*goal_g):
            return None
        if (not self.mp.is_free(*start_g)) or (not self.mp.is_free(*goal_g)):
            return None

        w, h = self.mp.width, self.mp.height
        gscore = [math.inf] * (w * h)
        came = [-1] * (w * h)

        def idx(p): return p[1] * w + p[0]

        pq = []
        gscore[idx(start_g)] = 0.0
        heappush(pq, (self.h(start_g, goal_g), 0.0, start_g))

        while pq:
            f, g, cur = heappop(pq)
            if cur == goal_g:
                path = [cur]
                ci = idx(cur)
                while came[ci] != -1:
                    ci = came[ci]
                    path.append((ci % w, ci // w))
                path.reverse()
                return path

            cx, cy = cur
            base_i = idx(cur)
            for k, (dx, dy) in enumerate(self.neighbors):
                nx, ny = cx + dx, cy + dy
                if not self.mp.in_bounds(nx, ny):
                    continue
                if not self.mp.is_free(nx, ny):
                    continue
                ni = ny * w + nx

                base_step = self.step_cost[k]
                extra = self._cell_cost(ni)
                tentative = gscore[base_i] + base_step + extra

                if tentative < gscore[ni]:
                    gscore[ni] = tentative
                    came[ni] = base_i
                    heappush(pq, (tentative + self.h((nx, ny), goal_g), tentative, (nx, ny)))
        return None


# ============ Navigation ===============
class Task2(Node):
    def __init__(self):
        super().__init__('task2_node')

        # States
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        self.has_goal = False
        self.map_ready = False

        # remember last goal
        self.last_goal_xy: Optional[Tuple[float, float]] = None

        # following
        self.path_pts: List[Tuple[float, float]] = []
        self.target_idx = 0

        # control memory
        self.prev_yaw_err = 0.0
        self.ang_err_int = 0.0

        # map + planner
        self.map_proc = MapProcessor()
        self.planner = AStar(self.map_proc, use_8_conn=allow_diag)

        # Subscribers
        self.create_subscription(PoseStamped,
                                 '/move_base_simple/goal',
                                 self.__goal_pose_cbk,
                                 10)
        self.create_subscription(PoseWithCovarianceStamped,
                                 '/amcl_pose',
                                 self.__ttbot_pose_cbk,
                                 10)

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid,
                                 '/map',
                                 self.__map_cbk,
                                 map_qos)

        # LaserScan for dynamic obstacles
        self.create_subscription(LaserScan,
                                 '/scan',
                                 self.__scan_cbk,
                                 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time', 10)  # DO NOT MODIFY

        inflated_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.inflated_map_pub = self.create_publisher(
            OccupancyGrid, 'inflated_map', inflated_qos
        )

        self.loop_hz = 10.0

    # ----- Callbacks -----
    def __map_cbk(self, msg: OccupancyGrid):
        self.map_proc.update_from_msg(msg, wall_buffer_m)
        self.map_ready = True

        # map changed -> force next plan
        self.last_goal_xy = None
        self.path_pts = []
        self.path = Path()

        self.get_logger().info(
            f'map received: {self.map_proc.width}x{self.map_proc.height}, '
            f'res={self.map_proc.resolution:.3f}'
        )

        # inflated map publish (RViz)
        inf_msg = self.map_proc.to_inflated_occgrid(self.get_clock().now().to_msg())
        self.inflated_map_pub.publish(inf_msg)

    def __goal_pose_cbk(self, data: PoseStamped):
        self.goal_pose = data
        self.has_goal = True

        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(
                self.goal_pose.pose.position.x,
                self.goal_pose.pose.position.y
            )
        )

    def __ttbot_pose_cbk(self, data: PoseWithCovarianceStamped):
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose = data.pose.pose

    # ----- Subpath replanning helper (A*) -----
    def _replan_subpath_around_obstacle(self, obs_x: float, obs_y: float) -> bool:
        if not self.path_pts or len(self.path_pts) < 3:
            return False

        N = len(self.path_pts)

        # 1) find nearest path index close to obstacle (after current target)
        min_d = 1e9
        obs_idx = -1
        for i in range(self.target_idx, N):
            px, py = self.path_pts[i]
            d = math.hypot(px - obs_x, py - obs_y)
            if d < min_d:
                min_d = d
                obs_idx = i

        if obs_idx < 0 or min_d > PATH_OBS_HIT_DIST * 1.5:
            return False

        # 2) find two safe points that are far enough from obstacle
        SAFE_CLEAR_RADIUS = 0.65  # subpath start must be at least this far from obstacle

        # backward safe point (toward robot)
        sub_start_idx = None
        for i in range(obs_idx - 1, self.target_idx - 1, -1):
            px, py = self.path_pts[i]
            if math.hypot(px - obs_x, py - obs_y) >= SAFE_CLEAR_RADIUS:
                sub_start_idx = i
                break

        # forward safe point (toward goal)
        sub_goal_idx = None
        for i in range(obs_idx + 1, N):
            px, py = self.path_pts[i]
            if math.hypot(px - obs_x, py - obs_y) >= SAFE_CLEAR_RADIUS:
                sub_goal_idx = i
                break

        if sub_start_idx is None or sub_goal_idx is None or sub_goal_idx <= sub_start_idx:
            return False

        sub_start_xy = self.path_pts[sub_start_idx]
        sub_goal_xy = self.path_pts[sub_goal_idx]

        # 3) Replan only that segment with A* (world coordinates)
        sg = self.map_proc.pose_to_grid(*sub_start_xy)
        gg = self.map_proc.pose_to_grid(*sub_goal_xy)
        sub_grid_path = self.planner.plan(sg, gg)
        if sub_grid_path is None or len(sub_grid_path) < 2:
            return False

        sub_path_xy: List[Tuple[float, float]] = []
        for gx, gy in sub_grid_path:
            x, y = self.map_proc.grid_to_pose(gx, gy)
            sub_path_xy.append((x, y))

        # 4) Replace sub_start_idx ~ sub_goal_idx in the original path with the new subpath
        new_path_pts: List[Tuple[float, float]] = []
        new_path_pts.extend(self.path_pts[:sub_start_idx])
        new_path_pts.extend(sub_path_xy)
        new_path_pts.extend(self.path_pts[sub_goal_idx + 1:])

        # 5) Rebuild Path message
        new_path_msg = Path()
        new_path_msg.header.stamp = self.get_clock().now().to_msg()
        new_path_msg.header.frame_id = 'map'

        for x, y in new_path_pts:
            ps = PoseStamped()
            ps.header = new_path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            new_path_msg.poses.append(ps)

        self.path_pts = new_path_pts
        self.path = new_path_msg
        self.path_pub.publish(self.path)

        self.target_idx = int(clamp(self.target_idx, 0, len(self.path_pts) - 1))

        self.get_logger().info(
            f'A* subpath replanned around obstacle at ({obs_x:.2f}, {obs_y:.2f}) '
            f'[sub_start_idx={sub_start_idx}, sub_goal_idx={sub_goal_idx}]'
        )
        return True

    # ----- LaserScan callback (dynamic obstacles + subpath replanning) -----
    def __scan_cbk(self, msg: LaserScan):
        if not self.map_ready or not self.path_pts:
            return

        # Robot position
        px = self.ttbot_pose.pose.position.x
        py = self.ttbot_pose.pose.position.y
        q = self.ttbot_pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        front_min = math.radians(-LASER_FRONT_DEG)
        front_max = math.radians(LASER_FRONT_DEG)

        for i, r in enumerate(msg.ranges):
            if r <= 0.0 or math.isinf(r) or math.isnan(r):
                continue
            if r > LASER_OBS_DIST:
                continue

            theta = msg.angle_min + i * msg.angle_increment
            if theta < front_min or theta > front_max:
                continue

            phi = yaw + theta

            # 0) LiDAR: check if this hit corresponds to a static wall
            wall_range = self.map_proc.raycast_to_static_obstacle(
                px, py, phi, LASER_OBS_DIST + 0.3
            )
            if wall_range is not None and abs(wall_range - r) <= STATIC_HIT_TOL:
                # Nearly same distance as a map wall -> not a dynamic obstacle
                continue

            # 1) Laser hit point in world coordinates
            wx = px + r * math.cos(phi)
            wy = py + r * math.sin(phi)

            # 1-1) Ignore if too close to a static wall (handled by cost-based wall avoidance)
            d_static = self.map_proc.obstacle_dist_m_world(wx, wy)
            if d_static < DYN_IGNORE_NEAR_STATIC:
                continue

            gx, gy = self.map_proc.pose_to_grid(wx, wy)
            if not self.map_proc.in_bounds(gx, gy):
                continue
            idx = gy * self.map_proc.width + gx

            # If the static grid already marks this as occupied (map mismatch etc.), skip
            if self.map_proc.grid is not None and self.map_proc.grid[idx] > 0:
                continue

            # If already marked in the dynamic layer, skip
            if self.map_proc.dynamic_occ is not None and self.map_proc.dynamic_occ[idx] != 0:
                continue

            # 2) Check whether the hit lies on the path (only check forward segment)
            on_path = False
            for (pxi, pyi) in self.path_pts[self.target_idx:]:
                if math.hypot(pxi - wx, pyi - wy) < PATH_OBS_HIT_DIST:
                    on_path = True
                    break
            if not on_path:
                continue

            # 3) Add to dynamic obstacle layer + update grid/inflation
            self.map_proc.add_dynamic_obstacle_world(wx, wy, DYN_OBS_INFLATION_RAD)

            # Republish inflated map including dynamic obstacles (for RViz debugging)
            inf_msg = self.map_proc.to_inflated_occgrid(self.get_clock().now().to_msg())
            self.inflated_map_pub.publish(inf_msg)

            # 4) Try A* subpath replanning
            success = self._replan_subpath_around_obstacle(wx, wy)
            if not success:
                # If it fails, fall back to full replanning (full A* computed next loop)
                self.last_goal_xy = None
                self.get_logger().warn(
                    'A* subpath replanning failed, will trigger full A* replan on next loop.'
                )

            break   # Process only once and exit the callback

    # ----- A* Global planner -----
    def a_star_path_planner(self, start_pose: PoseStamped, end_pose: PoseStamped) -> Path:
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(
                start_pose.pose.position, end_pose.pose.position
            )
        )
        # Do not edit this line (required for autograder)
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # if no goal or no map ==> empty path
        if not (self.has_goal and self.map_ready):
            astar_time = Float32()
            astar_time.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(astar_time)
            return path_msg

        # reuse path if goal not changed
        gx_now = end_pose.pose.position.x
        gy_now = end_pose.pose.position.y
        if self.path_pts and self.last_goal_xy == (gx_now, gy_now):
            astar_time = Float32()
            astar_time.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(astar_time)
            return self.path

        # world → grid
        s = start_pose.pose.position
        g = end_pose.pose.position
        sg = self.map_proc.pose_to_grid(s.x, s.y)
        gg = self.map_proc.pose_to_grid(g.x, g.y)

        grid_path = self.planner.plan(sg, gg)

        # astar_time publish
        astar_time = Float32()
        astar_time.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
        self.calc_time_pub.publish(astar_time)

        # if fail ==> empty path
        if grid_path is None or len(grid_path) < 2:
            self.path_pts = []
            self.path = path_msg
            self.last_goal_xy = None
            return self.path

        # grid path -> world coordinates
        self.path_pts = []
        for (gx, gy) in grid_path:
            x, y = self.map_proc.grid_to_pose(gx, gy)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
            self.path_pts.append((x, y))

        self.target_idx = 0
        self.path = path_msg
        self.path_pub.publish(self.path)
        self.last_goal_xy = (gx_now, gy_now)
        return self.path

    # ----- Path index chooser -----
    def get_path_idx(self, path: Path, vehicle_pose: PoseStamped) -> int:
        if not self.path_pts:
            self.target_idx = 0
            return 0

        px = vehicle_pose.pose.position.x
        py = vehicle_pose.pose.position.y

        # 1) if target is close enough, move to the next one
        while self.target_idx < len(self.path_pts):
            tx, ty = self.path_pts[self.target_idx]
            if math.hypot(tx - px, ty - py) < waypoint_tol_m:
                self.target_idx += 1
            else:
                break

        # 2) select the next point with lookahead
        def nearest_index(x: float, y: float) -> int:
            best_i = 0
            best_d = 1e9
            for i, (pxi, pyi) in enumerate(self.path_pts):
                d = (pxi - x) ** 2 + (pyi - y) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        if self.path_pts:
            base_i = max(nearest_index(px, py), self.target_idx)
            li = base_i
            for i in range(base_i, len(self.path_pts)):
                if math.hypot(self.path_pts[i][0] - px,
                              self.path_pts[i][1] - py) >= lookahead_m:
                    li = i
                    break
            self.target_idx = max(self.target_idx, li)

        if self.path_pts:
            self.target_idx = clamp(self.target_idx, 0, len(self.path_pts) - 1)
        else:
            self.target_idx = 0

        return int(self.target_idx)

    # ----- Path follower -----
    def path_follower(self, vehicle_pose: PoseStamped,
                      current_goal_pose: PoseStamped) -> Tuple[float, float]:
        if not self.path_pts:
            return 0.0, 0.0

        px = vehicle_pose.pose.position.x
        py = vehicle_pose.pose.position.y
        q = vehicle_pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        tx = current_goal_pose.pose.position.x
        ty = current_goal_pose.pose.position.y

        dx = tx - px
        dy = ty - py
        dist = math.hypot(dx, dy)
        des_yaw = math.atan2(dy, dx)
        yaw_err = wrap_angle(des_yaw - yaw)

        # PID on yaw
        self.ang_err_int += yaw_err / self.loop_hz
        ang_der = (yaw_err - self.prev_yaw_err) * self.loop_hz
        w_cmd = kp_yaw * yaw_err + ki_yaw * self.ang_err_int + kd_yaw * ang_der
        self.prev_yaw_err = yaw_err
        w_cmd = clamp(w_cmd, -max_ang_vel, max_ang_vel)

        if abs(yaw_err) > face_goal_rad:
            v_cmd = 0.0
        else:
            v_cmd = kp_lin * dist
            v_cmd = clamp(v_cmd, 0.0, max_lin_vel)

        return float(v_cmd), float(w_cmd)

    def move_ttbot(self, speed: float, heading: float):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(speed)
        cmd_vel.angular.z = float(heading)
        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        period = 1.0 / self.loop_hz
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)

            # 1. Create/update path
            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)

            # 2. Follow the path
            idx = self.get_path_idx(path, self.ttbot_pose)
            if 0 <= idx < len(path.poses):
                current_goal = path.poses[idx]
                speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            else:
                self.move_ttbot(0.0, 0.0)
                time.sleep(period)
                continue

            # final goal full stop
            if self.path_pts:
                gx, gy = self.path_pts[-1]
                if math.hypot(
                    gx - self.ttbot_pose.pose.position.x,
                    gy - self.ttbot_pose.pose.position.y
                ) < goal_tol_m:
                    speed, heading = 0.0, 0.0

            self.move_ttbot(speed, heading)
            time.sleep(period)


def main(args=None):
    rclpy.init(args=args)
    task2 = Task2()
    try:
        task2.run()
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
