#!/usr/bin/env python3

import math
import time
from collections import deque
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path, OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

USE_COST_ASTAR = True


#============ Parameters ===============


# Wall buffer
WALL_BUFFER_BIG   = 0.18  # for Plan A,B
WALL_BUFFER_SMALL = 0.15  # for Plan C (narrow hallway)

# Minimal clearance to wall 
WALL_CLEARANCE_DESIRED = 0.5   # desired clearance to wall

allow_diag      = True     # Allow diagonal moves
lookahead_m     = 0.45
waypoint_tol_m  = 0.22

# Velocity limits
max_lin_vel = 0.45
max_ang_vel = 1.2

# Controller gains / thresholds
kp_lin   = 1.3
kp_yaw   = 1.6
ki_yaw   = 0.0
kd_yaw   = 0.08

face_goal_rad   = 0.7
goal_tol_m      = 0.3

# Plan A/B/C - search radius
SEARCH_R_MIN_BASE    = 1.0    # minial radius for search
SEARCH_R_MAX_PLAN_A  = 4.0    # Plan A
SEARCH_R_MAX_PLAN_B  = 20.0   # Plan B
SEARCH_R_MAX_PLAN_C  = 4.0    # Plan C


# Cost parameters 
PATH_UNKNOWN_COST         = 20   # unknown cells penalty in path planning (Plan A,B)
PATH_UNKNOWN_COST_PLAN_C  = 0    # unknown cells penalty in path planning (Plan C)
PATH_OBSTACLE_COST_GAIN   = 100  # additional cost when path close to wall 
PATH_OBSTACLE_SAFE_DIST   = 0.6  # this distance from wall starts path cost increase [m]

# SLAM / mapping update thresholds
MAP_TRANS_UPDATE = 0.5  # [m] move at least this far
MAP_ROT_UPDATE   = 0.5  # [rad] or rotate at least this much


#============ utilities ===============


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_to_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

#============ Map Processor ===============

class MapProcessor:
    def __init__(self):
        self.grid: Optional[List[int]] = None
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.inflated: Optional[List[int]] = None
        self.obs_dist_cells: Optional[List[float]] = None  # distance to nearest obstacle [cells]
        self.ready = False

    def update_from_msg(self, msg: OccupancyGrid, inflation_radius_m: float = WALL_BUFFER_BIG):
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.grid = list(msg.data)  # [-1, 0..100]
        self.inflated = self._inflate_grid(self.grid, self.width, self.height, inflation_radius_m)
        self.ready = True

    def _inflate_grid(self, data: List[int], w: int, h: int, inflation_radius_m: float) -> List[int]:
        occ = [1 if (v > 0) else 0 for v in data]
        q = deque()
        dist = [math.inf] * (w * h)
        for idx, v in enumerate(occ):
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

        cells = int(math.ceil(inflation_radius_m / max(self.resolution, 1e-6)))
        inflated = [0] * (w * h)
        for i, d in enumerate(dist):
            if d <= cells:
                inflated[i] = 1
        return inflated

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
        return self.inflated[gy * self.width + gx] == 0

    def grid_clearance_m(self, gx: int, gy: int) -> float:
        if not self.in_bounds(gx, gy) or self.obs_dist_cells is None:
            return 0.0
        d_cells = self.obs_dist_cells[gy * self.width + gx]
        if d_cells == math.inf:
            return 1e9
        return d_cells * self.resolution

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

    def opposite_unknown_from_free(self, gx: int, gy: int) -> Optional[Tuple[int, int]]:
        if self.obs_dist_cells is None or self.grid is None:
            return None
        if not self.in_bounds(gx, gy):
            return None

        idx_center = gy * self.width + gx
        d_center = self.obs_dist_cells[idx_center]
        if d_center == math.inf or d_center <= 0:
            return None

        dirs8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
        best_dc = d_center
        toward = None
        for dx, dy in dirs8:
            ngx = gx + dx
            ngy = gy + dy
            if not self.in_bounds(ngx, ngy):
                continue
            nidx = ngy * self.width + ngx
            dc = self.obs_dist_cells[nidx]
            if dc < best_dc:
                best_dc = dc
                toward = (dx, dy)

        if toward is None:
            return None

        away_dx, away_dy = -toward[0], -toward[1]
        steps = int(round(d_center))
        if steps <= 0:
            steps = 1

        for k in range(steps, steps + 3):
            ngx = gx + away_dx * k
            ngy = gy + away_dy * k
            if not self.in_bounds(ngx, ngy):
                break
            nidx = ngy * self.width + ngx
            if self.grid[nidx] == -1:
                if self.inflated is not None and self.inflated[nidx] == 1:
                    continue
                return ngx, ngy

        return None


#============ A* Planner ===============

class AStarPlanner:
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

        self.cur_use_cost: bool = USE_COST_ASTAR
        self.cur_unknown_cost: float = PATH_UNKNOWN_COST

    def _cell_cost(self, idx: int) -> float:
        if not self.cur_use_cost:
            return 0.0
        if self.mp.grid is None or self.mp.obs_dist_cells is None:
            return 0.0

        gval = self.mp.grid[idx]
        dist_cells = self.mp.obs_dist_cells[idx]
        dist_m = dist_cells * self.mp.resolution if dist_cells < math.inf else 1e9

        cost = 0.0
        if gval == -1:
            cost += float(self.cur_unknown_cost)

        if dist_m < PATH_OBSTACLE_SAFE_DIST:
            cost += PATH_OBSTACLE_COST_GAIN * (PATH_OBSTACLE_SAFE_DIST - dist_m) / PATH_OBSTACLE_SAFE_DIST

        return cost

    def h(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        use_cost: bool = USE_COST_ASTAR,
        unknown_cost: float = PATH_UNKNOWN_COST,
    ) -> Optional[List[Tuple[float, float]]]:
        if not self.mp.ready:
            return None

        self.cur_use_cost = use_cost
        self.cur_unknown_cost = float(unknown_cost)

        start_g = self.mp.pose_to_grid(*start_xy)
        goal_g  = self.mp.pose_to_grid(*goal_xy)

        if not self.mp.in_bounds(*start_g) or not self.mp.in_bounds(*goal_g):
            return None
        if (not self.mp.is_free(*start_g)) or (not self.mp.is_free(*goal_g)):
            return None

        w, h = self.mp.width, self.mp.height
        import heapq

        gscore = [math.inf] * (w * h)
        came   = [-1] * (w * h)

        def idx(p: Tuple[int, int]) -> int:
            return p[1] * w + p[0]

        start_i = idx(start_g)
        goal_i  = idx(goal_g)
        gscore[start_i] = 0.0

        open_q = []
        heapq.heappush(open_q, (self.h(start_g, goal_g), 0.0, start_g))

        while open_q:
            f, g, cur = heapq.heappop(open_q)
            cur_i = idx(cur)

            if cur_i == goal_i:
                path_g: List[Tuple[int, int]] = []
                ci = cur_i
                while ci != -1:
                    x = ci % w
                    y = ci // w
                    path_g.append((x, y))
                    ci = came[ci]
                path_g.reverse()

                path_xy: List[Tuple[float, float]] = []
                for gx, gy in path_g:
                    x, y = self.mp.grid_to_pose(gx, gy)
                    path_xy.append((x, y))
                if len(path_xy) < 2:
                    return None
                return path_xy

            cx, cy = cur
            for k, (dx, dy) in enumerate(self.neighbors):
                nx, ny = cx + dx, cy + dy
                if not self.mp.in_bounds(nx, ny):
                    continue
                if not self.mp.is_free(nx, ny):
                    continue
                ni = ny * w + nx

                base_step = self.step_cost[k]
                extra     = self._cell_cost(ni)
                step      = base_step + extra

                tentative_g = gscore[cur_i] + step
                if tentative_g < gscore[ni]:
                    gscore[ni] = tentative_g
                    came[ni]   = cur_i
                    nb = (nx, ny)
                    heapq.heappush(open_q, (tentative_g + self.h(nb, goal_g), tentative_g, nb))

        return None

#============ Task1 Node ===============

class Task1(Node):
    def __init__(self):
        super().__init__('task1_node')

        self.map_proc = MapProcessor()
        self.astar_planner = AStarPlanner(self.map_proc, use_8_conn=allow_diag)

        self.current_buffer = WALL_BUFFER_BIG

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)

        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self.robot_yaw: Optional[float] = None

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_cov_cb, 10)
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_cov_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.goal_x: Optional[float] = None
        self.goal_y: Optional[float] = None
        self.have_goal: bool = False

        self.goal_sub1 = self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.goal_sub2 = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, 10)

        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/task1_astar_path', 10)

        inflated_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, 'inflated_map', inflated_qos)

        self.path_pts: List[Tuple[float, float]] = []
        self.target_idx = 0
        self.last_goal_xy: Optional[Tuple[float, float]] = None

        self.prev_yaw_err = 0.0
        self.ang_err_int  = 0.0

        self.last_update_x: Optional[float] = None
        self.last_update_y: Optional[float] = None
        self.last_update_yaw: Optional[float] = None

        self.loop_hz = 5.0
        self.timer = self.create_timer(1.0 / self.loop_hz, self.timer_cb)
        self.get_logger().info("Task1 node started (logs trimmed).")

# ============ Inflation Update for debugging ===============
    def update_inflation(self):
        if self.map_proc.grid is None:
            return
        self.map_proc.inflated = self.map_proc._inflate_grid(
            self.map_proc.grid,
            self.map_proc.width,
            self.map_proc.height,
            self.current_buffer,
        )
        inf_msg = self.map_proc.to_inflated_occgrid(self.get_clock().now().to_msg())
        self.inflated_map_pub.publish(inf_msg)

    def map_cb(self, msg: OccupancyGrid):
        self.map_proc.update_from_msg(msg, inflation_radius_m=self.current_buffer)
        self.path_pts = []
        self.last_goal_xy = None
        inf_msg = self.map_proc.to_inflated_occgrid(self.get_clock().now().to_msg())
        self.inflated_map_pub.publish(inf_msg)

    def pose_cov_cb(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

    def goal_cb(self, msg: PoseStamped):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.have_goal = True
        self.path_pts = []
        self.last_goal_xy = None
        self.get_logger().info(f"[GOAL-MANUAL] x={self.goal_x:.2f}, y={self.goal_y:.2f}")

    # =======================
    # Mapping update helpers
    # =======================
    def _moved_enough_for_new_update(self) -> bool:
        if (self.last_update_x is None or
                self.last_update_y is None or
                self.last_update_yaw is None):
            return True

        dx = self.robot_x - self.last_update_x
        dy = self.robot_y - self.last_update_y
        dist = math.hypot(dx, dy)
        d_yaw = abs(wrap_to_pi(self.robot_yaw - self.last_update_yaw))
        return (dist > MAP_TRANS_UPDATE) or (d_yaw > MAP_ROT_UPDATE)

    def _mark_update_pose(self):
        self.last_update_x = self.robot_x
        self.last_update_y = self.robot_y
        self.last_update_yaw = self.robot_yaw

    # =======================
    # reachable mask
    # =======================
    def compute_reachable_mask(self) -> Optional[List[bool]]:
        if self.map_proc.grid is None or self.map_proc.inflated is None:
            return None
        if self.robot_x is None or self.robot_y is None:
            return None

        w, h = self.map_proc.width, self.map_proc.height
        inflated = self.map_proc.inflated

        rgx, rgy = self.map_proc.pose_to_grid(self.robot_x, self.robot_y)
        if not self.map_proc.in_bounds(rgx, rgy):
            return None

        start_idx = rgy * w + rgx
        reachable = [False] * (w * h)
        q = deque()
        reachable[start_idx] = True
        q.append((rgx, rgy))

        nbrs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            x, y = q.popleft()
            for dx, dy in nbrs4:
                nx, ny = x + dx, y + dy
                if not self.map_proc.in_bounds(nx, ny):
                    continue
                nidx = ny * w + nx
                if reachable[nidx]:
                    continue
                if inflated[nidx] == 1:
                    continue
                reachable[nidx] = True
                q.append((nx, ny))

        return reachable

    # =======================
    # Frontier based goal selection
    # =======================
    def find_frontier_goal_in_radius(
        self,
        reachable: List[bool],
        r_min: float,
        r_max: float,
    ) -> Optional[Tuple[float, float, int]]:
        grid = self.map_proc.grid
        w, h = self.map_proc.width, self.map_proc.height
        if grid is None:
            return None

        min_clear_m = max(self.current_buffer, WALL_CLEARANCE_DESIRED)

        n_bins = 360
        counts = [0] * n_bins
        sum_x = [0.0] * n_bins
        sum_y = [0.0] * n_bins

        low_clear_candidates = []
        nbrs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for gy in range(h):
            for gx in range(w):
                idx = gy * w + gx

                if not reachable[idx]:
                    continue
                if grid[idx] != 0:
                    continue

                has_unknown_nbr = False
                for dx, dy in nbrs4:
                    ngx = gx + dx
                    ngy = gy + dy
                    if not self.map_proc.in_bounds(ngx, ngy):
                        continue
                    nidx = ngy * self.map_proc.width + ngx
                    if grid[nidx] == -1:
                        has_unknown_nbr = True
                        break
                if not has_unknown_nbr:
                    continue

                x, y = self.map_proc.grid_to_pose(gx, gy)
                dxw = x - self.robot_x
                dyw = y - self.robot_y
                r  = math.hypot(dxw, dyw)
                if r < r_min or r > r_max:
                    continue

                clear_m = self.map_proc.grid_clearance_m(gx, gy)
                if clear_m >= min_clear_m:
                    ang = math.atan2(dyw, dxw)
                    bin_idx = int((ang + math.pi) / (2.0 * math.pi) * n_bins)
                    bin_idx = max(0, min(n_bins - 1, bin_idx))

                    counts[bin_idx] += 1
                    sum_x[bin_idx]  += x
                    sum_y[bin_idx]  += y
                else:
                    low_clear_candidates.append((gx, gy))

        best_bin = max(range(n_bins), key=lambda i: counts[i])
        if counts[best_bin] > 0:
            goal_x = sum_x[best_bin] / counts[best_bin]
            goal_y = sum_y[best_bin] / counts[best_bin]
            return goal_x, goal_y, counts[best_bin]

        if not low_clear_candidates:
            return None

        counts_fb = [0] * n_bins
        sum_x_fb = [0.0] * n_bins
        sum_y_fb = [0.0] * n_bins

        for gx, gy in low_clear_candidates:
            opp = self.map_proc.opposite_unknown_from_free(gx, gy)
            if opp is None:
                continue
            ugx, ugy = opp

            ux, uy = self.map_proc.grid_to_pose(ugx, ugy)
            dxw = ux - self.robot_x
            dyw = uy - self.robot_y
            r = math.hypot(dxw, dyw)
            if r < r_min or r > r_max:
                continue

            ang = math.atan2(dyw, dxw)
            bin_idx = int((ang + math.pi) / (2.0 * math.pi) * n_bins)
            bin_idx = max(0, min(n_bins - 1, bin_idx))

            counts_fb[bin_idx] += 1
            sum_x_fb[bin_idx]  += ux
            sum_y_fb[bin_idx]  += uy

        best_bin_fb = max(range(n_bins), key=lambda i: counts_fb[i])
        if counts_fb[best_bin_fb] == 0:
            return None

        goal_x = sum_x_fb[best_bin_fb] / counts_fb[best_bin_fb]
        goal_y = sum_y_fb[best_bin_fb] / counts_fb[best_bin_fb]
        return goal_x, goal_y, counts_fb[best_bin_fb]

    def map_has_any_frontier(self) -> bool:
        grid = self.map_proc.grid
        if grid is None:
            return False

        w, h = self.map_proc.width, self.map_proc.height
        nbrs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for gy in range(h):
            for gx in range(w):
                idx = gy * w + gx
                if grid[idx] != 0:
                    continue
                for dx, dy in nbrs4:
                    ngx, ngy = gx + dx, gy + dy
                    if not self.map_proc.in_bounds(ngx, ngy):
                        continue
                    nidx = ngy * w + ngx
                    if grid[nidx] == -1:
                        return True
        return False

    # =======================
    # Plan A/B
    # =======================
    def choose_unknown_goal_big(self) -> bool:
        if (not self.map_proc.ready) or (self.map_proc.grid is None):
            return False
        if self.robot_x is None or self.robot_y is None:
            return False

        self.current_buffer = WALL_BUFFER_BIG
        self.update_inflation()
        reachable = self.compute_reachable_mask()
        if reachable is None:
            return False

        result = self.find_frontier_goal_in_radius(reachable, SEARCH_R_MIN_BASE, SEARCH_R_MAX_PLAN_A)
        if result is not None:
            goal_x, goal_y, cnt = result
            self.goal_x, self.goal_y = goal_x, goal_y
            self.have_goal = True
            self.path_pts = []
            self.last_goal_xy = None
            self._mark_update_pose()
            self.get_logger().info(f"[GOAL-AUTO] PlanA x={goal_x:.2f}, y={goal_y:.2f}, cnt={cnt}")
            return True

        result = self.find_frontier_goal_in_radius(reachable, SEARCH_R_MIN_BASE, SEARCH_R_MAX_PLAN_B)
        if result is not None:
            goal_x, goal_y, cnt = result
            self.goal_x, self.goal_y = goal_x, goal_y
            self.have_goal = True
            self.path_pts = []
            self.last_goal_xy = None
            self._mark_update_pose()
            self.get_logger().info(f"[GOAL-AUTO] PlanB x={goal_x:.2f}, y={goal_y:.2f}, cnt={cnt}")
            return True

        return False

    # =======================
    # Plan C
    # =======================
    def choose_unknown_goal_small(self) -> bool:
        if (not self.map_proc.ready) or (self.map_proc.grid is None):
            return False
        if self.robot_x is None or self.robot_y is None:
            return False

        self.current_buffer = WALL_BUFFER_SMALL
        self.update_inflation()
        reachable = self.compute_reachable_mask()
        if reachable is None:
            return False

        result = self.find_frontier_goal_in_radius(reachable, SEARCH_R_MIN_BASE, SEARCH_R_MAX_PLAN_C)
        if result is not None:
            goal_x, goal_y, cnt = result
            self.goal_x, self.goal_y = goal_x, goal_y
            self.have_goal = True
            self.path_pts = []
            self.last_goal_xy = None
            self._mark_update_pose()
            self.get_logger().info(f"[GOAL-AUTO] PlanC x={goal_x:.2f}, y={goal_y:.2f}, cnt={cnt}")
            return True

        return False

    # =======================
    # Path trim by clearance
    # =======================
    def trim_path_by_clearance(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if self.map_proc.grid is None or self.map_proc.obs_dist_cells is None:
            return path
        if len(path) < 2:
            return path

        desired_clear = max(self.current_buffer, WALL_CLEARANCE_DESIRED)

        clear_list: List[float] = []
        best_clear = -1.0
        best_idx = 0

        for i, (x, y) in enumerate(path):
            gx, gy = self.map_proc.pose_to_grid(x, y)
            c = self.map_proc.grid_clearance_m(gx, gy)
            clear_list.append(c)
            if c > best_clear:
                best_clear = c
                best_idx = i

        safe_last_idx = None
        for i in range(len(path) - 1, 0, -1):
            if clear_list[i] >= desired_clear:
                safe_last_idx = i
                break

        if safe_last_idx is not None and safe_last_idx >= 1:
            trimmed = path[:safe_last_idx + 1]
            self.goal_x, self.goal_y = trimmed[-1]
            return trimmed

        if best_idx > 0:
            trimmed = path[:best_idx + 1]
            self.goal_x, self.goal_y = trimmed[-1]
            return trimmed

        return path

    # =======================
    # A* path planning
    # =======================
    def plan_astar_path_to_goal(self) -> bool:
        if not self.map_proc.ready:
            return False
        if self.robot_x is None or self.robot_y is None:
            return False
        if self.goal_x is None or self.goal_y is None:
            return False

        if self.path_pts and self.last_goal_xy == (self.goal_x, self.goal_y):
            return True

        start_xy = (self.robot_x, self.robot_y)
        goal_xy  = (self.goal_x, self.goal_y)

        if self.map_proc.grid is None:
            return False

        t0 = time.time()

        buf = self.current_buffer
        self.map_proc.inflated = self.map_proc._inflate_grid(
            self.map_proc.grid,
            self.map_proc.width,
            self.map_proc.height,
            buf,
        )

        unknown_cost = PATH_UNKNOWN_COST_PLAN_C if self.current_buffer == WALL_BUFFER_SMALL else PATH_UNKNOWN_COST

        path = self.astar_planner.plan(
            start_xy,
            goal_xy,
            use_cost=True,
            unknown_cost=unknown_cost,
        )

        dt = time.time() - t0

        if path is None or len(path) < 2: # fail log for debugging
            self.get_logger().warn(
                f"[A*] No path (buf={buf:.2f}, unknown_cost={unknown_cost}, t={dt:.3f}s).",
                throttle_duration_sec=2.0,
            )
            self.path_pts = []
            self.last_goal_xy = None
            return False

        path = self.trim_path_by_clearance(path)

        self.path_pts = path
        self.target_idx = 0
        self.prev_yaw_err = 0.0
        self.ang_err_int  = 0.0
        self.last_goal_xy = (self.goal_x, self.goal_y)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for (x, y) in path:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

        self.get_logger().info(f"[A*] Path OK (N={len(path)}, buf={buf:.2f}, unknown_cost={unknown_cost}, t={dt:.3f}s).")
        return True

    # =======================
    # Path index chooser
    # =======================
    def select_target_point(self) -> Optional[Tuple[float, float]]:
        if not self.path_pts:
            self.target_idx = 0
            return None

        px = self.robot_x
        py = self.robot_y

        while self.target_idx < len(self.path_pts):
            tx, ty = self.path_pts[self.target_idx]
            if math.hypot(tx - px, ty - py) < waypoint_tol_m:
                self.target_idx += 1
            else:
                break

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
                if math.hypot(self.path_pts[i][0] - px, self.path_pts[i][1] - py) >= lookahead_m:
                    li = i
                    break
            self.target_idx = max(self.target_idx, li)

        if self.path_pts:
            self.target_idx = int(clamp(self.target_idx, 0, len(self.path_pts) - 1))
            return self.path_pts[self.target_idx]

        self.target_idx = 0
        return None

    # =======================
    # Path follower
    # =======================
    def compute_cmd_to_target(self, tx: float, ty: float) -> Tuple[float, float]:
        px = self.robot_x
        py = self.robot_y
        yaw = self.robot_yaw

        dx = tx - px
        dy = ty - py
        dist    = math.hypot(dx, dy)
        des_yaw = math.atan2(dy, dx)
        yaw_err = wrap_to_pi(des_yaw - yaw)

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

    # =======================
    # Timer loop
    # =======================
    def timer_cb(self):
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
            self.publish_cmd(0.0, 0.0)
            return
        if not self.map_proc.ready:
            self.publish_cmd(0.0, 0.0)
            return

        # 1) if path is true follow 
        if self.path_pts:
            gx, gy = self.path_pts[-1]
            dist_goal = math.hypot(gx - self.robot_x, gy - self.robot_y)
            if dist_goal < goal_tol_m:
                self.get_logger().info(f"[GOAL] Reached (dist={dist_goal:.2f}).")
                self.path_pts = []
                self.have_goal = False
                self.goal_x = None
                self.goal_y = None
                self.publish_cmd(0.0, 0.0)

                if self.current_buffer == WALL_BUFFER_SMALL:
                    self.current_buffer = WALL_BUFFER_BIG
                    self.update_inflation()
                return

            target = self.select_target_point()
            if target is None:
                self.publish_cmd(0.0, 0.0)
                return

            tx, ty = target
            v_cmd, w_cmd = self.compute_cmd_to_target(tx, ty)
            self.publish_cmd(v_cmd, w_cmd)
            return

        # 2) if path is not ready, choose goal
        if not self.have_goal or self.goal_x is None or self.goal_y is None:
            if not self._moved_enough_for_new_update():
                # rotate but no log
                self.publish_cmd(0.0, 0.4)
                return

            if self.current_buffer == WALL_BUFFER_SMALL:
                ok = self.choose_unknown_goal_small()
                if not ok:
                    self.publish_cmd(0.0, 0.0)
                    return
            else:
                ok = self.choose_unknown_goal_big()
                if not ok:
                    # changing to plan C
                    self.get_logger().info("[PLAN] Switch to Plan C.")
                    self.current_buffer = WALL_BUFFER_SMALL
                    self.update_inflation()
                    ok2 = self.choose_unknown_goal_small()
                    if not ok2:
                        self.publish_cmd(0.0, 0.0)
                        return

        if not self.plan_astar_path_to_goal():
            self.have_goal = False
            self.goal_x = None
            self.goal_y = None
            self.path_pts = []
            self.last_goal_xy = None
            self.publish_cmd(0.0, 0.0)
            return

        self.publish_cmd(0.0, 0.0)

    def publish_cmd(self, v: float, w: float):
        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.publish_cmd(0.0, 0.0)
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

