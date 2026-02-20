#!/usr/bin/env python3
import math
from collections import deque
from heapq import heappush, heappop
from typing import List, Tuple, Optional, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Point
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan, Image, CameraInfo

import cv2
import numpy as np
from cv_bridge import CvBridge


# =========================
# Parameters (NAVIGATION)
# =========================
wall_buffer_m   = 0.15
allow_diag      = True
lookahead_m     = 0.45
waypoint_tol_m  = 0.22

# Velocity limits
max_lin_vel_normal  = 0.30  # normal velocity
max_lin_vel_subpath = 0.20  # subpath velocity
max_ang_vel         = 1.2

# Controller gains / thresholds
kp_lin   = 1.3
kp_yaw   = 1.6
ki_yaw   = 0.0
kd_yaw   = 0.08

face_goal_rad   = 0.7

# ===== Cost-based A* (stay away from walls) =====
USE_COST_ASTAR     = True
UNKNOWN_COST       = 0.0
OBSTACLE_COST_GAIN = 10.0
OBSTACLE_SAFE_DIST = 0.65

# ===== LaserScan-based dynamic obstacles =====
LASER_FRONT_DEG       = 45.0
LASER_OBS_DIST        = 1.0
PATH_OBS_HIT_DIST     = 0.3
DYN_OBS_INFLATION_RAD = 0.15

# Static wall filter threshold:
# In raycasting, if the difference between the map wall distance and the LaserScan distance
# is less than or equal to this value, it is considered a static wall [m].
STATIC_HIT_TOL = 0.001

# Dynamic obstacle candidates within this distance of a static wall are ignored
# (since cost-based wall avoidance already handles that area).
DYN_IGNORE_NEAR_STATIC = max(OBSTACLE_SAFE_DIST - 0.05, 0.0)


# =========================
# Parameters (WAYPOINTS)
# =========================
WAYPOINTS: List[Tuple[float, float]] = [
    (6.945185661315918,   1.7837848663330078),
    (3.251347541809082,   1.9593312740325928),
    (8.167699813842773,  -3.8135714530944824),
    (-0.5765336751937866, 1.6513290405273438),
    (-3.9653501510620117, 2.0113327503204346),
    (-4.300878524780273, -3.909541368484497),
]
WP_REACHED_TOL_M = 0.25

SPIN_ANG_VEL = 1.0
SPIN_TARGET_RAD = 2.0 * math.pi


# =========================
# Parameters (BALL)
# =========================
BALL_TRIGGER_DIST_M = 0.50  # if Z <= this -> switch to servo
BALL_FINAL_DIST_M   = 0.25  # final stop distance

BALL_DIST_BAND_M = 0.03
BALL_ALIGN_BAND  = 0.08
BALL_ALIGN_STABLE_FRAMES = 6

BALL_ANG_KP   = 1.2
BALL_ANG_KI   = 0.0
BALL_ANG_KD   = 0.20
BALL_ANG_VMAX = 2.2

BALL_LIN_KP   = 1.4
BALL_LIN_KI   = 0.0
BALL_LIN_KD   = 0.10
BALL_LIN_VMAX = 0.25
BALL_BACK_VMAX = 0.25

BALL_MIN_ABS_LIN = 0.05
BALL_MIN_ABS_ANG = 0.01
BALL_SEARCHING_ANG = 1.0

BALL_DET_TIMEOUT_S = 0.6
BALL_LOST_ABORT_S  = 1.5

BALL_SPIN_SEARCH_RADIUS_M = 2.0


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


class PID:
    # Simple PID with basic anti-windup
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, umin=-1e9, umax=1e9):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.umin, self.umax = float(umin), float(umax)
        self.i = 0.0
        self.e_prev = None

    def reset(self):
        self.i = 0.0
        self.e_prev = None

    def step(self, e, dt):
        dt = max(float(dt), 1e-6)
        de = 0.0 if self.e_prev is None else (e - self.e_prev) / dt

        # anti-windup: only integrate if the tentative command is not saturated
        u_try = self.kp * e + self.ki * (self.i + e * dt) + self.kd * de
        if self.umin < u_try < self.umax:
            self.i += e * dt

        self.e_prev = e
        u = self.kp * e + self.ki * self.i + self.kd * de
        return max(self.umin, min(self.umax, u))


# =========================
# Map Processor
# =========================
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

        # 4-connected BFS distance transform (in cells)
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
                    if self.grid[idx] <= 0:
                        self.grid[idx] = 100

        # Recompute distance map + inflation.
        self.inflated = self._inflate_grid(self.grid, self.width, self.height, wall_buffer_m)

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
    def raycast_to_static_obstacle(self, x0: float, y0: float, phi: float, max_range: float) -> Optional[float]:
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


# =========================
# A* Algorithm (with cost map)
# =========================
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
                extra     = self._cell_cost(ni)
                tentative = gscore[base_i] + base_step + extra

                if tentative < gscore[ni]:
                    gscore[ni] = tentative
                    came[ni] = base_i
                    heappush(pq, (tentative + self.h((nx, ny), goal_g), tentative, (nx, ny)))
        return None


# =========================
# Task3 Node
# =========================
class Task3(Node):
    def __init__(self):
        super().__init__('task3_node')

        # -------------------------
        # States
        # -------------------------
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.ttbot_pose.pose.orientation.w = 1.0

        self.has_goal = False
        self.map_ready = False
        self.have_pose = False
        self.camera_ready = False

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

        # subpath segment tracking (for speed limiting)
        self.in_subpath_segment: bool = False
        self.subpath_start_idx: Optional[int] = None
        self.subpath_end_idx: Optional[int] = None

        # waypoint progression
        self.wp_idx = 0

        # spin state
        self.spin_accum = 0.0
        self.spin_last_yaw: Optional[float] = None

        # final confirmed ball positions (map frame)
        self.ball_pos_map: Dict[str, Optional[Tuple[float, float]]] = {
            "Red": None, "Green": None, "Blue": None
        }

        # mode machine: NAV_WP -> SPIN -> BALL_NAV -> BALL_SERVO
        self.mode = "NAV_WP"
        self.ball_target: Optional[str] = None
        self.ball_stage = "ALIGN"
        self.ball_align_stable = 0

        # ball detection cache (vision)
        self.ball_det: Dict[str, Dict[str, Any]] = {
            "Red":   {"found": False, "t": 0.0, "bbox": (0,0,0,0), "area": 0.0,
                      "x_err": 0.0, "Z": None, "X_cam": None, "map_xy": None},
            "Green": {"found": False, "t": 0.0, "bbox": (0,0,0,0), "area": 0.0,
                      "x_err": 0.0, "Z": None, "X_cam": None, "map_xy": None},
            "Blue":  {"found": False, "t": 0.0, "bbox": (0,0,0,0), "area": 0.0,
                      "x_err": 0.0, "Z": None, "X_cam": None, "map_xy": None},
        }

        # samples near the final approach to stabilize map coordinate
        self.ball_close_samples: Dict[str, List[Tuple[float, float]]] = {
            "Red": [], "Green": [], "Blue": []
        }
        self.close_samples_needed = 8

        # servo PID
        self.ball_ang_pid = PID(BALL_ANG_KP, BALL_ANG_KI, BALL_ANG_KD, -BALL_ANG_VMAX, BALL_ANG_VMAX)
        self.ball_lin_pid = PID(BALL_LIN_KP, BALL_LIN_KI, BALL_LIN_KD, -BALL_BACK_VMAX, BALL_LIN_VMAX)
        self.ball_last_pid_t = self.get_clock().now()

        # -------------------------
        # Subscribers
        # -------------------------
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self.__map_cbk, map_qos)

        # LaserScan for dynamic obstacles
        self.create_subscription(LaserScan, '/scan', self.__scan_cbk, 10)

        # -------------------------
        # Publishers
        # -------------------------
        self.path_pub      = self.create_publisher(Path,   'global_plan', 10)
        self.cmd_vel_pub   = self.create_publisher(Twist,  'cmd_vel',     10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time', 10)

        inflated_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, 'inflated_map', inflated_qos)

        # -------------------------
        # Camera params + topics
        # -------------------------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')

        self.declare_parameter('hsv_S_low', 110)
        self.declare_parameter('hsv_V_low', 90)

        # NOTE: OpenCV HSV hue is usually [0..179] (integers). Set these params accordingly.
        self.declare_parameter('hsv_R_H_low1', 0)
        self.declare_parameter('hsv_R_H_high1', 0.1)  # you will likely override this via launch/params
        self.declare_parameter('hsv_G_H_low', 60)
        self.declare_parameter('hsv_G_H_high', 70)
        self.declare_parameter('hsv_B_H_low', 110)
        self.declare_parameter('hsv_B_H_high', 130)

        self.declare_parameter('ball_size', 0.12)
        self.declare_parameter('show_window', True)
        self.declare_parameter('visualize', 1.0)

        self.bridge = CvBridge()
        self.fx = None

        self.create_subscription(Image, self.p('image_topic'), self.on_image, 10)
        self.create_subscription(CameraInfo, self.p('camera_info_topic'), self.on_cinfo, 10)

        # Publish only these topics: /red_pos /green_pos /blue_pos
        ball_pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_red_pos   = self.create_publisher(Point, '/red_pos',   ball_pub_qos)
        self.pub_green_pos = self.create_publisher(Point, '/green_pos', ball_pub_qos)
        self.pub_blue_pos  = self.create_publisher(Point, '/blue_pos',  ball_pub_qos)

        # main timer loop
        self.loop_hz = 10.0
        self.create_timer(1.0 / self.loop_hz, self.control_loop)

        if bool(self.p('show_window')):
            cv2.namedWindow('CB-frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('CB-frame', 900, 700)

        self.get_logger().info("Task3 running (pos-only publishers).")

    def p(self, name):
        return self.get_parameter(name).value

    def publish_point(self, pub, x, y):
        pub.publish(Point(x=float(x), y=float(y), z=0.0))

    def all_balls_found(self) -> bool:
        return (self.ball_pos_map["Red"] is not None and
                self.ball_pos_map["Green"] is not None and
                self.ball_pos_map["Blue"] is not None)

    def current_yaw(self) -> Optional[float]:
        if not self.have_pose:
            return None
        q = self.ttbot_pose.pose.orientation
        return yaw_from_quat(q.x, q.y, q.z, q.w)

    def set_goal_xy(self, x: float, y: float):
        # set a new map goal and force replanning
        g = PoseStamped()
        g.header.stamp = self.get_clock().now().to_msg()
        g.header.frame_id = 'map'
        g.pose.position.x = float(x)
        g.pose.position.y = float(y)
        g.pose.orientation.w = 1.0
        self.goal_pose = g
        self.has_goal = True

        self.last_goal_xy = None
        self.path_pts = []
        self.path = Path()
        self.in_subpath_segment = False
        self.subpath_start_idx = None
        self.subpath_end_idx = None

    def advance_waypoint(self):
        # go next waypoint and reset state
        self.wp_idx = (self.wp_idx + 1) % len(WAYPOINTS)
        self.has_goal = False
        self.last_goal_xy = None
        self.path_pts = []
        self.path = Path()
        self.in_subpath_segment = False
        self.subpath_start_idx = None
        self.subpath_end_idx = None
        self.mode = "NAV_WP"
        self.ball_target = None
        self.ball_stage = "ALIGN"
        self.spin_accum = 0.0
        self.spin_last_yaw = None

    # =========================
    # Camera callbacks / helpers
    # =========================
    def on_cinfo(self, m: CameraInfo):
        if len(m.k) == 9 and m.k[0] > 0:
            self.fx = float(m.k[0])

    def clean_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return mask

    def find_largest_bbox(self, mask):
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        if not contours:
            return False, 0, 0, 0, 0, 0.0
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        if w >= 6 and h >= 6:
            return True, x, y, w, h, area
        return False, 0, 0, 0, 0, 0.0

    def estimate_Z_X_from_bbox(self, x, y, w, h, W):
        # pinhole depth estimate using ball diameter and pixel size
        if self.fx is None:
            return None, None
        D = float(self.p('ball_size'))
        s_px = max(1.0, float(max(w, h)))
        Z = (self.fx * D) / s_px

        # lateral offset from image center
        cx_obj = x + w / 2.0
        u_offset = (cx_obj - (W / 2.0))
        theta = math.atan2(u_offset, self.fx)
        X_cam = Z * math.tan(theta)
        return Z, X_cam

    def cam_to_map(self, Z_forward: float, X_left: float) -> Optional[Tuple[float, float]]:
        # robot frame (forward Z, left X) -> map frame using AMCL pose
        if not self.have_pose:
            return None
        px = self.ttbot_pose.pose.position.x
        py = self.ttbot_pose.pose.position.y
        q  = self.ttbot_pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        x_r = Z_forward
        y_r = X_left
        x_map = px + x_r * math.cos(yaw) - y_r * math.sin(yaw)
        y_map = py + x_r * math.sin(yaw) + y_r * math.cos(yaw)
        return (float(x_map), float(y_map))

    def recently_seen(self, color: str, now_s: float, timeout_s: float) -> bool:
        det = self.ball_det[color]
        return bool(det["found"]) and ((now_s - float(det["t"])) <= timeout_s)

    def det_distance_m(self, det: Dict[str, Any]) -> Optional[float]:
        # prefer map distance if available, else use camera (Z,X)
        if det.get("map_xy") is not None and self.have_pose:
            rx = self.ttbot_pose.pose.position.x
            ry = self.ttbot_pose.pose.position.y
            bx, by = det["map_xy"]
            return float(math.hypot(bx - rx, by - ry))
        if det.get("Z") is not None and det.get("X_cam") is not None:
            return float(math.hypot(float(det["Z"]), float(det["X_cam"])))
        return None

    def on_image(self, msg: Image):
        # detect balls via HSV thresholding and publish map positions continuously
        self.camera_ready = True

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        W = frame.shape[1]

        hsv = cv2.cvtColor(cv2.medianBlur(frame, 5), cv2.COLOR_BGR2HSV)
        s_low, v_low = int(self.p('hsv_S_low')), int(self.p('hsv_V_low'))

        # Red (single range low1/high1)
        r1l = int(self.p('hsv_R_H_low1'))
        r1h = int(self.p('hsv_R_H_high1'))
        red_mask = self.clean_mask(cv2.inRange(hsv, (r1l, s_low, v_low), (r1h, 255, 255)))
        found_red, rx, ry, rw, rh, rarea = self.find_largest_bbox(red_mask)

        # Green
        gl, gh = int(self.p('hsv_G_H_low')), int(self.p('hsv_G_H_high'))
        green_mask = self.clean_mask(cv2.inRange(hsv, (gl, s_low, v_low), (gh, 255, 255)))
        found_green, gx, gy, gw, gh2, garea = self.find_largest_bbox(green_mask)

        # Blue
        bl, bh = int(self.p('hsv_B_H_low')), int(self.p('hsv_B_H_high'))
        blue_mask = self.clean_mask(cv2.inRange(hsv, (bl, s_low, v_low), (bh, 255, 255)))
        found_blue, bx, by, bw, bh2, barea = self.find_largest_bbox(blue_mask)

        now_s = self.get_clock().now().nanoseconds * 1e-9

        def update_det(color: str, found: bool, x: int, y: int, w: int, h: int, area: float, pub_pos):
            # already finalized -> stop updating
            if self.ball_pos_map[color] is not None:
                self.ball_det[color]["found"] = False
                return
            if not found:
                self.ball_det[color]["found"] = False
                return

            # normalized image x error (center=0)
            cx_obj = x + w / 2.0
            x_err = (cx_obj - (W / 2.0)) / (W * 0.5)

            # estimate (Z,X) and convert to map
            Z, X_cam = self.estimate_Z_X_from_bbox(x, y, w, h, W)

            map_xy = None
            if Z is not None and X_cam is not None and self.have_pose:
                map_xy = self.cam_to_map(Z, X_cam)
                if map_xy is not None:
                    # publish for task requirement
                    self.publish_point(pub_pos, map_xy[0], map_xy[1])

            self.ball_det[color].update({
                "found": True, "t": now_s, "bbox": (x, y, w, h),
                "area": float(area), "x_err": float(x_err),
                "Z": Z, "X_cam": X_cam, "map_xy": map_xy,
            })

        update_det("Red",   found_red,   rx, ry, rw, rh, rarea, self.pub_red_pos)
        update_det("Green", found_green, gx, gy, gw, gh2, garea, self.pub_green_pos)
        update_det("Blue",  found_blue,  bx, by, bw, bh2, barea, self.pub_blue_pos)

        # Optional debug visualization
        if bool(self.p('show_window')):
            disp = frame.copy()
            if found_red and self.ball_pos_map["Red"] is None:
                cv2.rectangle(disp, (rx, ry), (rx+rw, ry+rh), (0,0,255), 2)
            if found_green and self.ball_pos_map["Green"] is None:
                cv2.rectangle(disp, (gx, gy), (gx+gw, gy+gh2), (0,255,0), 2)
            if found_blue and self.ball_pos_map["Blue"] is None:
                cv2.rectangle(disp, (bx, by), (bx+bw, by+bh2), (255,0,0), 2)

            scale = float(self.p('visualize'))
            if 0.05 <= scale < 2.0 and abs(scale - 1.0) > 1e-3:
                disp = cv2.resize(disp, (int(disp.shape[1]*scale), int(disp.shape[0]*scale)))
            cv2.imshow('CB-frame', disp)
            cv2.waitKey(1)

    # =========================
    # Map / Pose callbacks
    # =========================
    def __map_cbk(self, msg: OccupancyGrid):
        self.map_proc.update_from_msg(msg, wall_buffer_m)
        self.map_ready = True

        # map changed -> force next plan
        self.has_goal = False
        self.last_goal_xy = None
        self.path_pts = []
        self.path = Path()

        self.in_subpath_segment = False
        self.subpath_start_idx = None
        self.subpath_end_idx = None

        # inflated map publish (RViz)
        self.inflated_map_pub.publish(self.map_proc.to_inflated_occgrid(self.get_clock().now().to_msg()))

        # reset mode
        self.mode = "NAV_WP"

    def __ttbot_pose_cbk(self, data: PoseWithCovarianceStamped):
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose = data.pose.pose
        self.have_pose = True

    # =========================
    # Subpath replanning helper (A*)
    # =========================
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
        SAFE_CLEAR_RADIUS = 0.65

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
        sub_goal_xy  = self.path_pts[sub_goal_idx]

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

        # mark subpath segment to limit speed while inside it
        self.subpath_start_idx = sub_start_idx
        self.subpath_end_idx   = sub_start_idx + len(sub_path_xy) - 1
        self.in_subpath_segment = True
        return True

    # =========================
    # LaserScan callback (dynamic obstacles + subpath replanning)
    # =========================
    def __scan_cbk(self, msg: LaserScan):
        if self.mode not in ("NAV_WP", "BALL_NAV"):
            return
        if not self.map_ready or not self.path_pts or not self.have_pose:
            return

        # Robot position
        px = self.ttbot_pose.pose.position.x
        py = self.ttbot_pose.pose.position.y
        q  = self.ttbot_pose.pose.orientation
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
            wall_range = self.map_proc.raycast_to_static_obstacle(px, py, phi, LASER_OBS_DIST + 0.3)
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
            self.inflated_map_pub.publish(self.map_proc.to_inflated_occgrid(self.get_clock().now().to_msg()))

            # 4) Try A* subpath replanning
            ok = self._replan_subpath_around_obstacle(wx, wy)
            if not ok:
                # If it fails, fall back to full replanning (full A* computed next loop)
                self.last_goal_xy = None

            break  # Process only once and exit the callback

    # =========================
    # A* Global planner
    # =========================
    def a_star_path_planner(self, start_pose: PoseStamped, end_pose: PoseStamped) -> Path:
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        start_time = self.get_clock().now().nanoseconds * 1e-9

        # if no goal or no map ==> empty path
        if not (self.has_goal and self.map_ready):
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - start_time)
            self.calc_time_pub.publish(tmsg)
            return path_msg

        # reuse path if goal not changed
        gx_now = end_pose.pose.position.x
        gy_now = end_pose.pose.position.y
        if self.path_pts and self.last_goal_xy == (gx_now, gy_now):
            tmsg = Float32()
            tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - start_time)
            self.calc_time_pub.publish(tmsg)
            return self.path

        # world → grid
        s = start_pose.pose.position
        g = end_pose.pose.position
        sg = self.map_proc.pose_to_grid(s.x, s.y)
        gg = self.map_proc.pose_to_grid(g.x, g.y)

        grid_path = self.planner.plan(sg, gg)

        # astar_time publish
        tmsg = Float32()
        tmsg.data = float(self.get_clock().now().nanoseconds * 1e-9 - start_time)
        self.calc_time_pub.publish(tmsg)

        # if fail ==> empty path
        if grid_path is None or len(grid_path) < 2:
            self.path_pts = []
            self.path = path_msg
            self.last_goal_xy = None
            self.has_goal = False
            self.in_subpath_segment = False
            self.subpath_start_idx = None
            self.subpath_end_idx = None
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

        # new global plan -> reset subpath flags
        self.in_subpath_segment = False
        self.subpath_start_idx = None
        self.subpath_end_idx = None
        return self.path

    # =========================
    # Path index chooser
    # =========================
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

        base_i = max(nearest_index(px, py), self.target_idx)
        li = base_i
        for i in range(base_i, len(self.path_pts)):
            if math.hypot(self.path_pts[i][0] - px, self.path_pts[i][1] - py) >= lookahead_m:
                li = i
                break
        self.target_idx = max(self.target_idx, li)
        self.target_idx = int(clamp(self.target_idx, 0, len(self.path_pts) - 1))

        # update in_subpath flag
        if self.subpath_start_idx is not None and self.subpath_end_idx is not None:
            if self.subpath_start_idx <= self.target_idx <= self.subpath_end_idx:
                self.in_subpath_segment = True
            elif self.target_idx > self.subpath_end_idx:
                self.in_subpath_segment = False

        return int(self.target_idx)

    # =========================
    # Path follower
    # =========================
    def path_follower(self, vehicle_pose: PoseStamped, current_goal_pose: PoseStamped, v_cap: Optional[float]=None) -> Tuple[float, float]:
        if not self.path_pts:
            return 0.0, 0.0

        px = vehicle_pose.pose.position.x
        py = vehicle_pose.pose.position.y
        q  = vehicle_pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        tx = current_goal_pose.pose.position.x
        ty = current_goal_pose.pose.position.y

        dx = tx - px
        dy = ty - py
        dist = math.hypot(dx, dy)
        des_yaw = math.atan2(dy, dx)
        yaw_err = wrap_angle(des_yaw - yaw)

        # choose speed limit
        v_max = max_lin_vel_subpath if self.in_subpath_segment else max_lin_vel_normal
        if v_cap is not None:
            v_max = min(v_max, float(v_cap))

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
            v_cmd = clamp(v_cmd, 0.0, v_max)

        return float(v_cmd), float(w_cmd)

    def move_ttbot(self, speed: float, heading: float):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(speed)
        cmd_vel.angular.z = float(heading)
        self.cmd_vel_pub.publish(cmd_vel)

    # =========================
    # Ball servo / finalize
    # =========================
    def enter_ball_servo(self, color: str):
        self.mode = "BALL_SERVO"
        self.ball_target = color
        self.ball_stage = "ALIGN"
        self.ball_align_stable = 0
        self.ball_ang_pid.reset()
        self.ball_lin_pid.reset()
        self.ball_last_pid_t = self.get_clock().now()
        self.ball_close_samples[color] = []

    def ball_servo_step(self):
        if self.ball_target is None:
            self.mode = "NAV_WP"
            return

        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        dt = (now - self.ball_last_pid_t).nanoseconds / 1e9
        self.ball_last_pid_t = now
        dt = max(dt, 1.0 / 60.0)

        det = self.ball_det[self.ball_target]
        recently = det["found"] and ((now_s - float(det["t"])) <= BALL_DET_TIMEOUT_S)

        # if lost -> rotate to reacquire
        if (not recently) or (det["Z"] is None) or (self.fx is None):
            self.move_ttbot(0.0, float(np.clip(BALL_SEARCHING_ANG, -BALL_ANG_VMAX, BALL_ANG_VMAX)))
            self.ball_ang_pid.reset()
            self.ball_lin_pid.reset()
            self.ball_align_stable = 0
            return

        x_err = float(det["x_err"])
        Z = float(det["Z"])

        # angular control from image x error
        ang = self.ball_ang_pid.step(-x_err, dt)
        if abs(x_err) > BALL_ALIGN_BAND * 1.5 and 0.0 < abs(ang) < BALL_MIN_ABS_ANG:
            ang = BALL_MIN_ABS_ANG if ang >= 0 else -BALL_MIN_ABS_ANG
        ang = float(np.clip(ang, -BALL_ANG_VMAX, BALL_ANG_VMAX))

        if self.ball_stage == "ALIGN":
            # stabilize alignment before moving forward
            self.ball_lin_pid.reset()
            self.ball_align_stable = self.ball_align_stable + 1 if abs(x_err) <= BALL_ALIGN_BAND * 0.6 else 0
            if self.ball_align_stable >= BALL_ALIGN_STABLE_FRAMES:
                self.ball_stage = "APPROACH"
                self.ball_align_stable = 0
                self.ball_ang_pid.reset()
                self.ball_lin_pid.reset()
            self.move_ttbot(0.0, ang)
            return

        # linear control on depth error
        target = BALL_FINAL_DIST_M
        z_err = Z - target

        if abs(z_err) <= BALL_DIST_BAND_M:
            lin = 0.0
            self.ball_lin_pid.reset()
        else:
            lin = self.ball_lin_pid.step(z_err / max(1e-6, target), dt)
            if 0.0 < abs(lin) < BALL_MIN_ABS_LIN:
                lin = BALL_MIN_ABS_LIN if lin >= 0 else -BALL_MIN_ABS_LIN

        # stop forward motion if not aligned
        if abs(x_err) > BALL_ALIGN_BAND:
            lin = 0.0

        lin = float(np.clip(lin, -BALL_BACK_VMAX, BALL_LIN_VMAX))

        # record close samples for stable finalize
        if det["map_xy"] is not None and Z <= 0.8:
            self.ball_close_samples[self.ball_target].append(det["map_xy"])
            if len(self.ball_close_samples[self.ball_target]) > 30:
                self.ball_close_samples[self.ball_target].pop(0)

        # if close enough -> finalize
        if Z <= (BALL_FINAL_DIST_M + BALL_DIST_BAND_M):
            self.ball_stage = "FINALIZE"
            self.move_ttbot(0.0, 0.0)
            return

        self.move_ttbot(lin, ang)

    def finalize_ball(self, color: str):
        det = self.ball_det[color]
        final_xy = None
        buf = self.ball_close_samples[color]

        if len(buf) >= self.close_samples_needed:
            xs = [p[0] for p in buf[-self.close_samples_needed:]]
            ys = [p[1] for p in buf[-self.close_samples_needed:]]
            final_xy = (sum(xs)/len(xs), sum(ys)/len(ys))
        elif det["map_xy"] is not None:
            final_xy = det["map_xy"]

        if final_xy is None:
            self.get_logger().warn(f"[FINALIZE FAIL] {color}")
            self.ball_stage = "ALIGN"
            return

        self.ball_pos_map[color] = (float(final_xy[0]), float(final_xy[1]))
        self.get_logger().info(f"[BALL FINAL] {color}: ({final_xy[0]:.3f}, {final_xy[1]:.3f})")

        # publish final position (task requirement)
        if color == "Red":
            self.publish_point(self.pub_red_pos, final_xy[0], final_xy[1])
        elif color == "Green":
            self.publish_point(self.pub_green_pos, final_xy[0], final_xy[1])
        else:
            self.publish_point(self.pub_blue_pos, final_xy[0], final_xy[1])

        # continue mission
        self.ball_target = None
        self.ball_stage = "ALIGN"
        self.mode = "NAV_WP"
        self.advance_waypoint()

    # =========================
    # Spin search
    # =========================
    def enter_spin(self):
        self.mode = "SPIN"
        self.spin_accum = 0.0
        self.spin_last_yaw = self.current_yaw()

        self.has_goal = False
        self.last_goal_xy = None
        self.path_pts = []
        self.path = Path()

    def spin_step(self):
        yaw = self.current_yaw()
        if yaw is None:
            self.move_ttbot(0.0, 0.0)
            return
        if self.spin_last_yaw is None:
            self.spin_last_yaw = yaw

        dy = wrap_angle(yaw - self.spin_last_yaw)
        self.spin_accum += abs(dy)
        self.spin_last_yaw = yaw

        self.move_ttbot(0.0, SPIN_ANG_VEL)

    def select_ball_candidate(self, now_s: float) -> Optional[str]:
        # pick a ball: recent, within radius, prefer larger area and nearer distance
        candidates = []
        for c in ["Red", "Green", "Blue"]:
            if self.ball_pos_map[c] is not None:
                continue
            if not self.recently_seen(c, now_s, BALL_DET_TIMEOUT_S):
                continue
            det = self.ball_det[c]
            d = self.det_distance_m(det)
            if d is None or d > BALL_SPIN_SEARCH_RADIUS_M:
                continue
            candidates.append((float(det["area"]), -float(d), c))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]

    # =========================
    # Main control loop
    # =========================
    def control_loop(self):
        if not (self.map_ready and self.have_pose and self.camera_ready):
            self.move_ttbot(0.0, 0.0)
            return

        # if all balls done -> keep publishing final positions and stop
        if self.all_balls_found():
            if self.ball_pos_map["Red"] is not None:
                x, y = self.ball_pos_map["Red"]
                self.publish_point(self.pub_red_pos, x, y)
            if self.ball_pos_map["Green"] is not None:
                x, y = self.ball_pos_map["Green"]
                self.publish_point(self.pub_green_pos, x, y)
            if self.ball_pos_map["Blue"] is not None:
                x, y = self.ball_pos_map["Blue"]
                self.publish_point(self.pub_blue_pos, x, y)
            self.move_ttbot(0.0, 0.0)
            return

        now_s = self.get_clock().now().nanoseconds * 1e-9

        # ----- SPIN -----
        if self.mode == "SPIN":
            cand = self.select_ball_candidate(now_s)
            if cand is not None and self.ball_det[cand]["map_xy"] is not None:
                self.ball_target = cand
                self.mode = "BALL_NAV"
                self.move_ttbot(0.0, 0.0)
                return

            if self.spin_accum < SPIN_TARGET_RAD:
                self.spin_step()
                return

            self.move_ttbot(0.0, 0.0)
            self.advance_waypoint()
            return

        # ----- BALL_NAV -----
        if self.mode == "BALL_NAV":
            if self.ball_target is None:
                self.enter_spin()
                return

            # abort if ball lost too long
            if not self.recently_seen(self.ball_target, now_s, BALL_LOST_ABORT_S):
                self.get_logger().warn(f"[BALL LOST] {self.ball_target}")
                self.ball_target = None
                self.enter_spin()
                return

            det = self.ball_det[self.ball_target]

            # close enough -> servo
            if det["Z"] is not None and float(det["Z"]) <= BALL_TRIGGER_DIST_M:
                self.move_ttbot(0.0, 0.0)
                self.enter_ball_servo(self.ball_target)
                return

            # if no map_xy yet -> rotate to get better view
            if det["map_xy"] is None:
                self.move_ttbot(0.0, 0.6)
                return

            # plan to the current estimated ball position
            bx, by = det["map_xy"]
            self.set_goal_xy(bx, by)

            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            if not path.poses:
                self.get_logger().warn("[A* FAIL] to ball")
                self.ball_target = None
                self.enter_spin()
                return

            idx = self.get_path_idx(path, self.ttbot_pose)
            if not (0 <= idx < len(path.poses)):
                self.move_ttbot(0.0, 0.0)
                return

            current_goal = path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal, v_cap=BALL_LIN_VMAX)
            self.move_ttbot(speed, heading)
            return

        # ----- BALL_SERVO -----
        if self.mode == "BALL_SERVO":
            if self.ball_stage == "FINALIZE" and self.ball_target is not None:
                self.move_ttbot(0.0, 0.0)
                self.finalize_ball(self.ball_target)
                return
            self.ball_servo_step()
            return

        # ----- NAV_WP -----
        if self.mode == "NAV_WP":
            wx, wy = WAYPOINTS[self.wp_idx]

            if not self.has_goal:
                self.set_goal_xy(wx, wy)

            rx = self.ttbot_pose.pose.position.x
            ry = self.ttbot_pose.pose.position.y

            # waypoint reached -> spin
            if math.hypot(wx - rx, wy - ry) <= WP_REACHED_TOL_M:
                self.move_ttbot(0.0, 0.0)
                self.enter_spin()
                return

            # normal waypoint navigation
            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            if not path.poses:
                self.move_ttbot(0.0, 0.0)
                return

            idx = self.get_path_idx(path, self.ttbot_pose)
            if not (0 <= idx < len(path.poses)):
                self.move_ttbot(0.0, 0.0)
                return

            current_goal = path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)
            return


def main(args=None):
    rclpy.init(args=args)
    node = Task3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
