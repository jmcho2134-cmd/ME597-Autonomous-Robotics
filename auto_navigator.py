import math
import time
from collections import deque
from heapq import heappush, heappop
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float32


# Parameters
wall_buffer_m   = 0.30   # obstacle inflation
allow_diag      = True   # allow diagonal moves
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
goal_tol_m      = 0.05


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


# Map Process : Load map , Node/Tree , Map Conversion 
class MapProcessor:
    def __init__(self):
        self.grid: Optional[List[int]] = None
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.inflated: Optional[List[int]] = None
        self.ready = False

    def update_from_msg(self, msg: OccupancyGrid, inflation_radius_m: float = wall_buffer_m):
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.grid = list(msg.data)  # [-1, 0..100]
        self.inflated = self._inflate_grid(self.grid, self.width, self.height, inflation_radius_m)
        self.ready = True

    def _inflate_grid(self, data: List[int], w: int, h: int, inflation_radius_m: float) -> List[int]:
        occ = [1 if (v < 0 or v > 0) else 0 for v in data]
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

        cells = int(math.ceil(inflation_radius_m / max(self.resolution, 1e-6)))
        inflated = [0] * (w * h)
        for i, d in enumerate(dist):
            if d <= cells:
                inflated[i] = 1
        return inflated

    # Map conversions
    def pose_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        gx = int((x_m - self.origin_x) / self.resolution + 0.5)
        gy = int((y_m - self.origin_y) / self.resolution + 0.5)
        return gx, gy

    def grid_to_pose(self, gx: int, gy: int) -> Tuple[float, float]:
        x = gx * self.resolution + self.origin_x
        y = gy * self.resolution + self.origin_y
        return x, y

    def is_free(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return False
        return self.inflated[gy * self.width + gx] == 0

    def in_bounds(self, gx: int, gy: int) -> bool:
        return (0 <= gx < self.width) and (0 <= gy < self.height)


# A* Algorithm 
class AStar:
    def __init__(self, mp: MapProcessor, use_8_conn: bool = allow_diag):
        self.mp = mp
        if use_8_conn:
            self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]
            self.step_cost = [1, 1, 1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]
        else:
            self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self.step_cost = [1, 1, 1, 1]

    def h(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

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
                tentative = gscore[base_i] + self.step_cost[k]
                if tentative < gscore[ni]:
                    gscore[ni] = tentative
                    came[ni] = base_i
                    heappush(pq, (tentative + self.h((nx, ny), goal_g), tentative, (nx, ny)))
        return None


# Navigation 
class Navigation(Node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)

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

        # controll memory
        self.prev_yaw_err = 0.0
        self.ang_err_int = 0.0

        # map + planner
        self.map_proc = MapProcessor()
        self.planner = AStar(self.map_proc, use_8_conn=allow_diag)

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)
        self.create_subscription(OccupancyGrid, '/map', self.__map_cbk, 1)

        # Publishers
        self.path_pub    = self.create_publisher(Path,   'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist,  'cmd_vel',     10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time', 10)  # DO NOT MODIFY

        # Node rate
        self.loop_hz = 10.0
       

    #Callbacks 
    def __map_cbk(self, msg: OccupancyGrid):
        self.map_proc.update_from_msg(msg, wall_buffer_m)
        self.map_ready = True
        # map changed -> force next plan
        self.last_goal_xy = None
        self.path_pts = []
        self.path = Path()
        
        self.get_logger().info(f'map received: {self.map_proc.width}x{self.map_proc.height}, res={self.map_proc.resolution:.3f}')

    def __goal_pose_cbk(self, data: PoseStamped):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.has_goal = True
        
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data: PoseWithCovarianceStamped):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose   = data.pose.pose
     
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

    
    def a_star_path_planner(self, start_pose: PoseStamped, end_pose: PoseStamped) -> Path:
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds * 1e-9  # Do not edit this line (required for autograder)

        # if no goal ==> empty path 
        if not (self.has_goal and self.map_ready):
            # astar_time publish 
            astar_time = Float32()
            astar_time.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(astar_time)
            return path_msg
            
        # reuse path if goal not changed
        gx_now = end_pose.pose.position.x
        gy_now = end_pose.pose.position.y
        if self.path_pts and self.last_goal_xy == (gx_now, gy_now):
            # astar_time publish
            astar_time = Float32()
            astar_time.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
            self.calc_time_pub.publish(astar_time)
            return self.path

        # cordinate transformation 
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

        # grid path -> map coordinate transformation 
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

    
    def get_path_idx(self, path: Path, vehicle_pose: PoseStamped) -> int:
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        if not self.path_pts:
            self.target_idx = 0
            return 0

        px = vehicle_pose.pose.position.x
        py = vehicle_pose.pose.position.y

        # 1) if target is close enough go to next one 
        while self.target_idx < len(self.path_pts):
            tx, ty = self.path_pts[self.target_idx]
            if math.hypot(tx - px, ty - py) < waypoint_tol_m:
                self.target_idx += 1
            else:
                break

        # 2) select the next point  
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
            self.target_idx = clamp(self.target_idx, 0, len(self.path_pts) - 1)
        else:
            self.target_idx = 0

        return int(self.target_idx)

    def path_follower(self, vehicle_pose: PoseStamped, current_goal_pose: PoseStamped) -> Tuple[float, float]:
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
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

        # PID controller 
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
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        cmd_vel.linear.x = float(speed)
        cmd_vel.angular.z = float(heading)
        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        period = 1.0 / self.loop_hz
        while rclpy.ok():
            # Call the spin_once to handle callbacks
            rclpy.spin_once(self, timeout_sec=0.0)  # Process callbacks without blocking

            # 1. Create the path to follow
            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)

            # 2. Loop through the path and move the robot
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
                if math.hypot(gx - self.ttbot_pose.pose.position.x, gy - self.ttbot_pose.pose.position.y) < goal_tol_m:
                    speed, heading = 0.0, 0.0

            self.move_ttbot(speed, heading)
            time.sleep(period)


def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(node_name='Navigation')
    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

