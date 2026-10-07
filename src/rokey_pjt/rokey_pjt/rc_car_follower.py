"""RC카 추적.

1) 접근: 웹캠 PC 의 /webcam/target_pose/car (map 좌표) -> Nav2 goal (RC카 위치, 경로상 STANDOFF 남으면 정지)
   (웹캠 좌표 근처에 처음 도착하기 전에는 로봇 카메라에 보여도 추적으로 넘어가지 않음)
2) 추적: 도착 후 로봇 OAK-D 로 RC카가 보이면 bbox 방향 + depth 거리로 RC카 map 좌표를 계산해
   Nav2 goal (RC카 위치, CHASE_DIST 남으면 정지) 로 따라감 -> 추적 중에도 Nav2 가 장애물 회피.
   CHASE_DIST 안이면 goal 취소 후 제자리 회전으로 RC카를 화면 중앙에 유지.
   로봇 카메라에서 SEARCH_AFTER(3초) 이상 놓치면 3) 으로
3) 탐색: Nav2 goal 취소 후 제자리 360도 회전하며 로봇 카메라로 RC카 탐색 (마지막으로 본 쪽으로 회전)
   회전 중 보이면 2) 로, 한 바퀴 돌아도 없으면 1) 로 복귀

전제: localization 실행 중. nav2 는 이 노드 전/후 언제 실행해도 됨.
     nav2 가 TF 없을 때 먼저 떠서 bringup 이 중단(planner/global_costmap 비활성)됐으면,
     TF 확인 후 nav2 노드들을 직접 configure/activate 해서 복구한다.
     (lifecycle_manager 는 bringup 실패 후 서비스 응답을 안 해서 RESET/STARTUP 으로는 복구가 안 됨)
시작 시 도킹 상태면 웹캠에서 RC카가 탐지될 때까지 도크에서 대기 -> 탐지되면 언도킹 후 UNDOCK_POSE 로
initial pose 자동 설정하고 1) 진행 (도킹 상태가 아니면 RViz 에서 initial pose 직접 지정).
"""
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, LaserScan
from std_srvs.srv import Empty
from nav2_simple_commander.robot_navigator import TaskResult
from tf2_ros import Buffer, TransformListener
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from ultralytics import YOLO

# ================================
# 설정 상수
# ================================
TARGET_TOPIC = '/webcam/target_pose/car'               # 웹캠 PC 가 발행 (map 좌표)
RGB_TOPIC = 'oakd/rgb/image_raw/compressed'            # 아래 3개는 __ns:=/robot6 기준 상대 이름
DEPTH_TOPIC = 'oakd/stereo/image_raw/compressedDepth'
CMD_TOPIC = 'cmd_vel'
SCAN_TOPIC = 'scan'
MODEL_PATH = '/home/mu-06/rokey_ws/my_best26n.pt'
CONF_THRESHOLD = 0.5
UNDOCK_POSE = ([-0.07968, 1.78908], 178.2)  # 언도킹 직후 map 좌표 (x, y), 방향(deg)
# nav2 bringup 순서 (lifecycle_manager_navigation 의 node_names 와 동일)
NAV2_NODES = ('controller_server', 'smoother_server', 'planner_server', 'route_server', 'behavior_server',
              'velocity_smoother', 'collision_monitor', 'bt_navigator', 'waypoint_follower', 'docking_server')
COSTMAP_OWNER = {'global_costmap/global_costmap': 'planner_server',
                 'local_costmap/local_costmap': 'controller_server'}
NAV2_STALL_SEC = 30.0   # nav2 상태가 이 시간 동안 변화 없고 준비도 안 됐으면 직접 복구 시작
CAR_CLASS = 'car'
# 접근 단계 (Nav2)
STANDOFF = 1.0          # Nav2 경로상 RC카까지 이 거리(m) 남으면 정지
ARRIVE_MARGIN = 0.15    # 추적 단계 직선거리 도착 판정 여유(m)
GOAL_PERIOD = 1.0       # goal 갱신 최소 간격(초)
GOAL_MOVE_THRESH = 0.2  # goal 이 이만큼(m) 이상 움직여야 재전송
TARGET_TIMEOUT = 4.0    # 웹캠 좌표가 이 시간(초) 이상 안 오면 무시 (진행 중인 Nav2 goal 은 유지, 새 goal 만 안 보냄)
# 추적 단계 (로봇 카메라 -> Nav2)
CHASE_DIST = 1.0        # RC카와 유지할 거리(m)
CHASE_GOAL_PERIOD = 0.5 # 추적 중 goal 갱신 최소 간격(초)
SEARCH_AFTER = 3.0      # 추적 중 로봇 카메라에서 이 시간(초) 이상 못 보면 360도 회전 탐색 시작
SEARCH_ANG_VEL = 0.6    # 탐색 회전 속도(rad/s). 빠르면 영상이 흔들려 YOLO 가 놓침 (한 바퀴 약 10초)
SEARCH_TIMEOUT = 25.0   # 회전 각도를 못 재는 경우(TF 없음 등) 대비 탐색 최대 시간(초)
CAM_HFOV_DEG = 69.0     # OAK-D RGB 수평 화각(도). bbox 가로 위치 -> 방향각 변환에 사용
K_ANG, MAX_ANG = 1.2, 1.0    # 도착 후 제자리 회전 P 게인, 최대 각속도(rad/s)
CENTER_DEADBAND = 0.1   # 화면 중앙 기준 이 비율 안이면 회전 안 함
DEPTH_ROI = 5           # bbox 중심 주변 (2N+1)^2 픽셀의 depth 중앙값 사용
# ================================


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def decode_depth(msg):
    """compressedDepth = 설정 헤더 + PNG(16UC1, mm). PNG 시그니처부터 디코딩"""
    data = bytes(msg.data)
    i = data.find(b'\x89PNG')
    if i < 0:
        return None
    return cv2.imdecode(np.frombuffer(data[i:], np.uint8), cv2.IMREAD_UNCHANGED)


class RcCarFollower(Node):
    def __init__(self):
        super().__init__('rc_car_follower')
        self.model = YOLO(MODEL_PATH)
        self.tf_buffer = Buffer()
        # TF 는 별도 노드+스레드에서 수신. 메인 루프는 spin_once 한 번에 콜백 1개만 처리하고 YOLO 로 느려서,
        # 같은 노드에 두면 카메라 콜백에 밀려 TF 가 늦게/덜 들어온다. (__ns, /tf remap 은 전역 인자라 그대로 적용됨)
        self.tf_listener = TransformListener(self.tf_buffer, None, spin_thread=True)
        self.cmd_pub = self.create_publisher(TwistStamped, CMD_TOPIC, 10)

        self.target = None  # (x, y, 수신 시각) - 웹캠 PC 와 시계가 다를 수 있어 수신 시각 기준
        self.rgb_msg = None
        self.depth_msg = None
        self.last_scan = None  # 마지막 /scan 수신 시각 (라이다 상태 진단용)
        self.create_subscription(PoseStamped, TARGET_TOPIC, self.on_target, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.on_scan, qos_profile_sensor_data)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, RGB_TOPIC, self.on_rgb, qos)
        self.create_subscription(CompressedImage, DEPTH_TOPIC, self.on_depth, qos)

        self.state_clients = {}  # lifecycle get_state 클라이언트 캐시
        self.navigator = TurtleBot4Navigator()
        self.wait_lifecycle_active('amcl')
        if self.navigator.getDockedStatus():  # 도킹 상태로 시작: 언도킹하면 UNDOCK_POSE 위치이므로 initial pose 설정
            self.wait_webcam_target()  # RC카가 탐지돼야 출발
            self.get_logger().info('웹캠에서 RC카 탐지 -> 언도킹 후 초기 위치 설정')
            self.navigator.undock()
            self.navigator.setInitialPose(self.navigator.getPoseStamped(*UNDOCK_POSE))
        self.ensure_lidar()
        # nav2 의 global_costmap 은 활성화 시점에 map->base_link TF 가 없으면 활성화에 실패하고,
        # 이후 planner 가 'Costmap timed out waiting for update' 로 경로를 못 만든다.
        # 그래서 TF 를 확인한 뒤 nav2 를 기다리고, costmap 이 비활성이면 nav2 를 재활성화한다.
        self.wait_map_tf()
        self.ensure_nav2_active()
        self.navigator.waitUntilNav2Active()  # 위에서 active 를 확인했으므로 초기 pose 수신만 남아 바로 통과
        self.mode = 'approach'
        self.arrived = False  # 웹캠 좌표 근처에 처음 도착하기 전에는 로봇 카메라에 보여도 추적으로 넘어가지 않음
        self.last_seen = 0.0
        self.rotating = False  # 도착 후 제자리 회전 중 (cmd_vel 직접 발행 중)
        self.last_offset = 0.0  # 마지막으로 본 RC카의 화면 가로 위치 (-1 왼쪽 ~ 1 오른쪽) -> 탐색 회전 방향
        self.search = None      # 탐색 상태: {'dir', 'turned', 'prev_yaw', 'start'}
        self.last_goal = None
        self.last_send = 0.0
        self.arrived_at = None   # 도착 판정 당시 RC카 좌표 (RC카가 움직이면 다시 출발)
        self.fb_at_send = None   # goal 전송 시점의 feedback 객체 (이전 goal 의 feedback 구분용)

    def wait_lifecycle_active(self, node_name, call_timeout=2.0):
        """<ns>/<node_name>/get_state 에 요청을 보내 active 가 될 때까지 대기 (예: /robot6/amcl/get_state).
        waitUntilNav2Active 는 응답이 없으면 로그 없이 멈추므로, 서비스 이름/응답 상태를 로그로 남긴다."""
        srv = f'{self.get_namespace().rstrip("/")}/{node_name}/get_state'
        client = self.create_client(GetState, srv)
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{srv} 서비스 대기 중...')
        state = 'unknown'
        while state != 'active':
            future = client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=call_timeout)
            if future.done() and future.result() is not None:
                state = future.result().current_state.label
                self.get_logger().info(f'{srv} 상태: {state}')
            else:
                future.cancel()
                self.get_logger().warn(f'{srv} 요청 후 {call_timeout}초 내 응답 없음, 재요청')
            if state != 'active':
                time.sleep(1.0)
        self.destroy_client(client)

    def target_fresh(self):
        return self.target is not None and time.monotonic() - self.target[2] <= TARGET_TIMEOUT

    def wait_webcam_target(self):
        """웹캠에서 RC카 좌표(/webcam/target_pose/car)가 들어올 때까지 도크에서 대기"""
        start = time.monotonic()
        while not self.target_fresh():
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info(f'도킹 상태로 대기 중 ({time.monotonic() - start:.0f}초): 웹캠에서 RC카 탐지되면 언도킹 '
                                   f'({TARGET_TOPIC})', throttle_duration_sec=5.0)

    def scan_age(self):
        return None if self.last_scan is None else time.monotonic() - self.last_scan

    def wait_scan(self, sec):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            age = self.scan_age()
            if age is not None and age < 1.0:
                return True
        return False

    def ensure_lidar(self):
        """/scan 이 없으면 amcl 이 map->odom TF 를 발행하지 않는다 (도킹 중엔 라이다 모터가 꺼져 있음).
        이미 스캔 중인 라이다에 start_motor 를 보내면 드라이버가 멈출 수 있어서, /scan 이 안 들어올 때만 호출한다."""
        if self.wait_scan(5.0):
            self.get_logger().info('/scan 수신 확인 (라이다 동작 중)')
            return
        self.get_logger().warn('/scan 5초간 없음 -> start_motor 호출')
        client = self.create_client(Empty, 'start_motor')
        if client.wait_for_service(timeout_sec=10.0):
            future = client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        self.destroy_client(client)
        if self.wait_scan(10.0):
            self.get_logger().info('/scan 수신 확인 (라이다 시작됨)')
        else:
            self.get_logger().error("/scan 여전히 없음: 'ros2 topic info /robot6/scan' 의 Publisher count 가 0 이면 "
                                    "로봇의 라이다 드라이버가 죽은 것 -> 로봇에서 turtlebot4 서비스 재시작/재부팅")

    def wait_map_tf(self):
        start = time.monotonic()
        while self.robot_xy() is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info(f'map->base_link TF 대기 중 ({time.monotonic() - start:.0f}초), {self.scan_status()}',
                                   throttle_duration_sec=3.0)
        self.get_logger().info('map->base_link TF 확인 -> nav2 대기 (아직 안 띄웠으면 지금 nav2.launch.py 실행)')

    def lifecycle_state(self, node_name, tries=2, call_timeout=3.0):
        """<ns>/<node_name>/get_state 조회. 응답 없으면 tries 번까지 재시도, 끝내 없으면 None"""
        client = self.state_clients.get(node_name)
        if client is None:  # 클라이언트를 매번 만들면 discovery 가 매번 느려서 재사용
            client = self.state_clients[node_name] = self.create_client(GetState, f'{node_name}/get_state')
        if not client.wait_for_service(timeout_sec=2.0):
            return None
        for _ in range(tries):
            future = client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=call_timeout)
            if future.done() and future.result() is not None:
                return future.result().current_state.label
            future.cancel()
        return None

    def change_state(self, node_name, transition_id, label, timeout=90.0):
        """<ns>/<node_name>/change_state 요청. 응답이 유실될 수 있어 결과는 get_state 로 다시 확인한다"""
        client = self.create_client(ChangeState, f'{node_name}/change_state')
        if client.wait_for_service(timeout_sec=5.0):
            req = ChangeState.Request()
            req.transition.id = transition_id
            self.get_logger().info(f'{node_name} {label} 요청')
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        self.destroy_client(client)
        state = self.lifecycle_state(node_name)
        self.get_logger().info(f'{node_name} -> {state}')
        return state

    def nav2_states(self):
        names = NAV2_NODES + tuple(COSTMAP_OWNER)
        return {n: self.lifecycle_state(n) for n in names}

    @staticmethod
    def nav2_ready(states):
        # route_server/docking_server 등은 버전에 따라 없을 수 있어(None) 필수 노드만 본다
        required = ('controller_server', 'planner_server', 'behavior_server', 'bt_navigator') + tuple(COSTMAP_OWNER)
        return all(states.get(n) == 'active' for n in required)

    def wait_nav2_progress(self):
        """nav2 bringup 이 정상 진행 중이면 기다린다. 준비되면 True, NAV2_STALL_SEC 동안 변화 없으면 False"""
        log = self.get_logger()
        while self.lifecycle_state('bt_navigator') is None:  # nav2 가 아직 안 떴음
            log.info('nav2 대기 중: nav2.launch.py 를 실행하세요', throttle_duration_sec=5.0)
            time.sleep(1.0)
        prev, last_change = None, time.monotonic()
        while True:
            states = self.nav2_states()
            if self.nav2_ready(states):
                return True
            if states != prev:
                prev, last_change = states, time.monotonic()
                not_active = {n: st for n, st in states.items() if st != 'active'}
                log.info(f'nav2 bringup 진행 대기: 비활성 {not_active}')
            elif time.monotonic() - last_change > NAV2_STALL_SEC:
                return False
            time.sleep(2.0)

    def ensure_nav2_active(self, attempts=3):
        """TF 없을 때 먼저 뜬 nav2 는 global_costmap 활성화 실패로 bringup 이 중단되고,
        lifecycle_manager 도 응답을 멈춘다 -> 노드들을 bringup 순서대로 직접 configure/activate"""
        log = self.get_logger()
        if self.wait_nav2_progress():
            log.info('nav2 활성 확인 (costmap 포함)')
            return
        for _ in range(attempts):
            log.warn('nav2 bringup 이 멈춤 (TF 없을 때 nav2 가 먼저 시작됨) -> nav2 노드 직접 활성화')
            for node in NAV2_NODES:
                st = self.lifecycle_state(node)
                if st == 'unconfigured':
                    st = self.change_state(node, Transition.TRANSITION_CONFIGURE, 'configure')
                if st == 'inactive':
                    self.change_state(node, Transition.TRANSITION_ACTIVATE, 'activate')
            # 노드는 active 인데 costmap 만 비활성인 경우 -> 해당 서버를 껐다 켜서 costmap 재활성화
            for costmap, owner in COSTMAP_OWNER.items():
                if self.lifecycle_state(costmap) != 'active' and self.lifecycle_state(owner) == 'active':
                    self.change_state(owner, Transition.TRANSITION_DEACTIVATE, 'deactivate')
                    self.change_state(owner, Transition.TRANSITION_ACTIVATE, 'activate')
            states = self.nav2_states()
            if self.nav2_ready(states):
                log.info('nav2 직접 활성화 완료')
                return
            log.warn(f'아직 비활성: { {n: st for n, st in states.items() if st != "active"} }')
        log.error('nav2 활성화 실패: nav2.launch.py 를 재시작하세요 (TF 확인 로그 이후에 실행)')

    def on_target(self, msg):
        self.target = (msg.pose.position.x, msg.pose.position.y, time.monotonic())

    def on_scan(self, msg):
        self.last_scan = time.monotonic()

    def scan_status(self):
        age = self.scan_age()
        return '/scan 한 번도 못 받음' if age is None else f'마지막 /scan {age:.1f}초 전'

    def on_rgb(self, msg):
        self.rgb_msg = msg  # 최신 프레임만 유지

    def on_depth(self, msg):
        self.depth_msg = msg

    # ---------- 공통 ----------
    def step(self):
        now = time.monotonic()
        if self.rgb_msg is not None:
            self.process_camera(now)
        if self.mode == 'chase' and now - self.last_seen > SEARCH_AFTER:
            self.start_search()
        if self.mode == 'search':
            self.search_step()
        if self.mode == 'approach':
            self.approach()

    # ---------- 탐색 단계 ----------
    def odom_yaw(self):
        try:
            q = self.tf_buffer.lookup_transform('odom', 'base_link', Time()).transform.rotation
        except Exception:
            return None
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def start_search(self):
        if self.last_goal is not None:  # 진행 중인 Nav2 goal 취소 (제자리 회전과 충돌 방지)
            self.navigator.cancelTask()
            self.last_goal = None
        self.arrived_at = None
        direction = -1.0 if self.last_offset > 0 else 1.0  # 마지막에 오른쪽에서 봤으면 시계 방향
        self.search = {'dir': direction, 'turned': 0.0, 'prev_yaw': self.odom_yaw(), 'start': time.monotonic()}
        self.mode = 'search'
        self.get_logger().info(f'RC카를 {SEARCH_AFTER:.0f}초간 못 봄 -> 제자리 360도 회전 탐색 '
                               f'({"시계" if direction < 0 else "반시계"} 방향)')

    def search_step(self):
        st = self.search
        yaw = self.odom_yaw()
        if yaw is not None and st['prev_yaw'] is not None:
            d = math.atan2(math.sin(yaw - st['prev_yaw']), math.cos(yaw - st['prev_yaw']))  # -pi~pi 로 감기
            st['turned'] += abs(d)
        if yaw is not None:
            st['prev_yaw'] = yaw
        if st['turned'] >= 2 * math.pi or time.monotonic() - st['start'] > SEARCH_TIMEOUT:
            self.publish_cmd(0.0, 0.0)
            self.rotating = False
            self.search = None
            self.mode = 'approach'
            self.get_logger().info(f'{math.degrees(st["turned"]):.0f}도 회전했지만 RC카 못 찾음 -> 접근 단계(웹캠 좌표)로 복귀')
            return
        self.publish_cmd(0.0, st['dir'] * SEARCH_ANG_VEL)
        self.rotating = True

    def publish_cmd(self, lin, ang):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = lin
        msg.twist.angular.z = ang
        self.cmd_pub.publish(msg)

    def stop_rotation(self):
        if self.rotating:
            self.publish_cmd(0.0, 0.0)
            self.rotating = False

    def robot_pose(self):
        """map 기준 (x, y, yaw). TF 없으면 None"""
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def robot_xy(self):
        pose = self.robot_pose()
        return None if pose is None else pose[:2]

    def warn_no_tf(self):
        self.get_logger().warn(
            f'map->base_link TF 없음 ({self.scan_status()}): /scan 이 끊기면 amcl 이 map->odom 을 갱신 못 해 '
            f'약 10초 뒤 TF 가 사라짐. 스캔이 정상인데 이러면 amcl 초기 위치(RViz 2D Pose Estimate) 확인',
            throttle_duration_sec=3.0)

    def go_near(self, cx, cy, standoff, period):
        """RC카 위치 (cx, cy) 자체를 Nav2 goal 로 보내고, Nav2 경로상 남은 거리가 standoff 이하가 되면 멈춘다.
        - goal 을 '로봇->RC카 직선 위 standoff 앞' 에 두면 사이에 벽이 있을 때 goal 이 벽 앞/안에 찍혀
          로봇이 벽 앞까지 일직선으로 가서 멈춘다. RC카 위치로 보내면 Nav2 가 벽을 돌아가는 경로를 만든다.
        - 도착 판정은 직선거리가 아니라 Nav2 feedback 의 distance_remaining (경로 길이) 로 한다.
          (직선거리로 보면 벽 너머 RC카도 가깝다고 판단해 벽 앞에서 멈춤)
        도착이면 True."""
        log = self.get_logger()
        robot = self.robot_xy()
        if robot is None:
            self.warn_no_tf()
            return False
        dx, dy = cx - robot[0], cy - robot[1]
        # 이미 도착했고 RC카가 그 뒤로 거의 안 움직였으면 계속 도착 상태 (goal 재전송/취소 반복 방지)
        if self.arrived_at is not None and \
                math.hypot(cx - self.arrived_at[0], cy - self.arrived_at[1]) < GOAL_MOVE_THRESH:
            return True
        if self.last_goal is not None:
            if self.navigator.isTaskComplete():
                result = self.navigator.getResult()
                if result == TaskResult.SUCCEEDED:
                    return self.arrive(cx, cy, 'Nav2 goal 도착')
                if result == TaskResult.FAILED:
                    log.warn('Nav2 goal 실패 (경로 계획/제어 실패) -> 재전송. 반복되면 planner/controller 로그 확인',
                             throttle_duration_sec=3.0)
                self.last_goal = None  # 끝난 goal -> 아래에서 재전송
            else:
                fb = self.navigator.getFeedback()
                # 이전 goal 의 feedback 이 남아 있을 수 있어 goal 전송 후 새로 온 feedback 만 사용
                if fb is not None and fb is not self.fb_at_send and 0.01 < fb.distance_remaining < standoff:
                    return self.arrive(cx, cy, f'경로상 남은 거리 {fb.distance_remaining:.2f}m')
        # 추적 단계는 로봇 카메라에 보이는 상태 = 사이에 벽이 없음 -> 직선거리로도 판정
        if self.mode == 'chase' and math.hypot(dx, dy) < standoff + ARRIVE_MARGIN:
            return self.arrive(cx, cy, f'직선거리 {math.hypot(dx, dy):.2f}m')
        if time.monotonic() - self.last_send < period:
            return False
        if self.last_goal is not None and math.hypot(cx - self.last_goal[0], cy - self.last_goal[1]) <= GOAL_MOVE_THRESH:
            return False  # 진행 중인 goal 과 거의 같은 위치 -> 재전송 안 함 (재전송하면 매번 경로 재계획)
        self.stop_rotation()
        yaw = math.atan2(dy, dx)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = cx, cy
        goal.pose.orientation.z, goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.fb_at_send = self.navigator.getFeedback()
        self.navigator.goToPose(goal)
        self.last_goal, self.last_send, self.arrived_at = (cx, cy), time.monotonic(), None
        log.info(f'[{self.mode}] RC카 map ({cx:.2f}, {cy:.2f}) 로 goal 전송 (경로상 {standoff:.1f}m 남으면 정지)')
        return False

    def arrive(self, cx, cy, why):
        if self.last_goal is not None:
            self.navigator.cancelTask()
            self.last_goal = None
        self.arrived_at = (cx, cy)
        self.arrived = True
        self.get_logger().info(f'[{self.mode}] 도착 ({why}): RC카 ({cx:.2f}, {cy:.2f})')
        return True

    # ---------- 추적 단계 ----------
    def process_camera(self, now):
        msg, self.rgb_msg = self.rgb_msg, None
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        r = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        annotated = r.plot()
        cars = [b for b in r.boxes if r.names[int(b.cls)] == CAR_CLASS]
        dist = None
        if cars and self.depth_msg is not None:
            x1, y1, x2, y2 = max(cars, key=lambda b: float(b.conf)).xyxy[0].tolist()
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            dist = self.depth_at(u, v, frame.shape)

        if dist is not None and not self.arrived:
            self.get_logger().info('RC카가 보이지만 웹캠 좌표 도착 전 -> 접근 계속', throttle_duration_sec=3.0)
            dist = None
        if dist is not None:
            if self.mode == 'search':
                self.stop_rotation()
                self.search = None
                self.get_logger().info('탐색 회전 중 RC카 발견 -> 추적 재개')
            if self.mode != 'chase':
                self.mode = 'chase'
                self.get_logger().info('로봇 카메라에서 RC카 발견 -> 추적 단계 (Nav2 로 따라감)')
            self.last_seen = now
            self.chase(u, frame.shape[1], dist)
            cv2.putText(annotated, f'dist {dist:.2f} m', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 잠깐 놓친 경우(SEARCH_AFTER 이내)는 진행 중인 Nav2 goal 을 그대로 둔다
        cv2.putText(annotated, self.mode, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('RC car follower', annotated)

    def depth_at(self, u, v, rgb_shape):
        """RGB 픽셀 (u,v) 의 거리(m). 없으면 None"""
        depth = decode_depth(self.depth_msg)
        if depth is None or depth.dtype != np.uint16:
            self.get_logger().warn('depth 디코딩 실패 (16UC1 PNG 아님)', throttle_duration_sec=5.0)
            return None
        # ponytail: RGB 와 depth 가 같은 화각으로 정렬돼 있다고 보고 해상도 비율로만 매핑.
        #           거리가 엉뚱하면 camera_info + TF 로 정식 투영 필요
        h, w = depth.shape[:2]
        du, dv = int(u * w / rgb_shape[1]), int(v * h / rgb_shape[0])
        n = DEPTH_ROI
        roi = depth[max(dv - n, 0):dv + n + 1, max(du - n, 0):du + n + 1]
        valid = roi[roi > 0]
        return float(np.median(valid)) / 1000.0 if valid.size else None

    def chase(self, u, width, depth):
        """로봇 카메라 탐지 -> RC카 map 좌표 -> Nav2 goal. 가까우면 제자리 회전으로 화면 중앙 유지"""
        robot = self.robot_pose()
        if robot is None:
            self.warn_no_tf()
            return
        offset = (u - width / 2) / (width / 2)  # -1(왼쪽) ~ 1(오른쪽)
        self.last_offset = offset
        bearing = -math.atan(offset * math.tan(math.radians(CAM_HFOV_DEG) / 2))  # 왼쪽이 + (ROS 규약)
        rng = depth / math.cos(bearing)  # depth 는 광축 방향 거리 -> 실제 거리
        # ponytail: 카메라를 base_link 원점에 있다고 근사 (OAK-D 는 몇 cm 앞). goal 이 계속 갱신되므로 영향 작음
        x, y, yaw = robot
        cx, cy = x + rng * math.cos(yaw + bearing), y + rng * math.sin(yaw + bearing)
        if self.go_near(cx, cy, CHASE_DIST, CHASE_GOAL_PERIOD):
            # 도착: 이동 없이 제자리 회전만 (둥근 로봇이라 회전은 충돌 위험 거의 없음)
            ang = 0.0 if abs(offset) < CENTER_DEADBAND else clip(-K_ANG * offset, -MAX_ANG, MAX_ANG)
            self.publish_cmd(0.0, ang)
            self.rotating = ang != 0.0

    # ---------- 접근 단계 ----------
    def approach(self):
        log = self.get_logger()
        if self.target is None:
            log.info(f'웹캠 좌표 미수신 ({TARGET_TOPIC}): webcam_detector 실행/탐지 여부 확인', throttle_duration_sec=3.0)
            return
        age = time.monotonic() - self.target[2]
        if age > TARGET_TIMEOUT:
            log.info(f'웹캠 좌표가 {age:.1f}초 전 것 (>{TARGET_TIMEOUT}초) -> 무시: RC카가 웹캠에서 탐지되는지 확인',
                     throttle_duration_sec=3.0)
            return
        cx, cy, _ = self.target
        if self.go_near(cx, cy, STANDOFF, GOAL_PERIOD):  # 로봇 카메라에 안 보이면 여기서 대기
            log.info('웹캠 좌표 도착: 로봇 카메라에 RC카가 보이면 추적 시작', throttle_duration_sec=3.0)


def main():
    rclpy.init()
    node = RcCarFollower()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)  # 토픽/TF 수신. navigator 는 내부에서 자체 spin
            node.step()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        node.navigator.cancelTask()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
