"""RC카 추적.

1) 접근: 웹캠 PC 의 /webcam/target_pose/car (map 좌표) -> Nav2 goal (RC카 앞 STANDOFF)
2) 추적: 로봇 OAK-D 로 RC카가 보이면 Nav2 취소 -> depth 거리로 cmd_vel 제어, CHASE_DIST 까지 접근
   로봇 카메라에서 LOST_TIMEOUT 이상 놓치면 다시 1) 로 복귀

전제: Nav2 + localization 실행 중, RViz에서 initial pose 지정 완료.
"""
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
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
MODEL_PATH = '/home/hv-06/rokey_ws/my_best26n.pt'
CONF_THRESHOLD = 0.5
CAR_CLASS = 'car'
# 접근 단계 (Nav2)
STANDOFF = 1.0          # Nav2 goal 을 RC카 앞 몇 m 에 둘지
ARRIVE_MARGIN = 0.15    # 이 거리 안이면 도착으로 보고 정지
GOAL_PERIOD = 1.0       # goal 갱신 최소 간격(초)
GOAL_MOVE_THRESH = 0.2  # goal 이 이만큼(m) 이상 움직여야 재전송
TARGET_TIMEOUT = 2.0    # 웹캠 좌표가 이 시간(초) 이상 안 오면 무시
# 추적 단계 (depth + cmd_vel)
CHASE_DIST = 1.0        # RC카와 유지할 거리(m)
LOST_TIMEOUT = 1.0      # 로봇 카메라에서 이 시간(초) 이상 못 보면 접근 단계로 복귀
K_LIN, MAX_LIN = 0.5, 0.25   # 전진 P 게인, 최대 속도(m/s)
K_ANG, MAX_ANG = 1.2, 1.0    # 회전 P 게인, 최대 각속도(rad/s)
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
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(TwistStamped, CMD_TOPIC, 10)

        self.target = None  # (x, y, 수신 시각) - 웹캠 PC 와 시계가 다를 수 있어 수신 시각 기준
        self.rgb_msg = None
        self.depth_msg = None
        self.create_subscription(PoseStamped, TARGET_TOPIC, self.on_target, 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, RGB_TOPIC, self.on_rgb, qos)
        self.create_subscription(CompressedImage, DEPTH_TOPIC, self.on_depth, qos)

        self.navigator = TurtleBot4Navigator()
        self.navigator.waitUntilNav2Active()
        self.mode = 'approach'
        self.last_seen = 0.0
        self.last_goal = None
        self.last_send = 0.0

    def on_target(self, msg):
        self.target = (msg.pose.position.x, msg.pose.position.y, time.monotonic())

    def on_rgb(self, msg):
        self.rgb_msg = msg  # 최신 프레임만 유지

    def on_depth(self, msg):
        self.depth_msg = msg

    # ---------- 공통 ----------
    def step(self):
        now = time.monotonic()
        if self.rgb_msg is not None:
            self.process_camera(now)
        if self.mode == 'chase' and now - self.last_seen > LOST_TIMEOUT:
            self.publish_cmd(0.0, 0.0)
            self.mode = 'approach'
            self.get_logger().info('로봇 카메라에서 RC카 놓침 -> 접근 단계(웹캠 좌표)로 복귀')
        if self.mode == 'approach':
            self.approach()

    def publish_cmd(self, lin, ang):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = lin
        msg.twist.angular.z = ang
        self.cmd_pub.publish(msg)

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

        if dist is not None:
            if self.mode != 'chase':
                self.navigator.cancelTask()
                self.last_goal = None
                self.mode = 'chase'
                self.get_logger().info('로봇 카메라에서 RC카 발견 -> 추적 단계')
            self.last_seen = now
            self.chase(u, frame.shape[1], dist)
            cv2.putText(annotated, f'dist {dist:.2f} m', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif self.mode == 'chase':
            self.publish_cmd(0.0, 0.0)  # 잠깐 놓침: 제자리 대기
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

    def chase(self, u, width, dist):
        offset = (u - width / 2) / (width / 2)  # -1(왼쪽) ~ 1(오른쪽)
        ang = clip(-K_ANG * offset, -MAX_ANG, MAX_ANG)
        lin = clip(K_LIN * (dist - CHASE_DIST), 0.0, MAX_LIN)  # CHASE_DIST 안쪽이면 전진 0 (후진 안 함)
        self.publish_cmd(lin, ang)

    # ---------- 접근 단계 ----------
    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def approach(self):
        if self.target is None or time.monotonic() - self.target[2] > TARGET_TIMEOUT:
            return
        robot = self.robot_xy()
        if robot is None or time.monotonic() - self.last_send < GOAL_PERIOD:
            return
        cx, cy, _ = self.target
        dx, dy = cx - robot[0], cy - robot[1]
        dist = math.hypot(dx, dy)
        if dist < STANDOFF + ARRIVE_MARGIN:  # 도착 (로봇 카메라에 안 보이면 여기서 대기)
            self.navigator.cancelTask()
            self.last_goal = None
            return
        gx, gy = cx - STANDOFF * dx / dist, cy - STANDOFF * dy / dist
        moved = self.last_goal is None or math.hypot(gx - self.last_goal[0], gy - self.last_goal[1]) > GOAL_MOVE_THRESH
        if not (moved or self.navigator.isTaskComplete()):
            return
        yaw = math.atan2(dy, dx)  # RC카 쪽을 바라봄 -> 도착 시 OAK-D 에 잡히도록
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = gx, gy
        goal.pose.orientation.z, goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.navigator.goToPose(goal)
        self.last_goal, self.last_send = (gx, gy), time.monotonic()
        self.get_logger().info(f'RC카 map ({cx:.2f}, {cy:.2f}) -> goal ({gx:.2f}, {gy:.2f})')


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
