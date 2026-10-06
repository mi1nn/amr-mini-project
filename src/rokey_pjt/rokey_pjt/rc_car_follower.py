"""웹캠으로 RC카 탐지 -> 호모그래피로 map 좌표 변환 -> TurtleBot4 가 접근.

전제: Nav2 + localization 실행 중, RViz에서 initial pose 지정 완료.
"""
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from ultralytics import YOLO
from visualization_msgs.msg import Marker

from rokey_pjt.calibrate_webcam import H_PATH, open_webcam

# ================================
# 설정 상수
# ================================
MODEL_PATH = '/home/hv-06/rokey_ws/my_best26n.pt'
CONF_THRESHOLD = 0.5
CAR_CLASS = 'car'
STANDOFF = 0.5         # RC카 앞 몇 m 에서 멈출지
ARRIVE_MARGIN = 0.15   # 이 거리 안이면 도착으로 보고 정지
GOAL_PERIOD = 1.0      # goal 갱신 최소 간격(초)
GOAL_MOVE_THRESH = 0.2  # goal 이 이만큼(m) 이상 움직여야 재전송
# ================================


class RcCarFollower(Node):
    def __init__(self):
        super().__init__('rc_car_follower')
        self.H = np.load(H_PATH)
        self.model = YOLO(MODEL_PATH)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(Marker, '/rc_car_marker', 1)
        self.navigator = TurtleBot4Navigator()
        self.navigator.waitUntilNav2Active()
        self.last_goal = None
        self.last_send = 0.0

    def detect_car(self, frame):
        """가장 신뢰도 높은 car -> (바닥 접점 픽셀 (u,v), 주석 이미지). 없으면 (None, 주석 이미지)"""
        r = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        cars = [b for b in r.boxes if r.names[int(b.cls)] == CAR_CLASS]
        if not cars:
            return None, r.plot()
        x1, _, x2, y2 = max(cars, key=lambda b: float(b.conf)).xyxy[0].tolist()
        return ((x1 + x2) / 2, y2), r.plot()  # bbox 하단 중앙 = 바닥에 닿는 점

    def pixel_to_map(self, u, v):
        x, y = cv2.perspectiveTransform(np.float32([[[u, v]]]), self.H)[0, 0]
        return float(x), float(y)

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def publish_marker(self, x, y):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'rc_car', 0, Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.r, m.color.a = 1.0, 1.0
        m.lifetime.sec = 1
        self.marker_pub.publish(m)

    def follow(self, cx, cy):
        robot = self.robot_xy()
        if robot is None or time.monotonic() - self.last_send < GOAL_PERIOD:
            return
        dx, dy = cx - robot[0], cy - robot[1]
        dist = math.hypot(dx, dy)
        if dist < STANDOFF + ARRIVE_MARGIN:  # 도착
            self.navigator.cancelTask()
            self.last_goal = None
            return
        gx, gy = cx - STANDOFF * dx / dist, cy - STANDOFF * dy / dist
        moved = self.last_goal is None or math.hypot(gx - self.last_goal[0], gy - self.last_goal[1]) > GOAL_MOVE_THRESH
        if not (moved or self.navigator.isTaskComplete()):
            return
        yaw = math.atan2(dy, dx)  # RC카 쪽을 바라봄
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
    cap = open_webcam()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)  # TF 수신용. navigator 는 내부에서 자체 spin
            ok, frame = cap.read()
            if not ok:
                continue
            pix, annotated = node.detect_car(frame)
            if pix:
                cx, cy = node.pixel_to_map(*pix)
                node.publish_marker(cx, cy)
                node.follow(cx, cy)
                cv2.putText(annotated, f'map ({cx:.2f}, {cy:.2f})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('RC car follower', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.navigator.cancelTask()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
