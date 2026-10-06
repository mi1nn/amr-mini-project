"""웹캠 픽셀 -> map 좌표 호모그래피 캘리브레이션.

사용법: 웹캠 창에서 바닥 점을 클릭한 뒤, map 좌표를 둘 중 하나로 입력 (반복).
  (1) RViz2 'Publish Point'로 같은 점을 클릭
  (2) 로봇을 그 점(로봇 중심 바닥)에 세워 두고 웹캠 창에서 'a' -> 현재 /amcl_pose 사용
키: a=amcl 위치 사용, s=계산+저장, u=마지막 점 취소, q=종료
"""
import threading

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# ================================
# 설정 상수 (rc_car_follower 와 공유)
# ================================
WEBCAM_INDEX = 2  # 0: 노트북 내장(HP Wide Vision), 2: USB 웹캠
FRAME_SIZE = (640, 480)
H_PATH = '/home/hv-06/rokey_ws/webcam_H.npy'
MIN_POINTS = 4
CLICKED_TOPIC = '/robot6/clicked_point'  # RViz Publish Point
AMCL_TOPIC = '/robot6/amcl_pose'         # 로봇을 점 위에 세워 두고 사용
# ================================
WIN = 'webcam calibration'


def open_webcam():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    return cap


class Calibrator(Node):
    def __init__(self):
        super().__init__('calibrate_webcam')
        self.pix_pts, self.map_pts = [], []
        self.pending = None  # 웹캠에서 클릭했고 map 좌표를 기다리는 픽셀
        self.amcl_xy = None
        self.create_subscription(PointStamped, CLICKED_TOPIC, self.on_point, 10)
        # amcl_pose 는 TRANSIENT_LOCAL 로 발행되므로 QoS 를 맞춤
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseWithCovarianceStamped, AMCL_TOPIC, self.on_amcl, qos)

    def on_mouse(self, event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending = (u, v)
            self.get_logger().info(f'픽셀 ({u}, {v}) 선택 -> 이제 RViz2에서 같은 점을 Publish Point로 클릭')

    def on_point(self, msg):
        if msg.header.frame_id != 'map':
            self.get_logger().warn(f"frame_id가 '{msg.header.frame_id}' 입니다. RViz Fixed Frame을 map으로 하세요")
            return
        self.add_pair(msg.point.x, msg.point.y)

    def on_amcl(self, msg):
        self.amcl_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def use_amcl(self):
        """'a' 키: 로봇이 pending 픽셀 위치에 서 있을 때 현재 amcl 위치를 map 좌표로 사용"""
        if self.amcl_xy is None:
            self.get_logger().warn(f'{AMCL_TOPIC} 수신 없음: localization/initial pose 확인')
            return
        self.add_pair(*self.amcl_xy)

    def add_pair(self, x, y):
        if self.pending is None:
            self.get_logger().warn('먼저 웹캠 창에서 점을 클릭하세요')
            return
        self.pix_pts.append(self.pending)
        self.map_pts.append((x, y))
        self.get_logger().info(f'[{len(self.pix_pts)}] 픽셀 {self.pending} <-> map ({x:.3f}, {y:.3f})')
        self.pending = None

    def undo(self):
        if self.pix_pts:
            self.pix_pts.pop()
            self.map_pts.pop()
        self.pending = None

    def compute_and_save(self):
        if len(self.pix_pts) < MIN_POINTS:
            self.get_logger().warn(f'점이 {MIN_POINTS}개 이상 필요합니다 (현재 {len(self.pix_pts)})')
            return
        pix = np.float32(self.pix_pts)
        mp = np.float32(self.map_pts)
        H, _ = cv2.findHomography(pix, mp, cv2.RANSAC, 0.1)
        if H is None:
            self.get_logger().error('호모그래피 계산 실패: 점이 한 직선 위에 있거나 겹칩니다')
            return
        proj = cv2.perspectiveTransform(pix.reshape(-1, 1, 2), H).reshape(-1, 2)
        err_cm = np.linalg.norm(proj - mp, axis=1) * 100
        for i, e in enumerate(err_cm, 1):
            self.get_logger().info(f'점 {i} 재투영 오차: {e:.1f} cm')
        self.get_logger().info(f'평균 {err_cm.mean():.1f} cm / 최대 {err_cm.max():.1f} cm -> {H_PATH} 저장')
        np.save(H_PATH, H)


def main():
    rclpy.init()
    node = Calibrator()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    cap = open_webcam()
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, node.on_mouse)
    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                continue
            for i, (u, v) in enumerate(node.pix_pts, 1):
                cv2.circle(frame, (u, v), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if node.pending:
                cv2.circle(frame, node.pending, 5, (0, 255, 255), 2)
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                node.compute_and_save()
            elif key == ord('a'):
                node.use_amcl()
            elif key == ord('u'):
                node.undo()
            elif key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
