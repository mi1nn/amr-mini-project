"""웹캠 픽셀 -> map 좌표 호모그래피 캘리브레이션 v2 (calibrate_webcam 복사본 + 기록/검증 기능).

사용법은 calibrate_webcam 과 같다: 웹캠 창에서 바닥 점을 클릭한 뒤, map 좌표를 둘 중 하나로 입력 (반복).
  (1) RViz2 'Publish Point'로 같은 점을 클릭
  (2) 로봇을 그 점(로봇 중심 바닥)에 세워 두고 웹캠 창에서 'a' -> 현재 /amcl_pose 사용
키: a=amcl 위치 사용, s=계산+저장, u=마지막 점 취소, q=종료

v1 과 달라진 점
  - 점을 추가/취소할 때마다 CSV(OUT_DIR/calib_points_<시작시각>.csv)에 전체 목록을 다시 기록
    (찍은 순서, 픽셀, map x/y, amcl yaw, amcl 표준편차, 입력 방식)
  - 's' 계산 시 점별 재투영 오차, RANSAC 인라이어 여부, leave-one-out 오차를 출력하고 CSV 에 저장
  - H 는 webcam_H_v2.npy 로 저장 (기존 webcam_H.npy 는 덮어쓰지 않음)
"""
import csv
import math
import os
import threading
import time

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
H_PATH = '/home/mu-06/rokey_ws/webcam_H_v2.npy'  # v1 은 webcam_H.npy
OUT_DIR = '/home/mu-06/rokey_ws/calib_logs'
MIN_POINTS = 4
RANSAC_THRESH = 0.1  # m
CLICKED_TOPIC = '/robot6/clicked_point'  # RViz Publish Point
AMCL_TOPIC = '/robot6/amcl_pose'         # 로봇을 점 위에 세워 두고 사용
# ================================
WIN = 'webcam calibration v2'


def open_webcam():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    return cap


class Calibrator(Node):
    def __init__(self):
        super().__init__('calibrate_webcam_v2')
        self.pix_pts, self.map_pts = [], []
        self.meta = []  # 점마다 (source, yaw_deg, std_x, std_y)
        self.pending = None  # 웹캠에서 클릭했고 map 좌표를 기다리는 픽셀
        self.amcl = None     # (x, y, yaw_deg, std_x, std_y)
        os.makedirs(OUT_DIR, exist_ok=True)
        self.csv_path = os.path.join(OUT_DIR, time.strftime('calib_points_%Y%m%d_%H%M%S.csv'))
        self.errs = None  # 's' 이후 점별 결과 (CSV 에 함께 기록)
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
        self.add_pair(msg.point.x, msg.point.y, 'rviz', None)

    def on_amcl(self, msg):
        p = msg.pose.pose
        yaw = math.degrees(math.atan2(2 * (p.orientation.w * p.orientation.z + p.orientation.x * p.orientation.y),
                                      1 - 2 * (p.orientation.y ** 2 + p.orientation.z ** 2)))
        cov = msg.pose.covariance
        self.amcl = (p.position.x, p.position.y, yaw, math.sqrt(max(cov[0], 0.0)), math.sqrt(max(cov[7], 0.0)))

    def use_amcl(self):
        """'a' 키: 로봇이 pending 픽셀 위치에 서 있을 때 현재 amcl 위치를 map 좌표로 사용"""
        if self.amcl is None:
            self.get_logger().warn(f'{AMCL_TOPIC} 수신 없음: localization/initial pose 확인')
            return
        self.add_pair(self.amcl[0], self.amcl[1], 'amcl', self.amcl)

    def add_pair(self, x, y, source, amcl):
        if self.pending is None:
            self.get_logger().warn('먼저 웹캠 창에서 점을 클릭하세요')
            return
        self.pix_pts.append(self.pending)
        self.map_pts.append((x, y))
        self.meta.append((source,) + ((amcl[2], amcl[3], amcl[4]) if amcl else (None, None, None)))
        self.errs = None
        std = f' / amcl 표준편차 ({amcl[3] * 100:.1f}, {amcl[4] * 100:.1f}) cm' if amcl else ''
        self.get_logger().info(
            f'[{len(self.pix_pts)}] 픽셀 {self.pending} <-> map ({x:.3f}, {y:.3f}) [{source}]{std}')
        self.pending = None
        self.write_csv()

    def undo(self):
        if self.pix_pts:
            self.pix_pts.pop()
            self.map_pts.pop()
            self.meta.pop()
            self.errs = None
            self.write_csv()
        self.pending = None

    def write_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['idx', 'pix_u', 'pix_v', 'map_x', 'map_y', 'source', 'amcl_yaw_deg',
                        'amcl_std_x_m', 'amcl_std_y_m', 'err_cm', 'inlier', 'loo_err_cm'])
            for i, ((u, v), (x, y), m) in enumerate(zip(self.pix_pts, self.map_pts, self.meta)):
                extra = ['', '', ''] if self.errs is None else self.errs[i]
                w.writerow([i + 1, u, v, f'{x:.4f}', f'{y:.4f}', m[0],
                            '' if m[1] is None else f'{m[1]:.1f}',
                            '' if m[2] is None else f'{m[2]:.4f}',
                            '' if m[3] is None else f'{m[3]:.4f}', *extra])

    @staticmethod
    def _err_cm(H, pix, mp):
        proj = cv2.perspectiveTransform(pix.reshape(-1, 1, 2), H).reshape(-1, 2)
        return np.linalg.norm(proj - mp, axis=1) * 100

    def compute_and_save(self):
        n = len(self.pix_pts)
        if n < MIN_POINTS:
            self.get_logger().warn(f'점이 {MIN_POINTS}개 이상 필요합니다 (현재 {n})')
            return
        pix = np.float32(self.pix_pts)
        mp = np.float32(self.map_pts)
        H, mask = cv2.findHomography(pix, mp, cv2.RANSAC, RANSAC_THRESH)
        if H is None:
            self.get_logger().error('호모그래피 계산 실패: 점이 한 직선 위에 있거나 겹칩니다')
            return
        inlier = mask.ravel().astype(bool)
        err = self._err_cm(H, pix, mp)
        # leave-one-out: 점 하나를 뺀 H(최소제곱)로 그 점을 예측한 오차 -> 새 점에 대한 실제 정확도
        loo = np.full(n, np.nan)
        if n > MIN_POINTS:
            for i in range(n):
                keep = np.arange(n) != i
                Hi, _ = cv2.findHomography(pix[keep], mp[keep], 0)
                if Hi is not None:
                    loo[i] = self._err_cm(Hi, pix[i:i + 1], mp[i:i + 1])[0]
        self.errs = [[f'{err[i]:.2f}', int(inlier[i]), '' if np.isnan(loo[i]) else f'{loo[i]:.2f}']
                     for i in range(n)]
        self.write_csv()

        log = self.get_logger().info
        for i in range(n):
            tag = '사용' if inlier[i] else '제외'
            loo_s = '-' if np.isnan(loo[i]) else f'{loo[i]:.1f}'
            log(f'점 {i + 1} [{tag}] 재투영 오차 {err[i]:.1f} cm / leave-one-out {loo_s} cm')
        log(f'인라이어 {int(inlier.sum())}/{n}, 전체 평균 {err.mean():.1f} cm / 최대 {err.max():.1f} cm, '
            f'인라이어 평균 {err[inlier].mean():.1f} cm')
        if not np.isnan(loo).all():
            log(f'leave-one-out 평균 {np.nanmean(loo):.1f} cm / 최대 {np.nanmax(loo):.1f} cm')
        log(f'-> {H_PATH} 저장, 기록: {self.csv_path}')
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
            cv2.putText(frame, f'points: {len(node.pix_pts)}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
