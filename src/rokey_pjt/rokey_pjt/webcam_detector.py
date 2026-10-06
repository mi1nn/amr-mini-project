"""웹캠 YOLO 객체 탐지 -> 호모그래피(webcam_H.npy)로 map 좌표 변환 -> 토픽 발행.

주행 없이 탐지/좌표 변환만 한다 (rc_car_follower 의 탐지 부분만 분리).
bbox 하단 중앙(바닥 접점)을 호모그래피에 넣으므로 바닥 평면 위 좌표만 정확하다.

발행:
  /webcam/detections      vision_msgs/Detection3DArray  (frame_id=map, position.x/y = map 좌표)
  /webcam/object_markers  visualization_msgs/MarkerArray (RViz 표시용)
  /webcam/target_pose/<클래스>  geometry_msgs/PoseStamped (frame_id=map, 해당 클래스에서 신뢰도 가장 높은 1개)
    예: /webcam/target_pose/car, /webcam/target_pose/dummy (해당 클래스가 안 보이면 발행 안 함)
    -> 다른 컴퓨터는 H 파일 없이 필요한 클래스 토픽만 구독해서 Nav2 goal 등으로 쓰면 됨
       (같은 ROS_DOMAIN_ID, 같은 map 사용 전제. orientation 은 단위값이라 방향 정보 없음)

예: ros2 run rokey_pjt webcam_detector --ros-args -p webcam_index:=2 -p classes:="['car']"
"""
import os

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

WS = os.path.expanduser('~/rokey_ws')
# ================================
# 기본값 (모두 ROS 파라미터로 덮어쓸 수 있음)
# ================================
WEBCAM_INDEX = 2  # calibrate_webcam 과 동일: 0 = 노트북 내장, 2 = USB 웹캠
FRAME_SIZE = (640, 480)  # 캘리브레이션 때와 같은 해상도여야 H 가 맞음
H_PATH = os.path.join(WS, 'webcam_H.npy')
MODEL_PATH = os.path.join(WS, 'yolov8nbest.pt')
CONF_THRESHOLD = 0.5
MAP_FRAME = 'map'
# ================================
WIN = 'webcam detector'


class WebcamDetector(Node):
    def __init__(self):
        super().__init__('webcam_detector')
        self.declare_parameter('webcam_index', WEBCAM_INDEX)
        self.declare_parameter('frame_width', FRAME_SIZE[0])
        self.declare_parameter('frame_height', FRAME_SIZE[1])
        self.declare_parameter('h_path', H_PATH)
        self.declare_parameter('model_path', MODEL_PATH)
        self.declare_parameter('conf', CONF_THRESHOLD)
        self.declare_parameter('classes', [''])  # 비어 있으면 모든 클래스. 예: ['car']
        self.declare_parameter('map_frame', MAP_FRAME)
        self.declare_parameter('show_window', True)

        p = self.get_parameter
        self.conf = p('conf').value
        self.classes = {c for c in p('classes').value if c}
        self.map_frame = p('map_frame').value
        self.show_window = p('show_window').value

        self.H = np.load(p('h_path').value)
        self.model = YOLO(p('model_path').value)
        self.cap = self.open_webcam(p('webcam_index').value,
                                    p('frame_width').value, p('frame_height').value)

        self.det_pub = self.create_publisher(Detection3DArray, '/webcam/detections', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/webcam/object_markers', 1)
        # 모델 클래스(car, dummy)마다 토픽 1개. classes 파라미터로 거른 클래스는 만들지 않음
        self.target_pubs = {
            name: self.create_publisher(PoseStamped, f'/webcam/target_pose/{name}', 10)
            for name in self.model.names.values() if not self.classes or name in self.classes}
        self.get_logger().info(
            f'탐지 시작: classes={sorted(self.classes) or "all"}, conf={self.conf}, '
            f'model={os.path.basename(p("model_path").value)}')

    def open_webcam(self, index, width, height):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f'웹캠 index {index} 를 열 수 없음: 연결/webcam_index 파라미터 확인')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        real = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if real != (width, height):
            self.get_logger().warn(f'요청 {width}x{height} 와 실제 해상도 {real} 가 다름: 호모그래피가 어긋날 수 있음')
        return cap

    def pixel_to_map(self, u, v):
        x, y = cv2.perspectiveTransform(np.float32([[[u, v]]]), self.H)[0, 0]
        return float(x), float(y)

    def detect(self, frame):
        """-> ([(class_name, conf, map_x, map_y), ...], 주석 이미지)"""
        r = self.model(frame, conf=self.conf, verbose=False)[0]
        objects = []
        for b in r.boxes:
            name = r.names[int(b.cls)]
            if self.classes and name not in self.classes:
                continue
            x1, _, x2, y2 = b.xyxy[0].tolist()
            mx, my = self.pixel_to_map((x1 + x2) / 2, y2)  # bbox 하단 중앙 = 바닥 접점
            if np.isfinite(mx) and np.isfinite(my):
                objects.append((name, float(b.conf), mx, my))
        return objects, r.plot()

    def publish(self, objects):
        stamp = self.get_clock().now().to_msg()

        arr = Detection3DArray()
        arr.header.stamp, arr.header.frame_id = stamp, self.map_frame
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL  # 사라진 객체의 마커 제거
        markers.markers.append(clear)

        for i, (name, score, x, y) in enumerate(objects):
            d = Detection3D()
            d.header = arr.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = name, score
            d.results.append(hyp)
            d.bbox.center.position.x, d.bbox.center.position.y = x, y
            d.bbox.center.orientation.w = 1.0
            arr.detections.append(d)

            for j, (mtype, z, text) in enumerate(((Marker.SPHERE, 0.1, ''),
                                                  (Marker.TEXT_VIEW_FACING, 0.35, f'{name} {score:.2f}'))):
                m = Marker()
                m.header = arr.header
                m.ns, m.id, m.type, m.action = 'webcam_object', 2 * i + j, mtype, Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.2 if mtype == Marker.SPHERE else 0.15
                m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.2, 0.2, 1.0) if not text else (1.0, 1.0, 1.0, 1.0)
                m.text = text
                m.lifetime.sec = 1
                markers.markers.append(m)

        self.det_pub.publish(arr)
        self.marker_pub.publish(markers)

        for name, pub in self.target_pubs.items():
            same = [o for o in objects if o[0] == name]
            if not same:
                continue
            _, _, x, y = max(same, key=lambda o: o[1])  # 클래스별로 신뢰도 최고 1개
            pose = PoseStamped()
            pose.header = arr.header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.w = 1.0
            pub.publish(pose)

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            ok, frame = self.cap.read()
            if not ok:
                continue
            objects, annotated = self.detect(frame)
            self.publish(objects)
            if self.show_window:
                for k, (name, _, x, y) in enumerate(objects):
                    cv2.putText(annotated, f'{name} map ({x:.2f}, {y:.2f})', (10, 30 + 28 * k),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(WIN, annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def main():
    rclpy.init()
    node = WebcamDetector()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
