import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from ultralytics import YOLO

# ================================
# 설정 상수
# ================================
IMAGE_TOPIC = '/robot6/oakd/rgb/image_raw/compressed'
MODEL_PATH = '/home/hv-06/rokey_ws/my_best26n.pt'
CONF_THRESHOLD = 0.5
# ================================


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.model = YOLO(MODEL_PATH)
        self.latest_msg = None
        self.lock = threading.Lock()

        # 실시간성: BEST_EFFORT + depth 1 → 큐에 쌓이지 않고 항상 최신 프레임만 유지
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, IMAGE_TOPIC, self.image_callback, qos)

    def image_callback(self, msg):
        # 콜백에서는 저장만 → 추론이 느려도 오래된 프레임은 자연스럽게 버려짐
        with self.lock:
            self.latest_msg = msg

    def take_latest(self):
        with self.lock:
            msg, self.latest_msg = self.latest_msg, None
        return msg


def main():
    rclpy.init()
    node = YoloDetector()
    # 수신은 백그라운드 스레드, 추론+imshow는 메인 스레드 (OpenCV GUI는 메인 스레드 권장)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    try:
        while rclpy.ok():
            msg = node.take_latest()
            if msg is not None:
                # cv_bridge 없이 바로 JPEG 디코딩
                frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                result = node.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
                annotated = result.plot()
                fps = 1000.0 / max(sum(result.speed.values()), 1e-3)
                cv2.putText(annotated, f'{fps:.1f} FPS', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow('YOLO Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
