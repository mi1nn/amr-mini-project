"""웹캠 영상을 CompressedImage 로 발행. 웹캠이 연결된 PC 에서 실행 (rclpy + cv2 만 필요).

실행: python3 webcam_publisher.py            (rokey_pjt 빌드 없이도 가능)
      ros2 run rokey_pjt webcam_publisher     (빌드된 경우)
"""
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

# ================================
# 설정 상수 (FRAME_SIZE 는 calibrate_webcam.py 와 같아야 함)
# ================================
WEBCAM_INDEX = 0
FRAME_SIZE = (640, 480)
IMAGE_TOPIC = '/webcam/image_raw/compressed'
FPS = 15.0
# ================================


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(CompressedImage, IMAGE_TOPIC, qos)
        self.create_timer(1.0 / FPS, self.tick)

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'webcam'
        msg.format = 'jpeg'
        msg.data = cv2.imencode('.jpg', frame)[1].tobytes()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
