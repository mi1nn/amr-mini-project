"""Publish generated RGB images without requiring a camera."""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImagePublisher(Node):
    """Publish test images on the image topic."""

    def __init__(self):
        """Create the image publisher and periodic timer."""
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(
            Image, 'image', qos_profile_sensor_data)
        self.count = 0
        self.timer = self.create_timer(1.0, self.publish_image)

    def publish_image(self):
        """Send a 320 by 240 RGB image with a changing red channel."""
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'test_camera'
        message.height = 240
        message.width = 320
        message.encoding = 'rgb8'
        message.is_bigendian = 0
        message.step = message.width * 3
        pixel = bytes([(self.count * 20) % 256, 128, 255])
        message.data = pixel * (message.width * message.height)
        self.publisher.publish(message)
        self.get_logger().info(
            f'Publishing image {self.count}: '
            f'{message.width}x{message.height} {message.encoding}')
        self.count += 1


def main(args=None):
    """Run the image publisher until interrupted."""
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
