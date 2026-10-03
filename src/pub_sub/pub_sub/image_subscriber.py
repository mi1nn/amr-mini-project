"""Receive images and log their metadata."""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageSubscriber(Node):
    """Subscribe to the image topic using sensor data QoS."""

    def __init__(self):
        """Create the image subscription."""
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image, 'image', self.receive_image, qos_profile_sensor_data)

    def receive_image(self, message):
        """Log the size, encoding, and byte count of the received image."""
        self.get_logger().info(
            f'Received image: {message.width}x{message.height} '
            f'{message.encoding}, {len(message.data)} bytes')


def main(args=None):
    """Run the image subscriber until interrupted."""
    rclpy.init(args=args)
    node = ImageSubscriber()
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
