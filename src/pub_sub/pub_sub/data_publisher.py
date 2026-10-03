"""Publish a numbered greeting once per second."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DataPublisher(Node):
    """Publish string messages on the data topic."""

    def __init__(self):
        """Create the publisher and periodic timer."""
        super().__init__('data_publisher')
        self.publisher = self.create_publisher(String, 'data', 10)
        self.count = 0
        self.timer = self.create_timer(1.0, self.publish_data)

    def publish_data(self):
        """Send the next greeting."""
        message = String()
        message.data = f'Hello ROS 2! {self.count}'
        self.publisher.publish(message)
        self.get_logger().info(f'Publishing: {message.data}')
        self.count += 1


def main(args=None):
    """Run the publisher until interrupted."""
    rclpy.init(args=args)
    node = DataPublisher()
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
