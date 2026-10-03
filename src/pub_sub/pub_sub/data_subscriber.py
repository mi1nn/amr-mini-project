"""Receive and log string messages."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DataSubscriber(Node):
    """Subscribe to the data topic."""

    def __init__(self):
        """Create the subscription."""
        super().__init__('data_subscriber')
        self.subscription = self.create_subscription(
            String, 'data', self.receive_data, 10)

    def receive_data(self, message):
        """Log each received greeting."""
        self.get_logger().info(f'Received: {message.data}')


def main(args=None):
    """Run the subscriber until interrupted."""
    rclpy.init(args=args)
    node = DataSubscriber()
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
