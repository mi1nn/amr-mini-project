import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNote, AudioNoteVector

class BeepNode(Node):
    def __init__(self):
        super().__init__('beep_node')
        self.publisher_ = self.create_publisher(AudioNoteVector, '/robot6/cmd_audio', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
    def timer_callback(self):
        msg = AudioNoteVector()
        msg.append = False
        msg.notes = [
            AudioNote(
                frequency=880,
                max_runtime=Duration(sec=0, nanosec=200_000_000),
            )
        ]
        self.publisher_.publish(msg)
        self.get_logger().info('비프음 명령 발행')

        
def main(args=None):
    rclpy.init(args=args)
    beep_node = BeepNode()
    rclpy.spin(beep_node)
    beep_node.destroy_node()
    rclpy.shutdown()    
    
if __name__ == '__main__':
    main()
    
