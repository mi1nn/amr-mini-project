"""Play a simplified, monophonic opening of Beethoven's Fur Elise."""

import math

from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNote, AudioNoteVector
import rclpy
from rclpy.node import Node


# (Pitch, quarter-note beats). 0.5 = eighth note, 1 = quarter note.
# This is the familiar opening melody, without the piano accompaniment.
FUR_ELISE = (
    ('E5', 0.5), ('D#5', 0.5), ('E5', 0.5), ('D#5', 0.5),
    ('E5', 0.5), ('B4', 0.5), ('D5', 0.5), ('C5', 0.5),
    ('A4', 1.0),
    ('C4', 0.5), ('E4', 0.5), ('A4', 0.5), ('B4', 1.0),
    ('E4', 0.5), ('G#4', 0.5), ('B4', 0.5), ('C5', 1.0),
    ('E4', 0.5),
    ('E5', 0.5), ('D#5', 0.5), ('E5', 0.5), ('D#5', 0.5),
    ('E5', 0.5), ('B4', 0.5), ('D5', 0.5), ('C5', 0.5),
    ('A4', 1.0),
    ('C4', 0.5), ('E4', 0.5), ('A4', 0.5), ('B4', 1.0),
    ('E4', 0.5), ('C5', 0.5), ('B4', 0.5), ('A4', 2.0),
)


def make_score(bpm=100.0):
    """Convert the melody into a sequence of frequency/duration messages."""
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError('bpm must be a finite positive number')
    pitches = {
        'C4': 262, 'E4': 330, 'G#4': 415, 'A4': 440,
        'B4': 494, 'C5': 523, 'D5': 587, 'D#5': 622, 'E5': 659,
    }
    notes = []
    for pitch, beats in FUR_ELISE * 2:
        duration_ns = round(60.0 / bpm * beats * 1_000_000_000)
        sec, nanosec = divmod(duration_ns, 1_000_000_000)
        notes.append(AudioNote(
            frequency=pitches[pitch],
            max_runtime=Duration(sec=sec, nanosec=nanosec),
        ))
    return AudioNoteVector(append=False, notes=notes)


class FurEliseNode(Node):
    """Publish the score once after an audio subscriber connects."""

    def __init__(self):
        super().__init__('fur_elise_node')
        self.declare_parameter('audio_topic', '/robot6/cmd_audio')
        self.declare_parameter('bpm', 100.0)
        self.score = make_score(float(self.get_parameter('bpm').value))
        topic = self.get_parameter('audio_topic').value
        self.publisher_ = self.create_publisher(AudioNoteVector, topic, 10)
        self.timer = self.create_timer(0.5, self.play_once)
        self.get_logger().info(f'오디오 연결 대기: {topic}')

    def play_once(self):
        """Send the complete melody without interrupting it every second."""
        if self.publisher_.get_subscription_count() == 0:
            return
        self.publisher_.publish(self.score)
        self.timer.cancel()
        self.get_logger().info('엘리제를 위하여 도입부 연주 명령 발행 (2회)')


def main(args=None):
    """Start the ROS node; keep it alive until the user presses Ctrl+C."""
    rclpy.init(args=args)
    node = None
    try:
        node = FurEliseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
