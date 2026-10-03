import os, sys
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

bag_path = sys.argv[1]                      # 예: Documents/bags
out_dir  = sys.argv[2] if len(sys.argv) > 2 else 'images'
start    = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # 시작 번호 (기본 0)
topic    = '/robot6/oakd/rgb/image_raw/compressed'

os.makedirs(out_dir, exist_ok=True)
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'),
            rosbag2_py.ConverterOptions('cdr', 'cdr'))
reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

i = 0       # 읽은 메시지 수
saved = 0   # 저장한 이미지 수
while reader.has_next():
    _, data, t = reader.read_next()
    if i % 20 == 0:   # 20프레임마다 1장 저장
        msg = deserialize_message(data, CompressedImage)
        ext = 'png' if 'png' in msg.format else 'jpg'
        with open(os.path.join(out_dir, f'frame_{start + saved:05d}.{ext}'), 'wb') as f:
            f.write(msg.data)
        saved += 1
    i += 1
print(f'{saved}장 저장 완료 → {out_dir}/')