# pub_sub

ROS 2 Jazzy용 기본 publisher/subscriber 예제입니다.
기존 빈 파일은 Python 모듈로 사용할 수 있도록 `data_publisher.py`,
`data_subscriber.py`, `image_publisher.py`, `image_subscriber.py`로 변경했습니다.

## 빌드

```bash
cd ~/rokey_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pub_sub --symlink-install
source install/setup.bash
```

## 문자열 송수신

각 터미널에서 `source ~/rokey_ws/install/setup.bash`를 실행한 뒤,
아래 명령을 각각 실행합니다.

```bash
ros2 run pub_sub data_publisher
```

```bash
ros2 run pub_sub data_subscriber
```

`/data` 토픽으로 `std_msgs/msg/String` 메시지를 1초마다 보냅니다.
subscriber는 받은 문자열을 터미널에 출력합니다.

## 이미지 송수신

별도 터미널에서 아래 명령을 각각 실행합니다.

```bash
ros2 run pub_sub image_publisher
```

```bash
ros2 run pub_sub image_subscriber
```

`/image` 토픽으로 `sensor_msgs/msg/Image` 메시지를 1초마다 보냅니다.
카메라 없이 320×240 RGB 테스트 이미지를 생성하며, 매번 빨간색 값이 바뀝니다.
subscriber는 이미지 크기, 인코딩, 데이터 바이트 수를 터미널에 출력합니다.
이미지 송수신은 양쪽 모두 sensor data QoS를 사용합니다.

토픽 이름은 ROS 인자로 변경할 수 있습니다. 예를 들어 이미지 subscriber를
실제 카메라에 연결하려면 카메라의 토픽에 맞게 다음과 같이 실행합니다.

```bash
ros2 run pub_sub image_subscriber --ros-args -r image:=/camera/image_raw
```

종료는 각 터미널에서 `Ctrl+C`를 누릅니다.
