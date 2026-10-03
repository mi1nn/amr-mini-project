# 엘리제를 위하여 — 비프음용 도입부 악보

베토벤의 「엘리제를 위하여」에서 익숙한 도입부를 반주 없이 단선율로
편곡했습니다. 전곡이 아닌 도입부이며, 아래 악보를 두 번 연주합니다.

기본 속도는 ♩ = 100입니다. `음이름:박자`에서 `0.5`는 8분음표,
`1`은 4분음표, `2`는 2분음표입니다. C4는 가운데 도, E5는 그 위의 미,
`#`는 올림표입니다. 줄바꿈은 선율 구분이며 마디 구분은 아닙니다.

```text
E5:0.5 D#5:0.5 E5:0.5 D#5:0.5 E5:0.5 B4:0.5 D5:0.5 C5:0.5 A4:1
C4:0.5 E4:0.5 A4:0.5 B4:1
E4:0.5 G#4:0.5 B4:0.5 C5:1 E4:0.5

E5:0.5 D#5:0.5 E5:0.5 D#5:0.5 E5:0.5 B4:0.5 D5:0.5 C5:0.5 A4:1
C4:0.5 E4:0.5 A4:0.5 B4:1
E4:0.5 C5:0.5 B4:0.5 A4:2
```

첫 선율의 계이름: **미 레♯ 미 레♯ 미 시 레 도 라**.

## 실행

워크스페이스에서 다음 명령을 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select turtlebot4_beep
source install/setup.bash
ros2 run turtlebot4_beep fur_elise
```

속도나 로봇 토픽을 변경할 수 있습니다.

```bash
ros2 run turtlebot4_beep fur_elise --ros-args -p bpm:=80.0 -p audio_topic:=/robot6/cmd_audio
```

기존 `beep_node`는 종료한 뒤 실행하세요. 두 노드가 동시에 실행되면
기존 비프 명령이 곡을 중간에 끊습니다. 새 노드는 오디오 구독자가 연결되면
악보를 한 번 전송합니다. 연주 후에도 대기하므로 Ctrl+C로 종료합니다.

악보 데이터는 `turtlebot4_beep/fur_elise_node.py`의 `FUR_ELISE`에서
수정할 수 있습니다. 주파수와 재생 시간 배열을 사용하는 방식은
[Create 3 오디오 API](https://iroboteducation.github.io/create3_docs/api/ui/)를
따릅니다.
