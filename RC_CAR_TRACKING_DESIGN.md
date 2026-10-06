# RC카 추적 터틀봇 설계 메모

> 상태: **설계 초안 (미구현)**. 웹캠 설치 환경과 맵 사용 방식은 아직 정해지지 않았다 → [미정 사항](#5-미정-사항) 참고.

목표: YOLO(`my_best26n.pt`)로 RC카(`car`)를 탐지하면서 터틀봇(robot6)이 따라간다. 로봇 카메라에서 놓치면 외부 웹캠이 RC카 위치를 알려준다.

---

## 1. 로봇 단독 추종 (`car_follower` 노드)

기존 `src/rokey_pjt/rokey_pjt/yolo_detector.py`에 제어만 붙인다. 탐지/제어 노드를 나누지 않는다 (토픽·동기화 부담만 늘어남).

```
/robot6/oakd/rgb/image_raw/compressed ─┐
                                       ├─> car_follower ──> /robot6/cmd_vel
(선택) /robot6/oakd/stereo/image_raw ──┘   ① YOLO → 'car' bbox 1개 선택
                                           ② 오차 → P 제어
                                           ③ 놓침 처리 / 안전장치
```

### ① 타깃 선택
- `car` 중 conf 최고, 또는 직전 bbox와 가장 가까운 것. `dummy`는 무시.

### ② 제어 (P 제어 2개)
- 회전: `err_x = (bbox_cx - W/2) / (W/2)` → `angular.z = -K_ANG * err_x`
- 전진: `err_d = 거리 - TARGET_DIST` (예: 0.6 m) → `linear.x = K_LIN * err_d`
- 최대 속도 clamp, deadband, 후진 금지(`linear.x >= 0`) 권장.

### ③ 놓침 / 안전
- 약 0.5초 미탐지 → 정지, 또는 마지막으로 보인 방향으로 천천히 회전(SEARCH).
- Wi-Fi 영상 지연(수백 ms) 때문에 게인은 작게 시작.
- 종료 시 0 속도 publish.

### 거리 추정: bbox 크기 우선
- `거리 ≈ fx * 실제_차폭(m) / bbox_폭(px)`, 또는 "bbox 폭을 목표 픽셀로 유지"하는 제어.
- 장점: depth 토픽 불필요(비압축 depth는 Wi-Fi로 무겁다), RGB-depth 정렬 문제 없음, `cv_bridge`/numpy 2 문제 회피.
- 단점: 차가 옆으로 돌면 약간 흔들림. 추종 용도엔 대체로 충분.

### depth를 쓴다면
- depth 화면에서 RC카 윤곽이 잘 안 보이는 것은 정상이다. 차가 낮아 뒤 바닥과 깊이 차이가 작고, 0~3 m JET 정규화에서 바닥 그라데이션에 묻힌다.
- 사람이 depth에서 차를 찾을 필요 없음 → **위치는 YOLO bbox, depth는 숫자만** 읽는다.
  - bbox 안쪽 50% 영역의 유효 depth(0/NaN 제외) 중 **하위 20% 백분위**를 거리로 사용 (bbox 안의 뒤쪽 바닥은 더 멀기 때문에 가까운 값이 차).
  - RGB와 depth 해상도가 다르면 bbox 좌표를 비율 변환. 두 `camera_info`의 width/height 먼저 확인. 정렬 안 되어 있으면 화각 오차 있음.
- 검증: depth 클릭 대신 RGB 화면에 bbox + 거리 숫자 오버레이 → 줄자로 0.5/1/1.5 m 비교.

### 확인 사항
- `ros2 topic info /robot6/cmd_vel`로 `Twist`인지 `TwistStamped`인지 확인.
- 로봇 언도킹 상태에서 테스트.

---

## 2. 외부 웹캠 보조 (`webcam_car_locator` 노드)

로봇 카메라에서 RC카를 놓쳤을 때 웹캠이 RC카 위치를 보내준다.

### 핵심 문제: 좌표계 통일
웹캠은 픽셀 좌표만 안다. 로봇이 쓰려면 같은 좌표계로 바꿔야 한다.

| | A. map 좌표 방식 | B. 웹캠 기준 상대좌표 방식 |
|---|---|---|
| 전제 | 맵 + AMCL로 로봇 위치 앎 | 맵 없음 |
| 웹캠 역할 | RC카 탐지 → 바닥 (x, y) 변환 | RC카 + 터틀봇 둘 다 탐지 |
| 로봇 처리 | Nav2로 그 위치 이동 (장애물 회피 포함) | 웹캠 화면에서 방향·거리 계산해 전달 |
| 추가 필요 | 호모그래피 보정 1회 | 터틀봇 위 ArUco 마커 (위치+방향) |

현재 초안은 **A안** 기준. (맵 사용 여부 미정)

### 구성 (A안)
```
[PC + 웹캠]                                       [터틀봇]
webcam_car_locator                                car_follower
  웹캠 프레임 → YOLO → 'car' bbox                   ┌ TRACK : 로봇 카메라로 추종 (1절)
  bbox 하단 중앙 ──H(호모그래피)──> map (x, y)        ├ GOTO  : 웹캠 위치로 Nav2 이동
  publish /rc_car/position ───────────────────────>├ SEARCH: 둘 다 없으면 회전/정지
  (geometry_msgs/PointStamped, frame_id=map)        └ 로봇 위치: /robot6/amcl_pose
```

### 웹캠 노드 상세
1. **탐지**: 같은 `my_best26n.pt`로 시작. 단, 로봇 시점(낮은 높이) 이미지로만 학습했으므로 위에서 내려다보는 웹캠에선 탐지율이 떨어질 수 있음 → 약하면 웹캠 이미지 100~200장 추가 라벨링 후 재학습.
2. **픽셀 → 바닥 좌표**: bbox **하단 중앙** `(cx, y2)` 사용 (바닥 접점이어야 바닥 평면 호모그래피가 맞음. 중심을 쓰면 차 높이만큼 밀림).
   `cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), H)`
3. **보정 (1회)**:
   - 바닥에 테이프로 4개 이상 점 표시.
   - 각 점의 map 좌표 확보 (로봇을 세워 `amcl_pose` 읽기, 또는 RViz "Publish Point").
   - 웹캠 화면에서 같은 점의 픽셀 클릭 (`depth_checker_mouse`의 마우스 콜백 방식 재사용).
   - `H, _ = cv2.findHomography(pixel_pts, map_pts)` → `.npy`로 저장.
4. **전송**: 탐지됐을 때만 5~10 Hz publish. 미탐지 시 보내지 않음 (오래된 위치로 엉뚱하게 가는 것 방지).
   PC와 로봇의 `ROS_DOMAIN_ID`/discovery 설정 일치 필요.

---

## 3. 로봇 노드 상태 머신

```
TRACK  ──(로봇캠 0.5초 미탐지 & 웹캠 위치 1초 이내)──> GOTO
TRACK  ──(로봇캠 미탐지 & 웹캠 위치 오래됨)──────────> SEARCH
GOTO   ──(로봇캠 재탐지)──> Nav2 goal 취소 → TRACK
SEARCH ──(로봇캠 탐지)──> TRACK  /  (웹캠 위치 수신)──> GOTO
```

### GOTO 구현
- `nav2_simple_commander`의 `BasicNavigator.goToPose()` 사용.
- 목표는 차 위치가 아니라 **차 앞 0.5 m** 지점 (충돌 방지).
- goal은 새 위치가 이전 goal에서 **0.3 m 이상** 바뀔 때만 재전송.
- 도착 후 차 방향으로 회전 → 로봇캠에 잡히면 TRACK.

### 최신성 판단은 수신 시각 기준
PC와 로봇 시계가 어긋날 수 있으므로 `header.stamp` 대신 수신 콜백에서 `self.get_clock().now()`를 저장해 비교.

---

## 4. 구현 순서

1. `car_follower`를 TRACK + SEARCH만으로 구현 (`yolo_detector.py` 복사 후 제어 추가).
   바퀴 띄운 상태로 cmd_vel 확인 → 바닥에서 저속 테스트. 거리가 부정확할 때만 depth 추가.
2. `webcam_car_locator`: 탐지 확인 → 호모그래피 보정 → RViz에 Point 띄워 실제 위치와 일치 확인.
3. GOTO 상태 추가, 상태 전환 테스트.

새 노드는 `src/rokey_pjt/setup.py`의 `console_scripts`에 등록 후 재빌드.

---

## 5. 미정 사항

- [ ] **맵 사용 여부**: 만들어둔 맵 + AMCL/Nav2를 쓸지 (A안) 맵 없이 ArUco 상대좌표로 갈지 (B안)
- [ ] **웹캠 설치 환경**: 천장 수직 하향 / 비스듬히 / 설치 높이·커버 범위, 웹캠을 연결할 PC
- [ ] 웹캠 시점에서 기존 모델 탐지율 → 재학습 필요 여부
- [ ] RC카 실제 폭(cm), 목표 추종 거리
- [ ] `/robot6/cmd_vel` 메시지 타입 (`Twist` / `TwistStamped`)
- [ ] RGB / stereo `camera_info` 해상도 및 depth 정렬 여부 (depth 사용 시)
