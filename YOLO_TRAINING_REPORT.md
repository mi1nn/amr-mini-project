# YOLO 모델 학습 통계 분석 보고서 (car / dummy)

> 작성일: 2026-10-03 · 환경: RTX 3060 Laptop GPU (6GB), Ultralytics 8.4.170
> 데이터셋: `~/Downloads/yolo88` — 2 클래스(car, dummy), train 87장 / valid 25장 / test 12장
> 원본 데이터: `~/rokey_ws/runs/car_dummy_<태그>/stats/`, `~/rokey_ws/runs/compare/`

---

## 1. 요약

- 학습한 모델 6종(YOLOv8n/s, YOLO11n/s, YOLO26n/s)은 **정상 학습 후 test mAP50 0.995, mAP50-95 0.93~0.98**을 기록했습니다.
- **YOLOv8s는 첫 학습(patience=20)에서 21 epoch 만에 조기 종료**되었습니다. 이때 test mAP50-95는 0.4998, dummy 클래스 검출은 0건이었습니다.
  - 원인: 2~3 epoch에 학습이 **발산**(loss 폭증)했고, 회복하는 동안 **epoch 1의 점수를 20 epoch 동안 넘지 못해서** early stopping이 작동했습니다.
  - patience=50으로 다시 학습하자 같은 발산이 똑같이 재현됐지만, 22 epoch부터 회복해서 test mAP50-95 **0.9533**까지 올라갔습니다.
- test mAP50-95 최고 모델은 **YOLO26n (0.9773)**, 추론 속도 최고 모델은 **YOLOv8n (3.70 ms/img, 270 FPS)**입니다.
- ⚠️ test 셋이 12장(박스 24개)뿐이라 mAP50-95가 0.01~0.02 차이 나는 것은 통계적으로 의미 있는 차이로 보기 어렵습니다.

---

## 2. 학습 파라미터 (어떻게 사용되었나)

### 2.1 사용자가 지정한 값 (`TRAIN_ARGS`)

| 파라미터 | v8n | v8s (1차) | v8s (2차) | 11n | 11s | 26n | 26s |
|---|---|---|---|---|---|---|---|
| epochs | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| patience | 20 | **20** | **50** | 20 | 20 | 20 | 20 |
| imgsz | 640 | 640 | 640 | 640 | 640 | 640 | 640 |
| batch | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| workers | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

### 2.2 기본값으로 사용된 주요 값 (모든 모델 동일)

| 파라미터 | 설정값 | 실제 동작 |
|---|---|---|
| optimizer | `auto` | 전체 iteration이 1만 회 미만이라 **AdamW(lr≈0.00167, momentum 0.9)가 자동 선택**됐습니다. 이 때문에 아래 `lr0`, `momentum`은 무시됐습니다. |
| lr0 / lrf | 0.01 / 0.01 | auto 모드에서는 lr0 무시. 실제 최대 lr(results.csv `lr/pg0`)은 **0.001617**입니다. |
| momentum / weight_decay | 0.937 / 0.0005 | |
| warmup_epochs / warmup_bias_lr | 3.0 / 0.1 | |
| cos_lr | False | 선형으로 lr 감소 |
| freeze | None | 전체 레이어를 학습 |
| amp | True | 혼합정밀도(fp16) |
| mosaic / close_mosaic | 1.0 / 10 | 마지막 10 epoch은 mosaic 끔 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | |
| translate / scale / fliplr | 0.1 / 0.5 / 0.5 | |
| seed / deterministic | 0 / True | 같은 설정이면 같은 결과가 재현됩니다(v8s 1·2차의 1~21 epoch 수치가 완전히 같음). |

### 2.3 모델 크기

| 모델 | Params (M) | GFLOPs | best.pt (MB) |
|---|---|---|---|
| YOLOv8n | 3.01 | 8.19 | 6.26 |
| YOLOv8s | 11.14 | 28.64 | 22.53 |
| YOLO11n | 2.59 | 6.50 | 5.48 |
| YOLO11s | 9.43 | 21.67 | 19.19 |
| YOLO26n | 2.50 | 5.89 | 5.39 |
| YOLO26s | 9.95 | 22.74 | 20.32 |

---

## 3. 학습 소요 시간

| 모델 | 설정 epochs | 실제 epochs | best epoch | 조기 종료 | Total Training Time (results.csv) | 실제 소요 시간 (준비·최종 검증 포함) |
|---|---|---|---|---|---|---|
| YOLOv8n | 100 | 100 | 94 | 아니오 | 1분 17초 (77.1 s) | 82.1 s |
| **YOLOv8s (1차, patience 20)** | 100 | **21** | **1** | **예** | 0분 32초 | 37.3 s |
| YOLOv8s (2차, patience 50) | 100 | 100 | 91 | 아니오 | 2분 29초 (148.9 s) | 154.3 s |
| YOLO11n | 100 | 100 | 82 | 아니오 | 1분 21초 (81.5 s) | 87.2 s |
| YOLO11s | 100 | 98 | 78 | 예 | 2분 32초 (152.2 s) | 158.2 s |
| YOLO26n | 100 | 65 | 45 | 예 | 1분 03초 (62.8 s) | 68.8 s |
| YOLO26s | 100 | 91 | 71 | 예 | 2분 51초 (171.4 s) | 175.6 s |

- n 모델은 epoch당 약 0.8~1.0초, s 모델은 약 1.5~1.9초 걸렸습니다.
- 11s, 26n, 26s의 조기 종료는 **이미 수렴한 뒤의 정상적인 종료**입니다(best epoch 이후 20 epoch 동안 개선 없음).

---

## 4. Validation 통계 (valid 25장, best.pt)

### 4.1 전체 지표

| 모델 | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 | best F1 (F1-Confidence) | best F1 conf |
|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 0.9952 | 1.0000 | 0.9976 | 0.995 | 0.995 | **0.9688** | 0.9986 | 0.791 |
| **YOLOv8s (1차)** | **0.4819** | **0.5000** | **0.4908** | **0.5425** | – | **0.4869** | **0.5423** | **0.029** |
| YOLOv8s (2차) | 0.9956 | 1.0000 | 0.9978 | 0.995 | 0.995 | 0.9577 | 0.9987 | 0.819 |
| YOLO11n | 0.9975 | 1.0000 | 0.9987 | 0.995 | 0.995 | 0.9597 | 0.9997 | 0.935 |
| YOLO11s | 0.9954 | 1.0000 | 0.9977 | 0.995 | 0.995 | 0.9331 | 0.9987 | 0.773 |
| YOLO26n | 0.9957 | 1.0000 | 0.9978 | 0.995 | 0.995 | 0.9521 | 0.9987 | 0.806 |
| YOLO26s | 0.9971 | 1.0000 | 0.9986 | 0.995 | 0.995 | 0.9628 | 0.9995 | 0.872 |

### 4.2 클래스별 mAP50-95 (val)

| 모델 | car | dummy |
|---|---|---|
| YOLOv8n | 0.9755 | 0.9620 |
| YOLOv8s (2차) | 0.9602 | 0.9551 |
| YOLO11n | 0.9551 | 0.9643 |
| YOLO11s | 0.9144 | 0.9518 |
| YOLO26n | 0.9438 | 0.9603 |
| YOLO26s | 0.9716 | 0.9540 |

### 4.3 F1-Confidence 해석

- **best F1 conf**는 F1이 가장 높아지는 confidence threshold입니다. 실제로 배포할 때 `conf` 값을 정하는 기준이 됩니다.
- 정상 모델은 모두 0.77~0.94 구간에서 F1이 0.998 이상입니다. confidence를 높게 잡아도 놓치는 객체가 거의 없다는 뜻입니다.
- v8s 1차는 best F1 conf가 **0.029**입니다. 거의 모든 예측의 confidence가 매우 낮아서 신뢰도가 사실상 없는 상태입니다.
- 곡선 그림: `runs/car_dummy_<태그>/stats/07_f1_confidence.png`

### 4.4 Confusion Matrix (val, conf ≥ 0.25)

정상적으로 학습된 6개 모델(v8n, v8s 2차, 11n, 11s, 26n, 26s)은 **모두 아래와 같은 대각 행렬**입니다. 오분류, 오탐(FP), 미탐(FN)이 모두 0입니다.

| 예측 \ 정답 | car | dummy | background |
|---|---|---|---|
| car | **25** | 0 | 0 |
| dummy | 0 | **25** | 0 |
| background (미탐) | 0 | 0 | – |

> v8s 1차의 val confusion matrix는 재학습 때 폴더가 덮어써져서 남아 있지 않습니다. test 결과는 5.3절에 있습니다.
> 그림: `runs/car_dummy_<태그>/stats/05_confusion_matrix.png`

---

## 5. Test Inference 통계 (test 12장, best.pt)

### 5.1 지표 정의

- **% Correct Predictions**: conf ≥ 0.25인 예측 박스 중에서 정답 박스와 IoU ≥ 0.5로 맞은 비율입니다(= precision 성격).
- **탐지율 %**: 정답 박스 중에서 검출된 비율입니다(= recall 성격). % Correct만 보면 놓친 객체를 알 수 없어서 함께 봐야 합니다.
- **Average Confidence**: conf ≥ 0.25 예측 박스들의 평균 confidence입니다.
- **Inference speed**: batch=1, test 이미지 10장 평균입니다. 전처리 + 추론 + 후처리 합계이며 이 PC(RTX 3060 Laptop) 기준입니다.

### 5.2 전체 결과

| 모델 | Precision | Recall | mAP50 | mAP50-95 | best F1 / conf | % Correct | 탐지율 % | Avg Conf | 전처리 ms | 추론 ms | 후처리 ms | **합계 ms/img** | **FPS** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 0.9918 | 1.0000 | 0.995 | 0.9493 | 0.9977 / 0.818 | 100.0 | 100.0 | 0.9336 | 0.49 | 2.75 | 0.46 | **3.70** | **270.0** |
| **YOLOv8s (1차)** | **0.5210** | **0.5417** | **0.5669** | **0.4998** | **0.6116 / 0.046** | 100.0 | **45.8** | **0.7205** | 0.57 | 6.11 | 0.43 | 7.10 | 140.8 |
| YOLOv8s (2차) | 0.9930 | 1.0000 | 0.995 | 0.9533 | 0.9985 / 0.874 | 100.0 | 100.0 | 0.9482 | 0.52 | 5.65 | 0.67 | 6.84 | 146.1 |
| YOLO11n | 0.9948 | 1.0000 | 0.995 | 0.9637 | 0.9992 / 0.923 | 100.0 | 100.0 | **0.9687** | 0.66 | 2.74 | 0.46 | 3.87 | 258.4 |
| YOLO11s | 0.9912 | 1.0000 | 0.995 | 0.9285 | 0.9971 / 0.759 | 100.0 | 100.0 | 0.8900 | 0.52 | 6.44 | 0.48 | 7.43 | 134.5 |
| YOLO26n | 0.9942 | 1.0000 | 0.995 | **0.9773** | 0.9992 / 0.899 | 100.0 | 100.0 | 0.9428 | 0.62 | 2.75 | 0.48 | 3.85 | 259.6 |
| YOLO26s | 0.9950 | 1.0000 | 0.995 | 0.9613 | 0.9997 / 0.902 | 100.0 | 100.0 | 0.9235 | 0.49 | 6.46 | 0.48 | 7.42 | 134.7 |

> 속도는 측정할 때마다 ±0.5 ms 정도 달라집니다(v8s 1차는 `runs/compare`에서 측정한 값, 나머지는 각 run의 `stats/speed.csv` 값. compare 측정에서는 26n이 3.17 ms로 가장 빨랐음).

### 5.3 클래스별 test 결과 (conf ≥ 0.25)

| 모델 | class | 정답 | 예측 | TP | FP | FN | % Correct | 탐지율 % | Avg Conf | mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | car / dummy | 12 / 12 | 12 / 12 | 12 / 12 | 0 / 0 | 0 / 0 | 100 / 100 | 100 / 100 | 0.920 / 0.947 | 0.914 / 0.985 |
| **YOLOv8s (1차)** | car / dummy | 12 / 12 | 11 / **0** | 11 / **0** | 0 / 0 | 1 / **12** | 100 / – | 91.7 / **0.0** | 0.720 / – | 0.877 / **0.122** |
| YOLOv8s (2차) | car / dummy | 12 / 12 | 12 / 12 | 12 / 12 | 0 / 0 | 0 / 0 | 100 / 100 | 100 / 100 | 0.934 / 0.963 | 0.928 / 0.979 |
| YOLO11n | car / dummy | 12 / 12 | 12 / 12 | 12 / 12 | 0 / 0 | 0 / 0 | 100 / 100 | 100 / 100 | 0.964 / 0.974 | 0.941 / 0.987 |
| YOLO11s | car / dummy | 12 / 12 | 12 / 12 | 12 / 12 | 0 / 0 | 0 / 0 | 100 / 100 | 100 / 100 | 0.875 / 0.905 | 0.890 / 0.968 |
| YOLO26n | car / dummy | 12 / 12 | 12 / 12 | 12 / 12 | 0 / 0 | 0 / 0 | 100 / 100 | 100 / 100 | 0.933 / 0.953 | 0.970 / 0.985 |
| YOLO26s | car / dummy | 12 / 12 | 12 / 12 | 12 / 12 | 0 / 0 | 0 / 0 | 100 / 100 | 100 / 100 | 0.920 / 0.927 | 0.953 / 0.969 |

### 5.4 Confusion Matrix (test, conf ≥ 0.25)

**정상 모델 6개 (동일)**

| 예측 \ 정답 | car | dummy | background |
|---|---|---|---|
| car | **12** | 0 | 0 |
| dummy | 0 | **12** | 0 |
| background (미탐) | 0 | 0 | – |

**YOLOv8s 1차 (21 epoch 조기 종료)**

| 예측 \ 정답 | car | dummy | background |
|---|---|---|---|
| car | **11** | 0 | 0 |
| dummy | 0 | **0** | 0 |
| background (미탐) | 1 | **12** | – |

→ dummy는 하나도 검출하지 못했습니다. **% Correct Predictions는 100%로 보이지만**, 예측한 11개가 맞았다는 뜻일 뿐 전체 24개 중 13개를 놓친 상태입니다.

### 5.5 모델 선택 관점 정리

| 목적 | 추천 | 근거 |
|---|---|---|
| 정확도 우선 | YOLO26n | test mAP50-95 0.9773 (최고), 3.85 ms |
| 속도 우선 (로봇 탑재) | YOLOv8n / YOLO11n / YOLO26n | 모두 약 3.7~3.9 ms. n 모델끼리의 차이는 측정 오차 수준 |
| confidence 안정성 | YOLO11n | 평균 confidence 0.969, best F1 conf 0.92 (높은 threshold에서도 안정) |
| s 모델 | 이 데이터 규모에서는 이점 없음 | n 대비 약 2배 느리고 정확도는 같거나 낮음. 학습 초반에 발산하기도 쉬움(6절) |

---

## 6. YOLOv8s가 22 epoch 만에 종료된 원인

### 6.1 결론

**오류로 멈춘 게 아니라 Early Stopping(`patience=20`)으로 정상 종료된 것입니다.** 다만 그렇게 된 근본 원인은 **학습 초반의 발산**입니다.

### 6.2 Early Stopping 판정 방식

- Ultralytics는 매 epoch마다 `fitness = 0.1 × mAP50 + 0.9 × mAP50-95`를 계산하고, 이 값이 가장 높은 epoch를 best로 삼습니다.
- best epoch 이후 `patience` epoch 동안 fitness가 더 높아지지 않으면 학습을 멈춥니다.

| epoch | lr (pg0) | train cls_loss | val cls_loss | mAP50 | mAP50-95 | fitness |
|---|---|---|---|---|---|---|
| **1** | 0.00046 | 2.86 | 1.38 | 0.542 | 0.489 | **0.494 ← best** |
| 2 | 0.00101 | 2.86 | **151.3** | 0.293 | 0.077 | 0.099 |
| 3 | 0.00154 | **21.78** | 122.9 | 0.285 | 0.075 | 0.096 |
| 4 | 0.00162 | 20.35 | 83.0 | 0.310 | 0.087 | 0.110 |
| 5 | 0.00160 | 26.13 | **3164.0** | 0 | 0 | 0 |
| 6 | 0.00158 | **35.44** | 1060.8 | 0.0003 | 0.0001 | ≈0 |
| 7 | 0.00157 | 31.43 | 668.4 | 0.001 | 0.0002 | ≈0 |
| 8~17 | ↓ | 8.6 → 1.4 | **NaN** | ≈0 | ≈0 | ≈0 |
| 18~19 | | 1.39 → 1.30 | 23.5 → 7.6 | ≈0 | ≈0 | ≈0 |
| 20 | | 1.14 | 5.6 | 0.420 | 0.104 | 0.136 |
| 21 | | 1.04 | 18.7 | 0.521 | 0.206 | 0.238 → **조기 종료** |
| *22 (2차 학습)* | | 0.99 | 4.7 | 0.959 | 0.628 | *0.661 → epoch 1 점수를 처음으로 넘음* |

- best가 **epoch 1**이었고, 21 − 1 = **20 epoch 동안 개선이 없어서** epoch 21이 끝난 직후 종료됐습니다. 마지막 최종 검증까지 포함하면 22번째 단계에서 끝난 것처럼 보입니다.
- seed가 고정된 2차 학습(patience=50)에서는 1~21 epoch 수치가 완전히 같았고, **바로 다음 epoch 22에서 epoch 1의 점수를 처음으로 넘었습니다.** 단 1 epoch 차이로 회복 직전에 끊긴 셈입니다.

### 6.3 근본 원인: 학습 초반 발산

| 원인 | 설명 |
|---|---|
| ① 자동 선택된 AdamW의 lr | `optimizer='auto'` → AdamW(lr≈0.00167). warmup으로 lr이 0.0005 → 0.0016까지 오르는 **epoch 2~3에 loss가 폭증**했습니다(train cls_loss 2.9 → 35, val cls_loss 최대 3164, epoch 8~17은 NaN). |
| ② 매우 작은 데이터 | train 87장 / batch 16 → **epoch당 6 iteration**. gradient 노이즈가 커서 큰 lr에 민감합니다. |
| ③ s 모델이 더 취약함 | 같은 설정으로 학습한 n 모델들은 cls_loss 최대 4~5 수준으로 안정적이었습니다. 반면 s 모델들은 사전학습 head가 epoch 1부터 높은 점수(fitness 0.45~0.49)를 냈다가 무너졌습니다. |
| ④ epoch 1의 점수가 높음 | 낮은 lr에서 사전학습 가중치가 아직 망가지지 않은 epoch 1의 점수가 0.494로 높게 나왔습니다. 그래서 회복 후에도 넘어야 할 기준이 높았습니다. |

**모델별 초반 안정성 비교** (같은 기본 설정)

| 모델 | train cls_loss 최대 | val cls_loss 최대 | epoch 1 fitness | epoch 1 점수를 처음 넘은 epoch |
|---|---|---|---|---|
| YOLOv8n | 3.97 | 4.0 | 0.014 | 2 |
| **YOLOv8s** | **35.44** | **3164** | **0.494** | **22** ← patience 20을 초과 |
| YOLO11n | 3.88 | 7.1 | 0.058 | 3 |
| YOLO11s | 12.04 | 139.4 | 0.474 | **18** ← 아슬아슬하게 통과 |
| YOLO26n | 5.37 | 6.1 | 0.034 | 2 |
| YOLO26s | 5.84 | 11.8 | 0.449 | 6 |

→ YOLO11s도 같은 현상을 겪었고 2 epoch 차이로 겨우 살아남았습니다. **기본 설정 + 소규모 데이터 + s 모델 조합에서 구조적으로 생기는 문제**로 볼 수 있습니다.

---

## 7. 파라미터 수정 및 테스트 계획

### 7.1 수정 방향

| 문제 | 파라미터 | 기존 → 변경 | 기대 효과 |
|---|---|---|---|
| 회복 중에 조기 종료 | `patience` | 20 → **50** | 발산 후 회복할 시간 확보 *(검증 완료, 7.3절)* |
| | `epochs` | 100 → **150** | lr을 낮춘 만큼 수렴 시간 확보 |
| lr이 높아서 발산 | `optimizer` | auto(AdamW) → **SGD** | 더 안정적이고, lr0가 실제로 적용됨 |
| | `lr0` | (자동 0.00167) → **0.005** (SGD) | 대안: `optimizer='AdamW', lr0=0.0005` |
| | `cos_lr` | False → **True** | 초반 lr을 유지하다가 끝에서 부드럽게 감소 |
| warmup 구간의 급격한 변화 | `warmup_epochs` | 3 → **5** | lr이 천천히 올라감 |
| | `warmup_bias_lr` | 0.1 → **0.05** | bias 초기 lr을 낮춰 초반 충격 완화 |
| 사전학습 특징 손상 / 과적합 | `freeze` | None → **10** | backbone(0~9 레이어) 고정, head만 학습 |
| 학습 속도 | `cache` | False → **'ram'** | 87장이라 RAM에 올려두면 빠름 |
| 일반화 | `degrees` | 0 → **5** | 약간의 회전 증강 |
| (그래도 발산 시) | `amp` | True → False | fp16 수치 불안정 배제 (1차 학습에서 val cls_loss가 NaN이 된 구간이 있었음) |

### 7.2 단계별 실험 계획 (한 번에 하나씩 변경)

| 실험 | RUN_SUFFIX | 변경 내용 (나머지는 기본값) | 상태 |
|---|---|---|---|
| E0 | (없음) | 기본값, patience 20 | ✅ 완료: 21 epoch 조기 종료, test mAP50-95 0.4998 |
| E1 | (없음) | patience 50 | ✅ 완료: 100 epoch, best 91, test mAP50-95 0.9533 (발산은 그대로 발생) |
| E2 | `_sgd` | E1 + `optimizer='SGD', lr0=0.005` | ⏳ |
| E3 | `_adamw` | E1 + `optimizer='AdamW', lr0=0.0005` | ⏳ |
| E4 | `_warmup` | E2 + `warmup_epochs=5, warmup_bias_lr=0.05` | ⏳ |
| E5 | `_freeze` | E4 + `freeze=10` | ⏳ |
| E6 | `_tuned` | 추천 조합 전체 (노트북에 적용된 값) | ⏳ |
| E7 | `_noamp` | E6 + `amp=False` (E6에서도 발산할 때만) | 선택 |

### 7.3 실행 방법

1. VS Code에서 `2_4_b_yolov26n_train_track_local.ipynb`를 엽니다.
2. **1. 경로 설정** 셀에서 아래 값을 지정합니다.
   ```python
   MODEL_WEIGHTS = 'yolov8s.pt'
   RUN_SUFFIX = '_tuned'      # 실험마다 바꿈 → runs/car_dummy_v8s_tuned/, my_bestv8s_tuned.pt
   ```
3. **4. 학습** 셀의 `TRAIN_ARGS`를 실험에 맞게 수정합니다. 아래는 E6 추천 조합입니다.
   ```python
   TRAIN_ARGS = dict(
       epochs=150, patience=50, imgsz=640, batch=16, workers=2, cache='ram',
       optimizer='SGD', lr0=0.005, cos_lr=True,
       warmup_epochs=5, warmup_bias_lr=0.05,
       freeze=10,
       degrees=5, scale=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=10,
   )
   ```
4. **Restart → Run All**을 실행합니다. 학습, 통계(`stats/summary.txt`), 트래킹까지 자동으로 진행됩니다.
5. 실험이 끝나면 비교 리포트를 만듭니다.
   ```bash
   source ~/venvs/rokey_venv/bin/activate && cd ~/rokey_ws
   python yolo_compare.py car_dummy_v8s car_dummy_v8s_sgd car_dummy_v8s_adamw car_dummy_v8s_tuned
   cat runs/compare/compare_summary.txt
   ```

### 7.4 판정 기준

| 확인 항목 | 확인 위치 | 성공 기준 |
|---|---|---|
| 발산 여부 | 노트북 에폭 로그, `results.csv`의 `train/cls_loss`, `val/cls_loss` | 초반 10 epoch 동안 train cls_loss < 5, val cls_loss가 튀지 않음 |
| mAP 붕괴 여부 | `stats/01_metrics_curve.png` | mAP50-95가 0 근처로 떨어지는 구간이 없음 |
| 수렴 속도 | `train_info.csv`의 best epoch | E1(best 91)보다 빠르게 수렴 |
| 최종 성능 | `overall_metrics.csv` (test) | mAP50-95 ≥ 0.95, 탐지율 100% |
| 신뢰도 | `prediction_stats.csv` Avg Conf, best F1 conf | Avg Conf ≥ 0.93 |
| 비용 | Total Training Time | E1(2분 29초) 대비 크게 늘지 않음 |

> 참고: `freeze=10`과 낮은 lr을 같이 쓰면 수렴이 느려져서 최종 mAP가 약간 낮아질 수 있습니다. E5와 E6 결과를 비교해서 freeze를 유지할지 결정하세요.

---

## 8. 관련 파일

| 파일 | 내용 |
|---|---|
| `runs/car_dummy_<태그>/stats/summary.txt` | 모델별 통계 요약 |
| `runs/car_dummy_<태그>/stats/01~08_*.png` | 지표 곡선, loss, lr, 클래스별 성능, confusion matrix, F1-Confidence, confidence 분포 |
| `runs/compare/compare_summary.txt`, `*.csv`, `*.png` | 모델 간 비교 (v8s 1차 결과 포함) |
| `2_4_b_yolov26n_train_track_local.ipynb` | 학습 노트북 (추천 조합 적용됨) |
| `yolo_stats.py`, `yolo_compare.py` | 통계 / 비교 스크립트 |
