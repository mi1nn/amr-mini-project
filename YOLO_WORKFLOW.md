# YOLO 학습 · 통계 · 모델 비교 사용법

`~/rokey_ws`에서 YOLO 모델을 학습하고, 학습 결과 통계를 만들고, 여러 모델을 비교하는 방법을 정리한 문서입니다.

| 순서 | 파일 | 역할 |
|---|---|---|
| 1 | `2_4_b_yolov26n_train_track_local.ipynb` | 모델 1개 학습, 검증, 예측, 트래킹, 통계 (모델마다 반복) |
| 2 | `yolo_stats.py` | 학습 결과 1개의 상세 통계 (1번 노트북 10번 셀에서 자동 실행) |
| 3 | `yolo_compare.py` | 여러 모델의 결과를 같은 조건으로 비교 (모든 모델 학습 후 1회) |
| - | `yolo_report_utils.py` | 위 세 파일이 같이 쓰는 함수 모음 (직접 실행하지 않음, 지우면 안 됨) |

```
[모델 A 학습] ─┐
[모델 B 학습] ─┼─→ runs/car_dummy_<태그>/  (+ train_info.yaml, stats/)
[모델 C 학습] ─┘              │
                              └─→ yolo_compare.py → runs/compare/
```


> 처음이라면 바로 아래 **실행 방법** 순서대로 따라 하면 됩니다. 이후 섹션은 각 파일의 상세 설명입니다.

---

## 실행 방법 (따라하기)

### Step 1. 노트북 열기 (VS Code)
1. 터미널에서 워크스페이스를 VS Code로 엽니다.
   ```bash
   cd ~/rokey_ws && code .
   ```
2. 왼쪽 탐색기에서 `2_4_b_yolov26n_train_track_local.ipynb`를 엽니다.
3. 오른쪽 위 **Select Kernel** → **Python Environments** → `~/venvs/rokey_venv/bin/python`을 선택합니다.
   - 목록에 없으면 **Select Another Kernel...** → **Python Environments** → **Enter interpreter path**에 `~/venvs/rokey_venv/bin/python`을 입력하세요.
   - 커널 이름이 `rokey_venv (Python 3.12)`로 보이면 정상입니다.

### Step 2. 학습할 모델·하이퍼파라미터 정하기
1. **1. 경로 설정** 셀에서 모델을 고릅니다.
   ```python
   MODEL_WEIGHTS = 'yolo26n.pt'   # yolov8n.pt, yolo11n.pt, yolo26n.pt, yolo26s.pt ...
   ```
2. (선택) **4. 학습** 셀의 `TRAIN_ARGS`를 고칩니다. 그대로 두면 epochs=100, patience=20, imgsz=640, batch=16입니다.
3. `Ctrl+S`로 저장합니다.

### Step 3. 학습 + 통계 실행
1. 노트북 위쪽 툴바에서 **Restart**(커널 재시작) → **Run All**을 누릅니다.
   - 이전 실행의 변수가 남지 않도록 꼭 재시작 후 실행하세요.
2. 진행 상황
   - 0~3번 셀: 환경·데이터셋 확인 (몇 초)
   - 4번 셀: 학습. 에폭마다 `[에폭  n/100] box_loss=... | P=..., R=..., mAP50=..., mAP50-95=...`가 출력됩니다.
   - 4번 다음 셀: Epochs(설정/실제/best), 조기 종료 여부, 학습 시간, 하이퍼파라미터가 출력됩니다.
   - 5~9번 셀: 결과 그래프, test 검증, 예측, 트래킹, `my_best<태그>.pt` 저장
   - 10번 셀: 통계 리포트. 요약이 출력되고 그래프가 셀 아래에 표시됩니다.
3. 결과 확인
   ```bash
   cat runs/car_dummy_26n/stats/summary.txt     # 26n의 경우
   ```
   그래프는 VS Code 탐색기에서 `runs/car_dummy_<태그>/stats/*.png`를 클릭하면 볼 수 있습니다.

### Step 4. 다른 모델도 학습
Step 2에서 `MODEL_WEIGHTS`만 바꾸고 Step 3을 반복합니다. 모델마다 `runs/car_dummy_<태그>/`가 따로 생깁니다.

### Step 5. 모델 비교
모든 모델을 학습한 뒤 **둘 중 하나**로 실행합니다.
- 노트북: 11번 셀에서 `RUN_COMPARE = True`로 바꾸고 그 셀만 실행 (셀 왼쪽 ▶ 또는 `Shift+Enter`)
- 터미널:
  ```bash
  source ~/venvs/rokey_venv/bin/activate && cd ~/rokey_ws
  python yolo_compare.py                                          # runs/ 아래 전부
  python yolo_compare.py car_dummy car_dummy_26n car_dummy_11n    # 골라서
  cat runs/compare/compare_summary.txt
  ```

### (참고) 통계만 다시 만들기
학습은 그대로 두고 통계만 다시 만들 때는 터미널에서 실행합니다(GPU로 val/test를 다시 검증하므로 모델당 수십 초 걸림).
```bash
source ~/venvs/rokey_venv/bin/activate && cd ~/rokey_ws
python yolo_stats.py car_dummy_26n      # 폴더 이름 지정
python yolo_stats.py                    # 가장 최근 학습 결과
```

### 자주 나는 문제
| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | 커널이 `rokey_venv`가 아님 → Step 1의 3번에서 커널 다시 선택 |
| `ModuleNotFoundError: No module named 'yolo_report_utils'` | 노트북/스크립트를 `~/rokey_ws`가 아닌 곳에서 실행함, 또는 파일이 지워짐 → `~/rokey_ws`에서 실행 |
| `CUDA out of memory` | `TRAIN_ARGS`의 `batch`를 8(또는 4)로 줄임 |
| 학습 결과 폴더 이름이 이상함 / 이전 모델 결과에 덮어씀 | 커널 재시작 없이 일부 셀만 실행함 → Restart → Run All |
| 노트북에 `MODEL_WEIGHTS`, `TRAIN_ARGS`가 안 보임 | 편집기가 예전 내용으로 저장해 덮어씀 → 노트북을 저장하지 말고 닫았다 다시 열기 |
| `학습 결과 폴더를 찾을 수 없습니다` | `python yolo_stats.py <이름>`의 이름이 `runs/` 아래 폴더와 다름 → `ls runs`로 확인 |
| 그래프 글자가 네모(□)로 나옴 | 한글 폰트 문제. 그래프 라벨은 영어로 되어 있고, 한글 내용은 `summary.txt`/csv에서 확인 |

---

## 0. 실행 환경

- 학습, 통계, 비교는 모두 **`rokey_venv`** 환경에서 실행합니다(ultralytics 설치됨). 노트북 커널도 `rokey_venv`로 선택하세요.
  ```bash
  source ~/venvs/rokey_venv/bin/activate
  cd ~/rokey_ws
  ```
- ROS 노드(`depth_checker_mouse` 등)는 venv를 **끄고** 실행합니다. venv의 NumPy 2.x와 ROS `cv_bridge`(NumPy 1.x용으로 빌드됨)가 충돌하기 때문이에요.
- 노트북을 편집기에 열어 둔 채로 파일이 바뀌면, 다음에 저장할 때 이전 내용으로 덮어써질 수 있습니다. 파일이 바뀌었다면 노트북을 닫았다가 다시 여세요.

---

## 통계 항목 정의

| 항목 | 의미 | 어디서 보나 |
|---|---|---|
| **Training Hyper Parameters** | 노트북 `TRAIN_ARGS`에 넣은 값 (`train_info.yaml`). `train_info.yaml`이 없는 예전 결과는 `args.yaml`의 주요 항목 + 기본값과 다른 값 | stats, compare |
| **Epochs** | 설정 epoch / 실제 학습한 epoch / best epoch (best.pt가 저장된 epoch) | stats, compare |
| **조기 종료** | `실제 < 설정`이고 `실제 - best ≥ patience`이면 조기 종료(EarlyStopping). 몇 epoch에서 끝났는지 함께 표시 | stats, compare |
| **Total Training Time** | `results.csv`의 누적 시간 (학습 + 에폭별 검증). 노트북에서 학습했다면 준비·최종 검증까지 포함한 실제 소요 시간도 표시 | stats, compare |
| **F1-Confidence** | confidence 기준값에 따른 F1 곡선, 최고 F1과 그때의 confidence (클래스 평균) | stats, compare |
| **% Correct Predictions** | conf ≥ 0.25인 예측 박스 중 같은 클래스 정답 박스와 IoU ≥ 0.5로 맞은 비율 (= TP / 예측 수) | stats, compare |
| **탐지율 %** | 정답 박스 중 찾아낸 비율 (= TP / 정답 수, recall) | stats, compare |
| **Average Confidence** | conf ≥ 0.25인 예측 박스들의 평균 confidence (전체 / 맞은 것(TP) / 틀린 것(FP)) | stats, compare |
| **Inference Speed** | 이미지 1장씩(batch=1) 넣었을 때 전처리 + 추론 + 후처리 ms/img와 FPS (처음 2장은 워밍업으로 제외) | stats, compare |

- 기준 confidence(0.25)는 `yolo_stats.py`, `yolo_compare.py` 맨 위의 `CONF_THRESHOLD`에서 바꿀 수 있습니다. 노트북 7번 예측 셀의 `conf=0.25`와 같은 값이에요.
- best.pt는 ultralytics 8.4 기준 **mAP50-95가 가장 높은 epoch**의 가중치입니다.

---

## 1. 학습 노트북: `2_4_b_yolov26n_train_track_local.ipynb`

### 사용법
1. **1. 경로 설정** 셀에서 `MODEL_WEIGHTS`를 바꿉니다.
   ```python
   MODEL_WEIGHTS = 'yolo26n.pt'   # yolov8n.pt, yolo11n.pt, yolo26n.pt, yolo26s.pt ...
   ```
   결과 폴더 이름(`RUN_NAME = car_dummy_<태그>`)과 저장 파일명은 이 값을 따라 자동으로 정해집니다. 태그는 모델 이름에서 `yolo`를 뺀 것입니다(`yolo26n.pt` → `26n`, `yolov8n.pt` → `v8n`, `yolo11n.pt` → `11n`).
2. (선택) **4. 학습** 셀의 `TRAIN_ARGS`에서 하이퍼파라미터를 바꾸거나 추가합니다. 여기 넣은 값이 통계에 "Training Hyper Parameters"로 표시됩니다.
   ```python
   TRAIN_ARGS = dict(
       epochs=100,
       patience=20,     # 이 epoch 수 동안 mAP50-95 개선이 없으면 조기 종료
       imgsz=640,
       batch=16,
       workers=2,
       # lr0=0.005, optimizer='AdamW', mosaic=0.5 ... 처럼 추가 가능
   )
   ```
3. 커널을 재시작한 뒤 노트북 전체를 실행합니다(Restart & Run All).
4. 다른 모델을 학습할 때는 `MODEL_WEIGHTS`만 바꿔서 다시 실행합니다.

### 셀 순서
| 섹션 | 내용 |
|---|---|
| 0. 패키지 설치 / 환경 확인 | ultralytics, CUDA 확인 |
| 1. 경로 설정 | `DATASET_DIR`, `CLASS_NAMES`, **`MODEL_WEIGHTS`**, `RUN_NAME` |
| 2. 데이터셋 확인 | split별 이미지/라벨 수, 클래스별 박스 수, 라벨 시각화 |
| 3. data.yaml 생성 | 절대경로로 된 `custom_data.yaml` 생성 |
| 4. 학습 | **`TRAIN_ARGS`** 로 `model.train(...)`, 매 에폭 loss와 P/R/mAP 요약 출력, 실제 소요 시간 측정 |
| (4 다음 셀) | `train_info.yaml` 저장 + Epochs(설정/실제/best), 조기 종료 여부, 학습 시간, 하이퍼파라미터 출력 |
| 5. 학습 결과 보기 | `results.png`, 혼동행렬 등 |
| 6. 검증 (test 셋) | best.pt로 test mAP 측정 |
| 7. 이미지 예측 | test 이미지 예측 결과 저장 |
| 8. 트래킹 | ByteTrack 영상 트래킹 (8-4 웹캠은 `RUN_WEBCAM = True`일 때만) |
| 9. 학습된 모델 저장 | `my_best<태그>.pt`로 복사 |
| 10. 학습 통계 리포트 | `%run yolo_stats.py {TRAIN_DIR}` 자동 실행 |
| 11. 모델 비교 (선택) | `RUN_COMPARE = True`일 때만 `yolo_compare.py` 실행 |

### Input
| 항목 | 위치 |
|---|---|
| 데이터셋 | `~/Downloads/yolo88/{train,valid,test}/{images,labels}` (`DATASET_DIR`) |
| 사전학습 가중치 | `MODEL_WEIGHTS` (워크스페이스에 없으면 자동 다운로드) |
| 하이퍼파라미터 | 4. 학습 셀의 `TRAIN_ARGS` |

### Output
| 항목 | 위치 |
|---|---|
| 학습 결과 | `runs/car_dummy_<태그>/` (`results.csv`, `args.yaml`, `weights/best.pt`, `last.pt`, 그래프) |
| 학습 정보 | `runs/car_dummy_<태그>/train_info.yaml` (모델, `TRAIN_ARGS`, 실제 소요 시간 `wall_time_s`) |
| test 검증 | `runs/car_dummy_<태그>_test/` |
| 이미지 예측 | `runs/car_dummy_<태그>_predict/` |
| 트래킹 영상 | `runs/car_dummy_<태그>_track/` |
| 최종 모델 | `my_best<태그>.pt` |
| 통계 리포트 | `runs/car_dummy_<태그>/stats/` (10번 셀) |
| 공용 파일 | `custom_data.yaml`, `track_test.mp4` (매번 같은 내용으로 다시 생성) |

### 주의
- **같은 모델을 다시 학습하면 이전 결과를 덮어씁니다**(`exist_ok=True`). 하이퍼파라미터를 바꿔서 비교하려면 `RUN_NAME` 끝에 `_ep200`, `_lr005`처럼 붙여 주세요.
- s, m처럼 큰 모델에서 `CUDA out of memory`가 나면 `TRAIN_ARGS`의 `batch=16`을 `8`로 줄이세요.
- 데이터셋을 바꾸면 `DATASET_DIR`, `CLASS_NAMES`와 함께 `RUN_NAME` 앞부분(`car_dummy`)도 바꿔야 결과가 섞이지 않습니다.
- loss 항목은 모델마다 다릅니다(v8: box/cls/**dfl**, YOLO26: box/cls/**l1**). 에폭 요약 출력은 이를 자동으로 반영해요.
- 기존 v8 결과는 `runs/car_dummy/`, `my_best.pt`에 있습니다. `2_4_b_yolov8_train_track_local.ipynb`로 만든 결과예요.
- `TRAIN_ARGS`가 생기기 전에 학습한 결과(`car_dummy`, `car_dummy_26n`, `car_dummy_11n`)에는 `train_info.yaml`이 없습니다. 그래서 하이퍼파라미터는 `args.yaml` 기준으로 표시되고, 실제 소요 시간은 나오지 않아요.

---

## 2. 학습 통계: `yolo_stats.py`

학습 결과 폴더 **1개**를 분석합니다. 학습 정보를 정리하고, 에폭별 기록을 그래프로 그리고, best.pt를 val/test에서 다시 검증·예측 분석하고, 추론 속도와 라벨 박스 통계를 만듭니다.

### 사용법
노트북 10번 셀에서 자동으로 실행됩니다. 따로 실행할 때는 아래처럼 하세요.
```bash
python yolo_stats.py                    # results.csv가 가장 최근에 바뀐 runs/ 하위 폴더
python yolo_stats.py car_dummy_26n      # runs/ 아래 폴더 이름
python yolo_stats.py runs/car_dummy     # 경로로 지정 (기존 v8 결과)
```
실행하면 첫 줄에 `분석 대상: ...`이 출력되니 원하는 결과가 맞는지 확인하세요.

### Input
| 항목 | 위치 |
|---|---|
| 에폭 기록 | `<학습폴더>/results.csv` |
| 학습 설정 | `<학습폴더>/args.yaml` (모델 이름, data yaml 경로, epochs, patience 등) |
| 사용자 하이퍼파라미터 / 실제 소요 시간 | `<학습폴더>/train_info.yaml` (없으면 `args.yaml`로 대체) |
| 가중치 | `<학습폴더>/weights/best.pt` |
| 데이터셋 | data yaml의 `path` 아래 `{train,valid,test}/{images,labels}` |

### Output: `<학습폴더>/stats/`
| 파일 | 내용 |
|---|---|
| `summary.txt` | **전체 요약**: 학습 정보(하이퍼파라미터, Epochs, 조기 종료, 학습 시간) + 핵심 지표(mAP, F1-Confidence, % Correct, Average Confidence, Inference Speed) + 아래 표 전부 |
| `train_info.csv` | 모델, 설정/실제/best epoch, patience, 조기 종료 여부, Total Training Time, 하이퍼파라미터 |
| `overall_metrics.csv` | val/test 전체 성능: P, R, F1, mAP50, mAP75, mAP50-95, **best F1, best F1 conf, % Correct @0.25, Avg Conf @0.25** |
| `per_class_metrics.csv` | val/test 클래스별 성능 |
| `prediction_stats.csv` | val/test × conf 기준(0.25, best F1 conf) × 클래스(+all): 정답/예측 박스 수, TP/FP/FN, **% Correct Predictions, 탐지율 %, Average Confidence (전체/TP/FP)** |
| `speed.csv` | **Inference Speed** (batch=1): preprocess / inference / postprocess / total ms, FPS |
| `epoch_summary.csv` | 지표·loss별 best / 마지막 / 최적값, 마지막 10 에폭 평균·표준편차 |
| `label_box_stats.csv` | 클래스별 박스 크기 기술통계 |
| `01_metrics_curve.png` | 에폭별 P / R / mAP50 / mAP50-95 (best epoch 빨간 선, 조기 종료 epoch 회색 점선) |
| `02_loss_curve.png` | train/val loss (모델에 맞게 자동으로 항목 선택) |
| `03_lr_time.png` | learning rate, 에폭당 소요 시간 (+ 총 학습 시간) |
| `04_per_class_metrics.png` | 클래스별 성능 (val / test) |
| `05_confusion_matrix.png` | 혼동행렬 (개수 / 정규화) |
| `06_label_box_stats.png` | 클래스·split별 박스 수, 박스 크기, 중심 위치 분포 |
| `07_f1_confidence.png` | **F1-Confidence 곡선** (클래스별 + 평균, 최고 F1 지점 표시) |
| `08_confidence_analysis.png` | 맞은/틀린 예측의 confidence 분포, conf 기준값에 따른 **% Correct / 탐지율 / Average Confidence** |
| `_val_val/`, `_val_test/` | ultralytics 기본 검증 그래프 (F1/P/R/PR 곡선, 혼동행렬) |

---

## 3. 모델 비교: `yolo_compare.py`

여러 학습 결과의 best.pt를 **같은 data, 같은 GPU, 같은 세션**에서 다시 검증해서 성능, 속도, 크기, 학습 정보를 비교합니다. 모든 모델을 학습한 뒤에 실행하세요.

### 사용법
```bash
python yolo_compare.py                                          # runs/ 아래 학습 결과 전부
python yolo_compare.py car_dummy car_dummy_26n car_dummy_11n    # 비교할 폴더만 지정
```
노트북에서는 11번 셀에서 `RUN_COMPARE = True`로 바꾸거나 `%run yolo_compare.py`로 실행합니다. 인자를 주지 않으면 `results.csv`와 `weights/best.pt`가 둘 다 있는 `runs/*` 폴더만 비교 대상이 됩니다(`_test`, `_predict`, `_track`, `compare` 폴더는 자동 제외).

### Input
| 항목 | 위치 |
|---|---|
| 학습 폴더들 | 각 `<학습폴더>/results.csv`, `args.yaml`, `train_info.yaml`(있으면), `weights/best.pt` |
| 데이터셋 | 각 폴더 `args.yaml`의 data yaml |

### Output: `runs/compare/`
| 파일 | 내용 |
|---|---|
| `compare_summary.txt` | **전체 요약**: 1위 모델(정확도/속도), 학습 정보, 하이퍼파라미터, 크기/속도, val/test 성능, 예측 분석, 클래스별 mAP50-95 |
| `compare_summary.csv` | 모델별 전체 수치: 설정/실제/best epoch, patience, 조기 종료, Total Training Time, params(M), GFLOPs, best.pt(MB), 속도(pre/inf/post/total ms, FPS), val/test P·R·F1·mAP·best F1·best F1 conf, test % Correct·탐지율·Avg Conf |
| `compare_hparams.csv` | **Training Hyper Parameters** 표 (행=항목, 열=모델, 모델마다 다른 값은 `*` 표시) |
| `compare_predictions.csv` | 모델 × 클래스별 test 예측 분석 (conf ≥ 0.25) |
| `compare_per_class.csv` | 모델 × 클래스별 test 성능 |
| `01_metrics_curve.png` | 에폭별 검증 지표를 모델별로 겹쳐 그린 그래프 (점 = 조기 종료 epoch) |
| `02_test_metrics.png` | test 성능 막대그래프 (P, R, F1, mAP50, mAP50-95, best F1, % Correct, Avg Conf) |
| `03_speed_size_vs_accuracy.png` | 추론 속도(batch=1)·파라미터 수 대비 test mAP50-95 |
| `04_per_class_mAP.png` | 클래스별 test mAP50-95 |
| `05_f1_confidence.png` | 모델별 **F1-Confidence 곡선** (test, 최고 F1 지점 표시) |
| `06_training_speed.png` | **Epochs**(설정/실제/best), **Total Training Time**, **Inference Speed**(전처리/추론/후처리) |
| `_val_<학습폴더>_{val,test}/` | 검증 작업 폴더 |

### 주의
- 추론 속도는 **이 PC(RTX 3060 Laptop)** 에서 batch=1로 잰 값입니다. 로봇이나 다른 장비에서는 절대값이 달라지니 모델끼리 상대 비교용으로만 보세요. 실행할 때마다 10% 정도 흔들릴 수 있습니다.
- 학습 시간은 학습할 당시의 GPU 상태에 따라 달라집니다(다른 프로그램이 GPU를 같이 썼다면 길어짐).
- 다시 실행하면 `runs/compare/`의 결과를 덮어씁니다.

