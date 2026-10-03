"""YOLO 학습 통계 리포트 생성

results.csv(에폭별 기록)와 best.pt 재검증 결과로 그래프/통계표를 만든다.
모델(v8, 11, 26 ...)마다 loss 항목이 달라도 results.csv의 컬럼을 자동으로 읽는다.
포함 항목: 학습 하이퍼파라미터, 설정/실제 epoch(조기 종료 여부), 총 학습 시간, 에폭별 지표/loss,
          val/test 성능, F1-Confidence, % Correct Predictions, Average Confidence, 추론 속도, 라벨 통계
사용:  python yolo_stats.py runs/car_dummy_26n    (터미널, 학습 결과 폴더 지정)
      python yolo_stats.py car_dummy_26n         (runs/ 아래 폴더 이름만 써도 됨)
      python yolo_stats.py                       (가장 최근에 학습한 runs/ 폴더)
      %run yolo_stats.py {TRAIN_DIR}             (노트북)
결과:  <학습 결과 폴더>/stats/
"""
import sys
from pathlib import Path

import matplotlib
try:
    get_ipython  # noqa: F821  노트북이면 inline 백엔드 유지
except NameError:
    matplotlib.use('Agg')  # 터미널 실행: 창 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR))
from yolo_report_utils import (IOU_MATCH, M, f1_confidence, fmt_duration, load_run, measure_speed,  # noqa: E402
                               predict_matches, run_info_lines, threshold_stats)

RUNS_DIR = WORK_DIR / 'runs'
CONF_THRESHOLD = 0.25   # % Correct Predictions / Average Confidence / 속도 측정 기준 (노트북 예측 conf와 동일)


def find_train_dir():
    """인자로 받은 학습 폴더, 없으면 results.csv가 가장 최근에 갱신된 runs/ 하위 폴더"""
    args = [a for a in sys.argv[1:] if not a.startswith('-')]  # 노트북 커널 인자(-f ...) 무시
    if args:
        for cand in (Path(args[0]), RUNS_DIR / args[0]):
            if (cand / 'results.csv').exists():
                return cand.resolve()
        sys.exit(f'학습 결과 폴더를 찾을 수 없습니다 (results.csv 없음): {args[0]}')
    runs = sorted(RUNS_DIR.glob('*/results.csv'), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f'{RUNS_DIR} 아래에 학습 결과(results.csv)가 없습니다')
    return runs[-1].parent


run = load_run(find_train_dir(), WORK_DIR)
TRAIN_DIR, df = run['dir'], run['df']
DATA_YAML, CLASS_NAMES = run['data_yaml'], run['class_names']
DATASET_DIR = Path(run['data_cfg']['path'])
OUT_DIR = TRAIN_DIR / 'stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"분석 대상: {TRAIN_DIR}  (model={run['model']}, data={DATA_YAML})")

plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})


def save_show(fig, name):
    """그림 저장 + 노트북이면 화면에 표시"""
    fig.tight_layout()
    fig.savefig(OUT_DIR / name)
    if 'inline' in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- 0. 학습 정보 (하이퍼파라미터 / epoch / 시간)
train_info = pd.DataFrame(
    [('model', run['model']), ('설정 epochs', run['set_epochs']), ('실제 학습 epochs', run['last_epoch']),
     ('best epoch', run['best_epoch']), ('patience', run['patience']),
     ('조기 종료', '예' if run['early_stop'] else '아니오'), ('종료 상태', run['status']),
     ('Total Training Time (s)', round(run['train_time_s'], 1)),
     ('Total Training Time', fmt_duration(run['train_time_s'])),
     ('실제 소요 시간 (s, 노트북 측정)', run['wall_time_s'])]
    + [(f'hparam: {k}', v) for k, v in run['hparams'].items()],
    columns=['항목', '값'])
train_info.to_csv(OUT_DIR / 'train_info.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 1. 에폭별 기록
best_ep = run['best_epoch']
best = df.loc[df['epoch'] == best_ep].iloc[0]
# loss 항목은 모델마다 다름 (v8: box/cls/dfl, YOLO26: box/cls/l1 ...)
LOSSES = [c.split('/', 1)[1] for c in df.columns if c.startswith('train/') and c.endswith('_loss')]


def mark_epochs(ax):
    """best epoch(빨강), 조기 종료 epoch(회색) 표시"""
    ax.axvline(best_ep, color='tab:red', ls='--', lw=1, label=f'best epoch {best_ep}')
    if run['early_stop']:
        ax.axvline(run['last_epoch'], color='gray', ls=':', lw=1, label=f"early stop {run['last_epoch']}")


# 1-a. 성능 지표 곡선
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, col) in zip(axes.flat, M.items()):
    ax.plot(df['epoch'], df[col], color='tab:blue', alpha=0.35, label='raw')
    ax.plot(df['epoch'], df[col].rolling(5, min_periods=1).mean(), color='tab:blue', label='smoothed (5)')
    mark_epochs(ax)
    ax.set_title(f'{name}  (max {df[col].max():.3f} @ ep {int(df.loc[df[col].idxmax(), "epoch"])})')
    ax.set_xlabel('epoch'); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
fig.suptitle(f"Validation metrics per epoch  (epochs {run['last_epoch']}/{run['set_epochs']})")
save_show(fig, '01_metrics_curve.png')

# 1-b. loss 곡선 (train vs val)
fig, axes = plt.subplots(1, len(LOSSES), figsize=(5 * len(LOSSES), 4.2), squeeze=False)
for ax, loss in zip(axes.flat, LOSSES):
    ax.plot(df['epoch'], df[f'train/{loss}'], label='train')
    if f'val/{loss}' in df:
        ax.plot(df['epoch'], df[f'val/{loss}'], label='val')
    mark_epochs(ax)
    ax.set_title(loss); ax.set_xlabel('epoch'); ax.legend(fontsize=8)
fig.suptitle('Loss per epoch')
save_show(fig, '02_loss_curve.png')

# 1-c. learning rate + 에폭 시간
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
ax1.plot(df['epoch'], df['lr/pg0']); ax1.set_title('learning rate'); ax1.set_xlabel('epoch')
ax2.plot(df['epoch'], df['time'].diff().fillna(df['time']), color='tab:green')
ax2.set_title(f"time per epoch (s)  - total {run['train_time_s'] / 60:.1f} min"); ax2.set_xlabel('epoch')
save_show(fig, '03_lr_time.png')

# 1-d. 에폭 통계 요약표
rows = []
loss_cols = {f'{split} {loss}': f'{split}/{loss}' for loss in LOSSES for split in ('train', 'val')
             if f'{split}/{loss}' in df}
for name, col in {**M, **loss_cols}.items():
    s = df[col]
    is_loss = 'loss' in name
    opt_idx = s.idxmin() if is_loss else s.idxmax()
    rows.append({'항목': name, 'best epoch 값': best[col], '마지막 epoch 값': s.iloc[-1],
                 '최적값(지표=최고, loss=최저)': s[opt_idx], '최적 epoch': int(df.loc[opt_idx, 'epoch']),
                 '마지막 10 epoch 평균': s.tail(10).mean(), '마지막 10 epoch 표준편차': s.tail(10).std()})
epoch_summary = pd.DataFrame(rows)
epoch_summary.to_csv(OUT_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 2. best.pt 재검증 (valid / test)
model = YOLO(str(TRAIN_DIR / 'weights' / 'best.pt'))
per_class_rows, overall_rows, cms, f1_curves = [], [], {}, {}
for split in ['val', 'test']:
    m = model.val(data=str(DATA_YAML), split=split, plots=True, verbose=False,
                  project=str(OUT_DIR), name=f'_val_{split}', exist_ok=True)
    b = m.box
    px, f1_cls, f1_mean, best_f1, best_f1_conf = f1_confidence(b)
    f1_curves[split] = (px, f1_cls, f1_mean, best_f1, best_f1_conf, list(m.ap_class_index))
    overall_rows.append({'split': split, 'precision': b.mp, 'recall': b.mr,
                         'F1': 2 * b.mp * b.mr / (b.mp + b.mr + 1e-9), 'mAP50': b.map50,
                         'mAP75': b.map75, 'mAP50-95': b.map,
                         'best F1 (F1-Confidence)': best_f1, 'best F1 conf': best_f1_conf})
    cm = m.confusion_matrix.matrix  # (nc+1, nc+1), 행=예측, 열=정답, 마지막=background
    cms[split] = cm
    # ultralytics 기본 혼동행렬 그림도 같이 저장됨: stats/_val_{split}/confusion_matrix*.png
    for i, ci in enumerate(m.ap_class_index):
        per_class_rows.append({'split': split, 'class': CLASS_NAMES[ci],
                               'GT boxes': int(cm[:, ci].sum()),
                               'precision': b.p[i], 'recall': b.r[i], 'F1': b.f1[i],
                               'mAP50': b.ap50[i], 'mAP50-95': b.ap[i]})
overall = pd.DataFrame(overall_rows)
per_class = pd.DataFrame(per_class_rows)

# 2-a. 클래스별 막대그래프
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
metrics_cols = ['precision', 'recall', 'F1', 'mAP50', 'mAP50-95']
for ax, split in zip(axes, ['val', 'test']):
    sub = per_class[per_class['split'] == split].set_index('class')[metrics_cols]
    x = np.arange(len(metrics_cols)); wbar = 0.8 / len(sub)
    for k, (cls, vals) in enumerate(sub.iterrows()):
        bars = ax.bar(x + k * wbar - 0.4 + wbar / 2, vals.values, wbar, label=cls)
        ax.bar_label(bars, fmt='%.2f', fontsize=7)
    ax.set_xticks(x, metrics_cols); ax.set_ylim(0, 1.1); ax.set_title(f'per-class metrics ({split})'); ax.legend()
save_show(fig, '04_per_class_metrics.png')

# 2-b. 혼동행렬 (개수 / 정규화)
labels = CLASS_NAMES + ['background']
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for r, split in enumerate(['val', 'test']):
    cm = cms[split]
    cm_norm = cm / np.clip(cm.sum(0, keepdims=True), 1e-9, None)  # 정답(열) 기준 정규화
    for c, (mat, title, fmt) in enumerate([(cm, 'count', '{:.0f}'), (cm_norm, 'normalized (by true)', '{:.2f}')]):
        ax = axes[r, c]
        ax.imshow(mat, cmap='Blues', vmin=0); ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt.format(mat[i, j]), ha='center', va='center',
                        color='white' if mat[i, j] > mat.max() / 2 else 'black')
        ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel('True'); ax.set_ylabel('Predicted'); ax.set_title(f'{split} confusion matrix - {title}')
save_show(fig, '05_confusion_matrix.png')

# 2-c. F1-Confidence 곡선
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, split in zip(axes, ['val', 'test']):
    px, f1_cls, f1_mean, best_f1, best_f1_conf, cls_idx = f1_curves[split]
    for k, ci in enumerate(cls_idx):
        ax.plot(px, f1_cls[k], lw=1, alpha=0.7, label=CLASS_NAMES[ci])
    ax.plot(px, f1_mean, color='black', lw=2.5, label=f'all classes {best_f1:.3f} at {best_f1_conf:.3f}')
    ax.axvline(best_f1_conf, color='tab:red', ls='--', lw=1)
    ax.axvline(CONF_THRESHOLD, color='gray', ls=':', lw=1, label=f'conf {CONF_THRESHOLD}')
    ax.set(title=f'F1-Confidence ({split})', xlabel='confidence', ylabel='F1', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
save_show(fig, '07_f1_confidence.png')

# ---------------------------------------------------------------- 3. 예측 분석 (% Correct Predictions / Average Confidence)
# 예측 박스가 같은 클래스 정답 박스와 IoU >= 0.5 로 매칭되면 '맞은 예측(TP)'
pred_rows, preds_by_split = [], {}
for split in ['val', 'test']:
    preds, gt_counts = predict_matches(model, run['data_cfg'], split, CLASS_NAMES)
    preds_by_split[split] = (preds, gt_counts)
    best_f1_conf = f1_curves[split][4]
    for thr in sorted({CONF_THRESHOLD, round(best_f1_conf, 3)}):
        t = threshold_stats(preds, gt_counts, thr)
        t.insert(0, 'split', split)
        pred_rows.append(t)
pred_stats = pd.concat(pred_rows, ignore_index=True)
pred_stats.to_csv(OUT_DIR / 'prediction_stats.csv', index=False, encoding='utf-8-sig')
for split in ['val', 'test']:
    a = pred_stats[(pred_stats['split'] == split) & (pred_stats['class'] == 'all')
                   & (pred_stats['conf 기준'] == CONF_THRESHOLD)].iloc[0]
    overall.loc[overall['split'] == split, f'% Correct @{CONF_THRESHOLD}'] = a['% Correct Predictions']
    overall.loc[overall['split'] == split, f'Avg Conf @{CONF_THRESHOLD}'] = a['Average Confidence']
overall.to_csv(OUT_DIR / 'overall_metrics.csv', index=False, encoding='utf-8-sig')
per_class.to_csv(OUT_DIR / 'per_class_metrics.csv', index=False, encoding='utf-8-sig')

# 3-a. confidence 분포 (TP/FP) + 기준값에 따른 % Correct / 탐지율 / Average Confidence
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
thr_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
for r, split in enumerate(['val', 'test']):
    preds, gt_counts = preds_by_split[split]
    p = preds[preds['conf'] >= 0.05]
    ax = axes[r, 0]
    bins = np.linspace(0.05, 1, 20)
    ax.hist(p.loc[p['tp'], 'conf'], bins=bins, alpha=0.7, label='correct (TP)', color='tab:blue')
    ax.hist(p.loc[~p['tp'], 'conf'], bins=bins, alpha=0.7, label='wrong (FP)', color='tab:orange')
    ax.axvline(CONF_THRESHOLD, color='gray', ls=':', lw=1)
    ax.set(title=f'prediction confidence ({split}, conf >= 0.05)', xlabel='confidence', ylabel='boxes')
    ax.legend(fontsize=8)
    curve = pd.concat([threshold_stats(preds, gt_counts, t).query("`class` == 'all'") for t in thr_grid])
    ax = axes[r, 1]
    ax.plot(curve['conf 기준'], curve['% Correct Predictions'], marker='.', label='% Correct Predictions')
    ax.plot(curve['conf 기준'], curve['탐지율 %'], marker='.', label='detection rate % (recall)')
    ax.plot(curve['conf 기준'], curve['Average Confidence'] * 100, marker='.', label='Average Confidence x100')
    ax.axvline(CONF_THRESHOLD, color='gray', ls=':', lw=1)
    ax.axvline(f1_curves[split][4], color='tab:red', ls='--', lw=1, label='best F1 conf')
    ax.set(title=f'by confidence threshold ({split})', xlabel='confidence threshold', ylabel='%', ylim=(0, 102))
    ax.legend(fontsize=8)
save_show(fig, '08_confidence_analysis.png')

# ---------------------------------------------------------------- 4. 추론 속도 (batch=1)
speed = measure_speed(model, run['data_cfg'], 'test', CONF_THRESHOLD)
speed_df = pd.DataFrame([{'conf': CONF_THRESHOLD, **speed}])
speed_df.to_csv(OUT_DIR / 'speed.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 5. 라벨 박스 통계 (데이터셋)
box_rows = []
for split in ['train', 'valid', 'test']:
    for lf in (DATASET_DIR / split / 'labels').glob('*.txt'):
        for line in lf.read_text().splitlines():
            if line.strip():
                c, xc, yc, bw, bh = map(float, line.split()[:5])
                box_rows.append({'split': split, 'class': CLASS_NAMES[int(c)], 'xc': xc, 'yc': yc,
                                 'w': bw, 'h': bh, 'area': bw * bh, 'aspect(w/h)': bw / max(bh, 1e-9)})
boxes = pd.DataFrame(box_rows)
box_stats = boxes.groupby('class')[['w', 'h', 'area', 'aspect(w/h)']].describe().round(4)
box_stats.to_csv(OUT_DIR / 'label_box_stats.csv', encoding='utf-8-sig')
box_count = boxes.pivot_table(index='class', columns='split', values='w', aggfunc='count', fill_value=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
box_count[['train', 'valid', 'test']].plot.bar(ax=axes[0], rot=0); axes[0].set_title('box count per class/split')
for cls, g in boxes.groupby('class'):
    axes[1].scatter(g['w'], g['h'], s=10, alpha=0.6, label=cls)
    axes[2].scatter(g['xc'], g['yc'], s=10, alpha=0.6, label=cls)
axes[1].set(title='box size (normalized)', xlabel='width', ylabel='height'); axes[1].legend()
axes[2].set(title='box center position', xlabel='x', ylabel='y', xlim=(0, 1), ylim=(1, 0)); axes[2].legend()
save_show(fig, '06_label_box_stats.png')

# ---------------------------------------------------------------- 6. 요약 출력
pd.set_option('display.width', 250); pd.set_option('display.float_format', '{:.4f}'.format)
test_row = overall[overall['split'] == 'test'].iloc[0]
summary = [
    f"# YOLO 학습 통계 ({TRAIN_DIR.name}, model={run['model']})",
    '', '## 학습 정보', *run_info_lines(run),
    '', '## 핵심 지표 (test, best.pt)',
    f"- mAP50 / mAP50-95: {test_row['mAP50']:.4f} / {test_row['mAP50-95']:.4f}",
    f"- F1-Confidence: 최고 F1 {test_row['best F1 (F1-Confidence)']:.4f} (confidence {test_row['best F1 conf']:.3f})",
    f"- % Correct Predictions @conf {CONF_THRESHOLD}: {test_row[f'% Correct @{CONF_THRESHOLD}']:.2f}%"
    f" (예측 박스 중 IoU>={IOU_MATCH}로 맞은 비율)",
    f"- Average Confidence @conf {CONF_THRESHOLD}: {test_row[f'Avg Conf @{CONF_THRESHOLD}']:.4f}",
    f"- Inference Speed (batch=1, {speed['images']}장 평균): preprocess {speed['preprocess ms']:.2f} + inference "
    f"{speed['inference ms']:.2f} + postprocess {speed['postprocess ms']:.2f} = {speed['total ms']:.2f} ms/img"
    f" ({speed['FPS']:.1f} FPS)",
    '', '## 전체 성능 (best.pt)', overall.to_string(index=False),
    '', '## 클래스별 성능', per_class.to_string(index=False),
    '', f'## 예측 분석 (IoU>={IOU_MATCH} 매칭, conf 기준 {CONF_THRESHOLD} / best F1 conf)',
    pred_stats.to_string(index=False),
    '', '## 에폭 기록 요약', epoch_summary.to_string(index=False),
    '', '## 라벨 박스 수', box_count.to_string(),
]
(OUT_DIR / 'summary.txt').write_text('\n'.join(summary), encoding='utf-8')
print('\n'.join(summary))
print(f'\n저장 위치: {OUT_DIR}')
