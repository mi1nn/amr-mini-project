"""YOLO 모델별 학습 결과 비교 리포트

여러 학습 결과 폴더의 best.pt를 같은 조건(같은 data, 같은 장비)으로 다시 검증해서
성능/속도/크기/학습 정보를 표와 그래프로 비교한다.
포함 항목: 학습 하이퍼파라미터, 설정/실제 epoch(조기 종료 여부), 총 학습 시간, val/test 성능,
          F1-Confidence, % Correct Predictions, Average Confidence, 추론 속도(batch=1), 모델 크기
사용:  python yolo_compare.py                              (runs/ 아래 학습 결과 전부)
      python yolo_compare.py car_dummy car_dummy_26n      (비교할 폴더만 지정)
      %run yolo_compare.py                                (노트북)
결과:  runs/compare/
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
from ultralytics.utils.torch_utils import get_flops, get_num_params

WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR))
from yolo_report_utils import (IOU_MATCH, M, f1_confidence, fmt_duration, load_run, measure_speed,  # noqa: E402
                               predict_matches, threshold_stats)

RUNS_DIR = WORK_DIR / 'runs'
OUT_DIR = RUNS_DIR / 'compare'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONF_THRESHOLD = 0.25   # % Correct Predictions / Average Confidence / 속도 측정 기준 (yolo_stats.py와 동일)

plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})


def save_show(fig, name):
    """그림 저장 + 노트북이면 화면에 표시"""
    fig.tight_layout()
    fig.savefig(OUT_DIR / name)
    if 'inline' in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)


def find_train_dirs():
    """인자로 받은 폴더들, 없으면 runs/ 아래 results.csv + best.pt가 있는 폴더 전부"""
    args = [a for a in sys.argv[1:] if not a.startswith('-')]  # 노트북 커널 인자(-f ...) 무시
    if args:
        dirs = []
        for a in args:
            cand = next((c for c in (Path(a), RUNS_DIR / a) if (c / 'results.csv').exists()), None)
            if cand is None:
                sys.exit(f'학습 결과 폴더를 찾을 수 없습니다 (results.csv 없음): {a}')
            dirs.append(cand.resolve())
        return dirs
    dirs = sorted(p.parent for p in RUNS_DIR.glob('*/results.csv') if (p.parent / 'weights' / 'best.pt').exists())
    if not dirs:
        sys.exit(f'{RUNS_DIR} 아래에 학습 결과(results.csv + weights/best.pt)가 없습니다')
    return dirs


TRAIN_DIRS = find_train_dirs()
print('비교 대상:', ', '.join(d.name for d in TRAIN_DIRS))

# ---------------------------------------------------------------- 1. 폴더별 기록 수집 + best.pt 재검증
runs, histories, f1_curves = [], {}, {}
summary_rows, per_class_rows, pred_rows = [], [], []
for d in TRAIN_DIRS:
    run = load_run(d, WORK_DIR)
    runs.append(run)
    label = f"{run['name']} ({Path(run['model']).stem})"
    histories[label] = run

    best_pt = d / 'weights' / 'best.pt'
    model = YOLO(str(best_pt))
    try:
        n_params, gflops = get_num_params(model.model), get_flops(model.model, run['args'].get('imgsz', 640))
    except Exception:
        n_params, gflops = np.nan, np.nan

    row = {'run': run['name'], 'model': run['model'], 'imgsz': run['args'].get('imgsz'),
           '설정 epochs': run['set_epochs'], '실제 epochs': run['last_epoch'], 'best epoch': run['best_epoch'],
           'patience': run['patience'], '조기 종료': '예' if run['early_stop'] else '아니오',
           'train time (s)': run['train_time_s'], 'Total Training Time': fmt_duration(run['train_time_s']),
           'params (M)': n_params / 1e6, 'GFLOPs': gflops, 'best.pt (MB)': best_pt.stat().st_size / 1e6}
    for split in ['val', 'test']:
        print(f"[{run['name']}] {split} 검증 중...")
        m = model.val(data=str(run['data_yaml']), split=split, plots=False, verbose=False,
                      project=str(OUT_DIR), name=f"_val_{run['name']}_{split}", exist_ok=True)
        b = m.box
        px, _, f1_mean, best_f1, best_f1_conf = f1_confidence(b)
        row.update({f'{split} precision': b.mp, f'{split} recall': b.mr,
                    f'{split} F1': 2 * b.mp * b.mr / (b.mp + b.mr + 1e-9),
                    f'{split} mAP50': b.map50, f'{split} mAP50-95': b.map,
                    f'{split} best F1': best_f1, f'{split} best F1 conf': best_f1_conf})
        if split == 'test':
            f1_curves[label] = (px, f1_mean, best_f1, best_f1_conf)
            for i, ci in enumerate(m.ap_class_index):
                per_class_rows.append({'run': label, 'class': run['class_names'][ci],
                                       'precision': b.p[i], 'recall': b.r[i],
                                       'mAP50': b.ap50[i], 'mAP50-95': b.ap[i]})

    # % Correct Predictions / Average Confidence (test, IoU>=0.5 매칭)
    print(f"[{run['name']}] test 예측 분석 / 속도 측정 중...")
    preds, gt_counts = predict_matches(model, run['data_cfg'], 'test', run['class_names'])
    t = threshold_stats(preds, gt_counts, CONF_THRESHOLD)
    t.insert(0, 'run', label)
    pred_rows.append(t)
    a = t[t['class'] == 'all'].iloc[0]
    row.update({'test % Correct': a['% Correct Predictions'], 'test 탐지율 %': a['탐지율 %'],
                'test Avg Conf': a['Average Confidence'], 'test Avg Conf (TP)': a['Avg Conf (TP)']})

    # 추론 속도 (batch=1)
    sp = measure_speed(model, run['data_cfg'], 'test', CONF_THRESHOLD)
    row.update({'preprocess ms': sp['preprocess ms'], 'inference ms': sp['inference ms'],
                'postprocess ms': sp['postprocess ms'], 'total ms/img': sp['total ms'], 'FPS': sp['FPS']})
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
per_class = pd.DataFrame(per_class_rows)
pred_stats = pd.concat(pred_rows, ignore_index=True)
# 하이퍼파라미터: 행=항목, 열=학습 폴더 (값이 다른 항목이 위로)
hparams = pd.DataFrame({r['name']: pd.Series(r['hparams'], dtype=object) for r in runs})
hparams['_diff'] = hparams.astype(str).nunique(axis=1) > 1
hparams = hparams.sort_values('_diff', ascending=False, kind='stable')
hp_out = hparams.drop(columns='_diff').rename_axis('hparam')
hp_out.insert(0, '모델별로 다름', hparams['_diff'].map({True: '*', False: ''}))

summary.to_csv(OUT_DIR / 'compare_summary.csv', index=False, encoding='utf-8-sig')
per_class.to_csv(OUT_DIR / 'compare_per_class.csv', index=False, encoding='utf-8-sig')
pred_stats.to_csv(OUT_DIR / 'compare_predictions.csv', index=False, encoding='utf-8-sig')
hp_out.to_csv(OUT_DIR / 'compare_hparams.csv', encoding='utf-8-sig')
labels = list(histories)

# ---------------------------------------------------------------- 2. 그래프
# 2-a. 에폭별 검증 지표 겹쳐 그리기 (점 = 조기 종료 epoch)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, col) in zip(axes.flat, M.items()):
    for label, run in histories.items():
        df = run['df']
        line, = ax.plot(df['epoch'], df[col].rolling(5, min_periods=1).mean(), label=label)
        if run['early_stop']:
            ax.plot(df['epoch'].iloc[-1], df[col].rolling(5, min_periods=1).mean().iloc[-1], 'o', color=line.get_color())
    ax.set_title(f'{name} (smoothed 5, dot = early stop)'); ax.set_xlabel('epoch'); ax.set_ylim(0, 1.02)
axes[0, 0].legend(fontsize=8)
fig.suptitle('Validation metrics per epoch')
save_show(fig, '01_metrics_curve.png')

# 2-b. test 성능 막대그래프
metric_cols = ['precision', 'recall', 'F1', 'mAP50', 'mAP50-95', 'best F1']
fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(metric_cols)), 4.5))
x = np.arange(len(metric_cols) + 2); wbar = 0.8 / len(summary)
for k, (label, (_, r)) in enumerate(zip(labels, summary.iterrows())):
    vals = [r[f'test {c}'] for c in metric_cols] + [r['test % Correct'] / 100, r['test Avg Conf']]
    bars = ax.bar(x + k * wbar - 0.4 + wbar / 2, vals, wbar, label=label)
    ax.bar_label(bars, fmt='%.3f', fontsize=7, rotation=90, padding=2)   # 세로 글자: 모델이 많아도 안 겹침
ax.set_xticks(x, metric_cols + [f'% correct\n@{CONF_THRESHOLD} (/100)', f'avg conf\n@{CONF_THRESHOLD}'])
ax.set_ylim(0, 1.2); ax.set_title('test metrics (best.pt)'); ax.legend(fontsize=8, loc='lower left')
save_show(fig, '02_test_metrics.png')

# 2-c. 속도/크기 vs 정확도
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, xcol in zip(axes, ['total ms/img', 'params (M)']):
    for label, (_, r) in zip(labels, summary.iterrows()):
        ax.scatter(r[xcol], r['test mAP50-95'], s=60)
        ax.annotate(label, (r[xcol], r['test mAP50-95']), textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel(xcol + (' (batch=1)' if 'ms' in xcol else '')); ax.set_ylabel('test mAP50-95')
    ax.set_title(f'test mAP50-95 vs {xcol}')
save_show(fig, '03_speed_size_vs_accuracy.png')

# 2-d. 클래스별 test mAP50-95
pivot = per_class.pivot_table(index='class', columns='run', values='mAP50-95')[labels]
ax = pivot.plot.bar(rot=0, figsize=(max(8, 1.5 * len(pivot) * len(labels)), 4.5))
for c in ax.containers:
    ax.bar_label(c, fmt='%.3f', fontsize=7)
ax.set_ylim(0, 1.1); ax.set_title('per-class test mAP50-95'); ax.legend(fontsize=8)
save_show(ax.figure, '04_per_class_mAP.png')

# 2-e. F1-Confidence 곡선 (test, 전체 클래스 평균)
fig, ax = plt.subplots(figsize=(8, 4.8))
for label, (px, f1_mean, best_f1, best_conf) in f1_curves.items():
    line, = ax.plot(px, f1_mean, lw=2, label=f'{label}: {best_f1:.3f} at {best_conf:.3f}')
    ax.plot(best_conf, best_f1, 'o', color=line.get_color())
ax.axvline(CONF_THRESHOLD, color='gray', ls=':', lw=1)
ax.set(title='F1-Confidence (test, all classes)', xlabel='confidence', ylabel='F1', xlim=(0, 1), ylim=(0, 1.02))
ax.legend(fontsize=8)
save_show(fig, '05_f1_confidence.png')

# 2-f. 학습 epoch / 시간 / 추론 속도
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
y = np.arange(len(summary))
axes[0].barh(y - 0.2, summary['설정 epochs'], 0.4, label='set', color='lightgray')
axes[0].barh(y + 0.2, summary['실제 epochs'], 0.4, label='trained', color='tab:blue')
axes[0].scatter(summary['best epoch'], y + 0.2, color='tab:red', zorder=3, label='best')
axes[0].set(title='epochs (set / trained / best)', xlabel='epoch'); axes[0].legend(fontsize=8)
axes[1].barh(y, summary['train time (s)'] / 60, color='tab:green')
axes[1].bar_label(axes[1].containers[0], fmt='%.1f min', fontsize=8)
axes[1].set(title='total training time', xlabel='min')
parts = ['preprocess ms', 'inference ms', 'postprocess ms']
left = np.zeros(len(summary))
for pcol in parts:
    axes[2].barh(y, summary[pcol], left=left, label=pcol.replace(' ms', ''))
    left += summary[pcol].to_numpy()
for yi, (tot, fps) in enumerate(zip(summary['total ms/img'], summary['FPS'])):
    axes[2].text(tot, yi, f' {tot:.1f} ms ({fps:.0f} FPS)', va='center', fontsize=8)
axes[2].set(title='inference speed (batch=1)', xlabel='ms / image'); axes[2].legend(fontsize=8)
axes[1].margins(x=0.25); axes[2].margins(x=0.45)   # 막대 끝 글자 공간
for ax in axes:
    ax.set_yticks(y, labels)
save_show(fig, '06_training_speed.png')

# ---------------------------------------------------------------- 3. 요약 출력
pd.set_option('display.width', 250); pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', None)
best_row = summary.loc[summary['test mAP50-95'].idxmax()]
fast_row = summary.loc[summary['total ms/img'].idxmin()]
report = [
    f'# YOLO 모델 비교 ({len(summary)}개)',
    f"- test mAP50-95 최고: {best_row['run']} ({best_row['model']}) = {best_row['test mAP50-95']:.4f}",
    f"- 추론 속도 최고: {fast_row['run']} ({fast_row['model']}) = {fast_row['total ms/img']:.2f} ms/img"
    f" ({fast_row['FPS']:.1f} FPS)",
    f'- % Correct Predictions = conf >= {CONF_THRESHOLD} 예측 중 정답 박스와 IoU >= {IOU_MATCH}로 맞은 비율',
    '- 속도는 이 PC에서 batch=1로 측정한 전처리+추론+후처리 시간 (로봇/다른 장비에서는 달라짐)',
    '', '## 학습 정보', summary[['run', 'model', '설정 epochs', '실제 epochs', 'best epoch', 'patience',
                              '조기 종료', 'Total Training Time']].to_string(index=False),
    '', '## Training Hyper Parameters (* = 모델별로 다른 값)', hp_out.to_string(),
    '', '## 모델 크기 / 추론 속도 (batch=1)',
    summary[['run', 'params (M)', 'GFLOPs', 'best.pt (MB)', 'preprocess ms', 'inference ms', 'postprocess ms',
             'total ms/img', 'FPS']].to_string(index=False),
    '', '## val 성능', summary[['run'] + [f'val {c}' for c in ['precision', 'recall', 'F1', 'mAP50', 'mAP50-95',
                                                             'best F1', 'best F1 conf']]].to_string(index=False),
    '', '## test 성능', summary[['run'] + [f'test {c}' for c in ['precision', 'recall', 'F1', 'mAP50', 'mAP50-95',
                                                               'best F1', 'best F1 conf']]].to_string(index=False),
    '', f'## test 예측 분석 (conf >= {CONF_THRESHOLD})',
    summary[['run', 'test % Correct', 'test 탐지율 %', 'test Avg Conf', 'test Avg Conf (TP)']].to_string(index=False),
    '', '## 클래스별 test mAP50-95', pivot.to_string(),
]
(OUT_DIR / 'compare_summary.txt').write_text('\n'.join(report), encoding='utf-8')
print('\n'.join(report))
print(f'\n저장 위치: {OUT_DIR}')
