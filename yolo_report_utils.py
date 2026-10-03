"""yolo_stats.py / yolo_compare.py / 학습 노트북이 같이 쓰는 함수

- load_run         : 학습 폴더 1개의 기록(results.csv, args.yaml, train_info.yaml) 정리
                     → 하이퍼파라미터, 설정/실제 epoch, 조기 종료 여부, 학습 시간
- f1_confidence    : model.val() 결과에서 F1-Confidence 곡선과 최고 F1 / 그때의 confidence
- predict_matches  : split 이미지 예측 → 정답 라벨과 IoU 매칭 (TP/FP 판정 + confidence)
- threshold_stats  : confidence 기준값별 % Correct Predictions, 탐지율, Average Confidence
- measure_speed    : batch=1 추론 속도 (로봇에서 프레임 하나씩 넣는 상황과 같음)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TRAIN_INFO = 'train_info.yaml'   # 학습 노트북이 저장하는 사용자 하이퍼파라미터 + 실제 소요 시간
IOU_MATCH = 0.5                  # 예측 박스를 '정답'으로 볼 IoU 기준 (mAP50과 같은 기준)

M = {'precision': 'metrics/precision(B)', 'recall': 'metrics/recall(B)',
     'mAP50': 'metrics/mAP50(B)', 'mAP50-95': 'metrics/mAP50-95(B)'}

# 항상 보여줄 주요 하이퍼파라미터 (train_info.yaml이 없는 예전 학습 결과용)
CORE_HPARAMS = ['epochs', 'patience', 'batch', 'imgsz', 'optimizer', 'lr0', 'lrf', 'momentum',
                'weight_decay', 'warmup_epochs', 'close_mosaic', 'workers', 'seed']
# 기본값과 달라도 하이퍼파라미터로 보지 않는 항목 (경로/출력 관련)
BOOKKEEPING = {'task', 'mode', 'model', 'data', 'project', 'name', 'exist_ok', 'save_dir', 'device',
               'verbose', 'plots', 'save', 'save_period', 'resume', 'val'}


def fmt_duration(sec):
    """초 → '1시간 02분 03초'"""
    if sec is None or not np.isfinite(sec):
        return '-'
    sec = int(round(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f'{h}시간 {m:02d}분 {s:02d}초' if h else f'{m}분 {s:02d}초'


def read_results(train_dir):
    """results.csv 읽기 + fitness(best.pt 선정 기준) 계산"""
    df = pd.read_csv(Path(train_dir) / 'results.csv')
    df.columns = [c.strip() for c in df.columns]
    # ultralytics 8.4의 best.pt 선정 기준(DetMetrics.fitness): mAP50-95
    df['fitness'] = df[M['mAP50-95']]
    return df


def load_run(train_dir, work_dir):
    """학습 폴더 1개의 설정/기록 요약을 dict로 반환"""
    from ultralytics.utils import DEFAULT_CFG_DICT

    d = Path(train_dir).resolve()
    args = yaml.safe_load((d / 'args.yaml').read_text()) if (d / 'args.yaml').exists() else {}
    info = yaml.safe_load((d / TRAIN_INFO).read_text()) if (d / TRAIN_INFO).exists() else {}
    df = read_results(d)

    data_yaml = Path(args.get('data', Path(work_dir) / 'custom_data.yaml'))
    if not data_yaml.exists():
        data_yaml = Path(work_dir) / 'custom_data.yaml'
    data_cfg = yaml.safe_load(data_yaml.read_text())

    # epoch / 조기 종료 판정 (ultralytics EarlyStopping: 최고 epoch 이후 patience 동안 개선 없으면 종료)
    best_epoch = int(df.loc[df['fitness'].idxmax(), 'epoch'])
    last_epoch = int(df['epoch'].iloc[-1])
    set_epochs = int(args.get('epochs', last_epoch))
    patience = args.get('patience')
    early_stop = bool(last_epoch < set_epochs and patience is not None and last_epoch - best_epoch >= patience)
    if early_stop:
        status = f'조기 종료 (epoch {last_epoch}에서 종료, {last_epoch - best_epoch} epoch 동안 개선 없음)'
    elif last_epoch >= set_epochs:
        status = '설정한 epoch까지 완료'
    else:
        status = f'중단됨 (epoch {last_epoch}에서 끝남, 조기 종료 조건은 아님)'

    # 사용자가 넣은 하이퍼파라미터: 노트북이 저장한 train_info.yaml 우선, 없으면 기본값과 다른 값 + 주요 항목
    if info.get('user_hparams'):
        hparams, hp_source = dict(info['user_hparams']), '학습 노트북 TRAIN_ARGS (train_info.yaml)'
    else:
        hparams = {k: args[k] for k in CORE_HPARAMS if k in args}
        hparams.update({k: v for k, v in args.items()
                        if k not in BOOKKEEPING and k in DEFAULT_CFG_DICT and DEFAULT_CFG_DICT[k] != v})
        hp_source = 'args.yaml (주요 항목 + 기본값과 다른 값)'

    return {
        'dir': d, 'name': d.name, 'model': str(args.get('model', '?')), 'args': args, 'df': df,
        'data_yaml': data_yaml, 'data_cfg': data_cfg, 'class_names': list(data_cfg['names']),
        'set_epochs': set_epochs, 'last_epoch': last_epoch, 'best_epoch': best_epoch,
        'patience': patience, 'early_stop': early_stop, 'status': status,
        'train_time_s': float(df['time'].iloc[-1]),          # results.csv 기준 (학습+에폭별 검증)
        'wall_time_s': info.get('wall_time_s'),              # 노트북에서 model.train() 전체 소요 시간
        'hparams': hparams, 'hparams_source': hp_source,
    }


def run_info_lines(run):
    """학습 정보 요약 텍스트 (노트북 출력 / summary.txt 공용)"""
    lines = [
        f"- 모델: {run['model']}",
        f"- Epochs: 설정 {run['set_epochs']} / 실제 {run['last_epoch']} / best {run['best_epoch']}"
        f" (patience {run['patience']})",
        f"- 종료 상태: {run['status']}",
        f"- Total Training Time: {fmt_duration(run['train_time_s'])} (results.csv 기준)",
    ]
    if run['wall_time_s'] is not None:
        lines.append(f"- 실제 소요 시간(준비·최종 검증 포함): {fmt_duration(run['wall_time_s'])}")
    lines.append(f"- Training Hyper Parameters ({run['hparams_source']}):")
    lines += [f'    {k}: {v}' for k, v in run['hparams'].items()]
    return lines


def f1_confidence(box):
    """val 결과(box metric)의 F1-Confidence 곡선 → (conf축, 클래스별 F1, 평균 F1, 최고 F1, 그때 conf)"""
    px, f1 = np.asarray(box.px), np.asarray(box.f1_curve)
    if f1.size == 0:
        return px, f1, np.zeros_like(px), 0.0, float('nan')
    mean = f1.mean(0)
    i = int(mean.argmax())
    return px, f1, mean, float(mean[i]), float(px[i])


def split_dirs(data_cfg, split):
    """data yaml 기준 split의 (images 폴더, labels 폴더). split: 'val' | 'test' | 'train'"""
    img_dir = Path(data_cfg['path']) / data_cfg[split]
    return img_dir, img_dir.parent / 'labels'


def _iou(box, boxes):
    """box (4,) 와 boxes (N,4) 의 IoU, xyxy"""
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def predict_matches(model, data_cfg, split, class_names, iou_thr=IOU_MATCH):
    """split 이미지 전체를 낮은 confidence(0.001)로 예측하고 정답 라벨과 매칭.

    confidence 높은 예측부터 같은 클래스·아직 안 쓴 정답 박스와 IoU >= iou_thr 이면 TP.
    이 방식은 높은 conf 예측의 판정이 낮은 conf 예측에 영향을 받지 않으므로,
    한 번 예측해 두고 conf 기준값만 바꿔 가며 threshold_stats()로 집계할 수 있다.
    반환: (예측 DataFrame[image, class, conf, tp], 클래스별 정답 박스 수 dict)
    """
    img_dir, lbl_dir = split_dirs(data_cfg, split)
    images = sorted(p for p in img_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    rows, gt_counts = [], {c: 0 for c in class_names}
    for r in model.predict(source=[str(p) for p in images], conf=0.001, stream=True, verbose=False):
        h, w = r.orig_shape
        lf = lbl_dir / (Path(r.path).stem + '.txt')
        gt = np.array([list(map(float, l.split()[:5])) for l in lf.read_text().splitlines() if l.strip()]
                      if lf.exists() else np.zeros((0, 5))).reshape(-1, 5)
        gt_cls = gt[:, 0].astype(int)
        gt_xyxy = np.stack([(gt[:, 1] - gt[:, 3] / 2) * w, (gt[:, 2] - gt[:, 4] / 2) * h,
                            (gt[:, 1] + gt[:, 3] / 2) * w, (gt[:, 2] + gt[:, 4] / 2) * h], 1)
        for c in gt_cls:
            gt_counts[class_names[c]] += 1
        used = np.zeros(len(gt), bool)
        xyxy, conf, cls = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int))
        for k in np.argsort(-conf):
            tp = False
            cand = np.where((gt_cls == cls[k]) & ~used)[0]
            if cand.size:
                ious = _iou(xyxy[k], gt_xyxy[cand])
                j = ious.argmax()
                if ious[j] >= iou_thr:
                    used[cand[j]] = tp = True
            rows.append({'image': Path(r.path).name, 'class': class_names[cls[k]],
                         'conf': float(conf[k]), 'tp': tp})
    return pd.DataFrame(rows, columns=['image', 'class', 'conf', 'tp']), gt_counts


def threshold_stats(preds, gt_counts, conf_thr):
    """conf_thr 이상인 예측만 남겼을 때의 클래스별/전체 통계"""
    p = preds[preds['conf'] >= conf_thr]
    rows = []
    for cls in list(gt_counts) + ['all']:
        g = p if cls == 'all' else p[p['class'] == cls]
        n_gt = sum(gt_counts.values()) if cls == 'all' else gt_counts[cls]
        tp = int(g['tp'].sum()); n = len(g)
        rows.append({'conf 기준': round(conf_thr, 3), 'class': cls, '정답 박스': n_gt, '예측 박스': n,
                     '정답(TP)': tp, '오탐(FP)': n - tp, '미탐(FN)': n_gt - tp,
                     '% Correct Predictions': 100 * tp / n if n else np.nan,   # 예측 중 맞은 비율
                     '탐지율 %': 100 * tp / n_gt if n_gt else np.nan,          # 정답 중 찾아낸 비율
                     'Average Confidence': g['conf'].mean() if n else np.nan,
                     'Avg Conf (TP)': g.loc[g['tp'], 'conf'].mean() if tp else np.nan,
                     'Avg Conf (FP)': g.loc[~g['tp'], 'conf'].mean() if n - tp else np.nan})
    return pd.DataFrame(rows)


def measure_speed(model, data_cfg, split, conf, warmup=2):
    """batch=1로 split 이미지를 하나씩 추론한 평균 속도(ms/img)와 FPS. 앞 warmup장은 제외"""
    img_dir, _ = split_dirs(data_cfg, split)
    images = sorted(str(p) for p in img_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    sp = pd.DataFrame([r.speed for r in model.predict(source=images, conf=conf, stream=True, verbose=False)])
    sp = sp.iloc[warmup:] if len(sp) > warmup else sp
    out = {f'{k} ms': sp[k].mean() for k in ['preprocess', 'inference', 'postprocess']}
    out['total ms'] = sum(out.values())
    out['FPS'] = 1000 / out['total ms'] if out['total ms'] else np.nan
    out['images'] = len(sp)
    return out
