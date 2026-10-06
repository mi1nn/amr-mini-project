"""캘리브레이션 오차 검증 도구.

로봇을 치우고 십자만 보이는 상태에서, amcl_pose 로 찍었던 순서와 같은 순서로
십자 중심을 클릭하면 아래 오차를 비교한다.
  A) 저장된 H + 당시 클릭한 픽셀   (기존 캘리브레이션 상태)
  B) 저장된 H + 이번에 십자를 클릭한 픽셀  (H 자체가 맞는지)
  C) 이번 클릭으로 새로 구한 H (전체 점, 최소제곱) + 잔차
  D) C 와 같은 방식의 leave-one-out 오차 (새 점에 대한 일반화 오차)

조작: c=현재 프레임 고정 / 클릭=십자 지정 / u=되돌리기 / s=계산·저장 / q=종료
실행: python3 check_calib.py   (ROS 불필요, 웹캠만 필요)
"""
import csv
import os
import sys

import cv2
import numpy as np

WEBCAM_INDEX = 2
FRAME_SIZE = (640, 480)
H_PATH = '/home/mu-06/rokey_ws/webcam_H.npy'
OUT_DIR = '/home/mu-06/rokey_ws/calib_check'

# (당시 클릭 픽셀 u, v, amcl map x, y) - 클릭/amcl 찍은 순서 그대로
POINTS = [
    (615, 176, -3.944, 3.422), (614, 83, -4.405, 4.098), (477, 116, -4.631, 3.803),
    (368, 134, -4.790, 3.389), (227, 165, -4.925, 2.598), (146, 184, -4.882, 2.315),
    (234, 352, -3.947, 2.129), (102, 459, -3.481, 1.858), (438, 455, -3.249, 2.069),
    (608, 448, -3.051, 2.380), (562, 307, -3.347, 2.767), (344, 287, -3.786, 2.670),
    (324, 225, -4.170, 2.494), (484, 200, -4.164, 2.816), (478, 318, -3.608, 2.672),
]
ORIG = np.float32([p[:2] for p in POINTS])
MAP = np.float32([p[2:] for p in POINTS])
WIN = 'check calibration'


def project(H, pix):
    return cv2.perspectiveTransform(np.float32(pix).reshape(-1, 1, 2), H).reshape(-1, 2)


def err_cm(H, pix, mp=MAP):
    return np.linalg.norm(project(H, pix) - mp, axis=1) * 100


def fit(pix, mp):
    H, _ = cv2.findHomography(np.float32(pix), np.float32(mp), 0)
    return H


def loo_err_cm(pix):
    out = []
    for i in range(len(pix)):
        keep = [j for j in range(len(pix)) if j != i]
        H = fit(pix[keep], MAP[keep])
        out.append(err_cm(H, pix[i:i + 1], MAP[i:i + 1])[0])
    return np.array(out)


def compute_and_save(new_pix, frame):
    os.makedirs(OUT_DIR, exist_ok=True)
    new = np.float32(new_pix)
    H_saved = np.load(H_PATH)
    H_new = fit(new, MAP)
    a = err_cm(H_saved, ORIG)
    b = err_cm(H_saved, new)
    c = err_cm(H_new, new)
    d = loo_err_cm(new)
    shift = np.linalg.norm(new - ORIG, axis=1)

    print(f'\n{"#":>2} {"당시px":>11} {"십자px":>11} {"px차":>5} | '
          f'{"A 저장H+당시":>11} {"B 저장H+십자":>11} {"C 새H잔차":>9} {"D LOO":>6}  (cm)')
    for i in range(len(POINTS)):
        print(f'{i + 1:>2} ({ORIG[i][0]:>4.0f},{ORIG[i][1]:>4.0f}) ({new[i][0]:>4.0f},{new[i][1]:>4.0f}) '
              f'{shift[i]:>5.1f} | {a[i]:>11.1f} {b[i]:>11.1f} {c[i]:>9.1f} {d[i]:>6.1f}')
    print(f'{"평균":>38} | {a.mean():>11.1f} {b.mean():>11.1f} {c.mean():>9.1f} {d.mean():>6.1f}')
    print(f'{"최대":>38} | {a.max():>11.1f} {b.max():>11.1f} {c.max():>9.1f} {d.max():>6.1f}')

    with open(os.path.join(OUT_DIR, 'points.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'orig_u', 'orig_v', 'cross_u', 'cross_v', 'map_x', 'map_y',
                    'pix_shift', 'err_A_cm', 'err_B_cm', 'err_C_cm', 'err_D_loo_cm'])
        for i in range(len(POINTS)):
            w.writerow([i + 1, *ORIG[i], *new[i], *MAP[i], f'{shift[i]:.2f}',
                        f'{a[i]:.2f}', f'{b[i]:.2f}', f'{c[i]:.2f}', f'{d[i]:.2f}'])
    np.save(os.path.join(OUT_DIR, 'H_new_all_points.npy'), H_new)  # webcam_H.npy 는 덮어쓰지 않음
    cv2.imwrite(os.path.join(OUT_DIR, 'snapshot.png'), frame)
    print(f'\n저장: {OUT_DIR}/points.csv, H_new_all_points.npy, snapshot.png (webcam_H.npy 는 그대로)')


def main():
    if not os.path.exists(H_PATH):
        sys.exit(f'{H_PATH} 없음')
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    if not cap.isOpened():
        sys.exit(f'웹캠 {WEBCAM_INDEX} 열기 실패')

    state = {'frozen': None, 'pts': []}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and state['frozen'] is not None:
            if len(state['pts']) < len(POINTS):
                state['pts'].append((x, y))
                print(f'[{len(state["pts"])}/{len(POINTS)}] 십자 클릭 ({x}, {y})')

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    try:
        while True:
            if state['frozen'] is None:
                ok, frame = cap.read()
                if not ok:
                    continue
                view = frame.copy()
                cv2.putText(view, "robot removed? press 'c' to freeze", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                frame = state['frozen']
                view = frame.copy()
                for i, (u, v) in enumerate(ORIG):  # 빨강: 당시 클릭 위치
                    cv2.circle(view, (int(u), int(v)), 6, (0, 0, 255), 1)
                for i, (u, v) in enumerate(state['pts'], 1):  # 초록: 이번 십자 클릭
                    cv2.circle(view, (u, v), 4, (0, 255, 0), -1)
                    cv2.putText(view, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                n = len(state['pts'])
                msg = f'click cross #{n + 1}' if n < len(POINTS) else "done: press 's'"
                cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if n < len(POINTS):  # 다음에 클릭할 점의 당시 위치를 힌트로 강조
                    cv2.circle(view, (int(ORIG[n][0]), int(ORIG[n][1])), 12, (0, 255, 255), 2)
            cv2.imshow(WIN, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                state['frozen'] = frame.copy()
                state['pts'] = []
            elif key == ord('u') and state['pts']:
                state['pts'].pop()
            elif key == ord('s'):
                if len(state['pts']) == len(POINTS):
                    compute_and_save(state['pts'], state['frozen'])
                else:
                    print(f'{len(POINTS)}점을 모두 클릭하세요 (현재 {len(state["pts"])})')
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
