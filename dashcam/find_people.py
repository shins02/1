#!/usr/bin/env python3
"""블랙박스 SD카드에서 사람이 등장하는 영상을 찾아 파일명을 정리한다.

대상: <SD카드>/Driving, Driving_Shock, Parking, Parking_Shock 폴더 안의
      파일명에 20261001 또는 20261002 가 들어간 동영상.

사용법 (맥북 터미널):
    pip3 install ultralytics opencv-python
    python3 find_people.py /Volumes/Untitled

결과:
    ./결과/결과목록.csv       사람이 감지된 영상 목록
    ./결과/결과요약.txt       폴더별 파일명 정리
    ./결과/사진/*.jpg        영상마다 사람이 가장 크게 잡힌 장면 (빨간 박스 표시)
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

FOLDERS = ["Driving", "Driving_Shock", "Parking", "Parking_Shock"]
DATES = ["20261001", "20261002"]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".h264"}
PERSON = 0  # COCO 클래스 0 = 사람


def find_videos(root):
    for folder in FOLDERS:
        d = root / folder
        if not d.is_dir():
            print(f"[알림] 폴더 없음: {d}")
            continue
        for p in sorted(d.rglob("*")):
            if (p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._")
                    and any(day in p.name for day in DATES)):
                yield folder, p


def scan(model, path, step_sec, conf, device):
    """영상을 step_sec 간격으로 훑어 사람 감지 결과를 돌려준다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * step_sec))
    hits, times = 0, []
    best = (0.0, None, 0.0)  # (사람 크기 비율, 표시된 프레임, 시각)
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            r = model.predict(frame, classes=[PERSON], conf=conf,
                              device=device, verbose=False)[0]
            if len(r.boxes):
                hits += 1
                t = idx / fps
                times.append(t)
                h, w = frame.shape[:2]
                area = max(float((b[2] - b[0]) * (b[3] - b[1]))
                           for b in r.boxes.xyxy.tolist()) / (w * h)
                if area > best[0]:
                    shown = frame.copy()
                    for x1, y1, x2, y2 in r.boxes.xyxy.int().tolist():
                        cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    best = (area, shown, t)
        idx += 1
    cap.release()
    return hits, times, best


def fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sd", nargs="?", default="/Volumes/Untitled", help="SD카드 경로")
    ap.add_argument("--step", type=float, default=0.5, help="몇 초마다 검사할지 (기본 0.5)")
    ap.add_argument("--conf", type=float, default=0.45, help="사람 판정 기준 0~1 (기본 0.45)")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="사람이 최소 몇 번 잡혀야 목록에 넣을지 (기본 2, 오탐 줄이기)")
    ap.add_argument("--out", default="결과", help="결과 폴더")
    args = ap.parse_args()

    root = Path(args.sd)
    if not root.is_dir():
        sys.exit(f"SD카드 경로를 찾을 수 없습니다: {root}")
    videos = list(find_videos(root))
    if not videos:
        sys.exit("조건(20261001/20261002)에 맞는 영상이 없습니다.")
    print(f"검사할 영상: {len(videos)}개")

    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        device = "cpu"
    model = YOLO("yolov8s.pt")

    out = Path(args.out)
    (out / "사진").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (folder, path) in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {folder}/{path.name} ... ", end="", flush=True)
        res = scan(model, path, args.step, args.conf, device)
        if res is None:
            print("열 수 없음")
            rows.append([folder, path.name, "열 수 없음", "", "", ""])
            continue
        hits, times, (area, shown, t_best) = res
        if hits < args.min_hits:
            print("사람 없음")
            continue
        img = out / "사진" / f"{folder}__{path.stem}.jpg"
        cv2.imwrite(str(img), shown)
        print(f"사람 감지 {hits}회 ({fmt(times[0])}~{fmt(times[-1])})")
        rows.append([folder, path.name, hits, fmt(times[0]), fmt(times[-1]),
                     f"{area * 100:.1f}%"])

    with open(out / "결과목록.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["폴더", "파일명", "사람감지횟수", "처음등장", "마지막등장", "사람최대크기(화면비)"])
        w.writerows(rows)
    with open(out / "결과요약.txt", "w", encoding="utf-8") as f:
        for folder in FOLDERS:
            names = [r for r in rows if r[0] == folder]
            f.write(f"■ {folder} ({len(names)}개)\n")
            for r in names:
                f.write(f"  - {r[1]}  감지 {r[2]}회, {r[3]}~{r[4]}, 크기 {r[5]}\n")
            f.write("\n")

    print(f"\n완료: 사람이 나온 영상 {len(rows)}개 / 전체 {len(videos)}개")
    print(f"목록: {out / '결과요약.txt'}")
    print(f"장면 사진: {out / '사진'}  (사진만 넘겨보며 빠르게 확인하세요)")


if __name__ == "__main__":
    main()
