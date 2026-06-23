# -*- coding: utf-8 -*-
"""
extract_keypoints.py
第二阶段 Day 1 脚本：从视频中逐帧提取手部 21 个关键点，保存为 csv。

使用方法（Windows PowerShell 或 cmd）：
    python extract_keypoints.py --weights "C:\\home\\ubuntu\\runs\\hand_pose\\baseline_full_gpu_10e\\weights\\best.pt" --video_dir "D:\\thesis_stage2\\raw_videos" --out_dir "D:\\thesis_stage2\\keypoints_csv"

输出：
    out_dir/seg1_openclose.csv
    out_dir/seg2_wave.csv
    ...
    每个 csv 一行代表一帧，列为 frame_idx, x0, y0, c0, x1, y1, c1, ..., x20, y20, c20, bbox_x1, bbox_y1, bbox_x2, bbox_y2
"""

import argparse
import os
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def extract_one_video(model, video_path: Path, out_csv: Path, imgsz: int = 640, conf: float = 0.25):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {video_path.name}: {total} 帧, {fps:.2f} fps")

    header = ["frame_idx"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"c{i}"]
    header += ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "box_conf"]

    n_written = 0
    n_missing = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False, device=0)
            r = results[0]

            row = [frame_idx]
            got_hand = False
            if r.keypoints is not None and len(r.keypoints) > 0:
                # 取置信度最高的那只手（万一检测出两只手）
                if r.boxes is not None and len(r.boxes) > 0:
                    box_confs = r.boxes.conf.cpu().numpy()
                    best = int(box_confs.argmax())
                else:
                    best = 0

                kp = r.keypoints.data[best].cpu().numpy()  # (21, 3): x, y, conf
                if kp.shape[0] == 21:
                    for i in range(21):
                        row += [float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2])]
                    if r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy[best].cpu().numpy()
                        bc = float(r.boxes.conf[best].cpu().numpy())
                        row += [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), bc]
                    else:
                        row += [0.0, 0.0, 0.0, 0.0, 0.0]
                    got_hand = True

            if not got_hand:
                # 本帧未检测到手，写 NaN，后续训练脚本会处理
                row += [float("nan")] * (21 * 3)
                row += [float("nan")] * 5
                n_missing += 1

            writer.writerow(row)
            n_written += 1
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  处理 {frame_idx}/{total} 帧")

    cap.release()
    print(f"[DONE] {video_path.name} -> {out_csv.name}, 写入 {n_written} 帧, 其中 {n_missing} 帧未检测到手")
    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="best.pt 路径")
    parser.add_argument("--video_dir", type=str, required=True, help="视频文件夹")
    parser.add_argument("--out_dir", type=str, required=True, help="输出 csv 文件夹")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 加载模型: {args.weights}")
    model = YOLO(args.weights)

    video_dir = Path(args.video_dir)
    videos = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4")) + list(video_dir.glob("*.mov")))
    if not videos:
        print(f"[ERROR] 在 {video_dir} 中没有找到视频文件")
        return

    print(f"[INFO] 找到 {len(videos)} 个视频")
    total_frames = 0
    for v in videos:
        out_csv = out_dir / (v.stem + ".csv")
        total_frames += extract_one_video(model, v, out_csv, imgsz=args.imgsz, conf=args.conf)

    print(f"\n[SUMMARY] 全部完成，共 {total_frames} 帧")


if __name__ == "__main__":
    main()
