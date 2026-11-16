"""
Real-time Object Detection + Tracking + Counting
Using Ultralytics YOLO11
"""

from ultralytics import YOLO
import cv2
from collections import defaultdict, Counter
import numpy as np
import os
from datetime import datetime

# ================== CONFIG ==================
MODEL_PATH = "yolo11s.pt"          # Best balance: yolo11n.pt (fast) | yolo11s.pt (better) | yolo11m.pt (best accuracy)
SOURCE = 0                         # 0 = Webcam | "videos/sample.mp4" = video file
CONFIDENCE = 0.45
TRACKER = "bytetrack.yaml"
SHOW_TRAJECTORY = True
SAVE_OUTPUT = False

# None = all classes | Example: [0, 73, 39] for person, book, bottle
TARGET_CLASSES = None
# ============================================

def main():
    print("=" * 60)
    print("  YOLO Real-time Object Tracking & Counting")
    print("  Model:", MODEL_PATH)
    print("=" * 60)

    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Total classes: {len(model.names)}")

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {SOURCE}")
        return

    writer = None
    if SAVE_OUTPUT:
        os.makedirs("output", exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"output/tracked_{timestamp}.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"[INFO] Saving to: {out_path}")

    track_history = defaultdict(lambda: [])
    unique_ids = set()
    class_unique_ids = defaultdict(set)

    print("\n[INFO] Starting... Press 'q' to quit | 's' for screenshot")
    print("-" * 60)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            conf=CONFIDENCE,
            tracker=TRACKER,
            classes=TARGET_CLASSES,
            verbose=False
        )

        annotated = frame.copy()
        current_counts = Counter()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[cls_id]

                current_counts[class_name] += 1
                unique_ids.add(track_id)
                class_unique_ids[class_name].add(track_id)

                # Draw box
                color = (0, 255, 120)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{track_id} {class_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # Trajectory
                if SHOW_TRAJECTORY:
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    track = track_history[track_id]
                    track.append((cx, cy))
                    if len(track) > 40:
                        track.pop(0)
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [points], False, (180, 180, 180), 2)

        # Count Panel
        panel_h = 45 + 26 * (len(current_counts) + 2)
        cv2.rectangle(annotated, (10, 10), (340, panel_h), (25, 25, 25), -1)
        cv2.rectangle(annotated, (10, 10), (340, panel_h), (0, 220, 120), 2)

        y = 35
        cv2.putText(annotated, "OBJECT COUNT (Current Frame)", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 160), 2)

        y += 28
        if current_counts:
            for name, cnt in sorted(current_counts.items()):
                cv2.putText(annotated, f"  {name}: {cnt}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                y += 25
        else:
            cv2.putText(annotated, "  No objects detected", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
            y += 25

        y += 8
        cv2.putText(annotated, f"Unique Tracked: {len(unique_ids)}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)

        cv2.putText(annotated, f"Frame: {frame_count}", (annotated.shape[1] - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("YOLO Object Tracking & Counting | Press Q to Quit", annotated)

        if writer:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("output", exist_ok=True)
            snap = f"output/snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(snap, annotated)
            print(f"[INFO] Snapshot saved: {snap}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total frames processed : {frame_count}")
    print(f"Unique objects tracked : {len(unique_ids)}")
    print("\nPer-class unique counts:")
    for name, ids in sorted(class_unique_ids.items()):
        print(f"  {name}: {len(ids)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
