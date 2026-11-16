"""
Single Image Object Counting
Usage: python count_from_image.py path/to/image.jpg
"""

from ultralytics import YOLO
import cv2
import sys
from collections import Counter
import os

MODEL_PATH = "yolo11s.pt"

def count_objects(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    model = YOLO(MODEL_PATH)
    results = model(image_path, conf=0.4)

    counts = Counter()
    annotated = results[0].plot()

    if results[0].boxes is not None:
        for cls_id in results[0].boxes.cls.int().tolist():
            name = model.names[cls_id]
            counts[name] += 1

    print("\n" + "="*45)
    print("OBJECT COUNT RESULT")
    print("="*45)
    if counts:
        for name, cnt in counts.most_common():
            print(f"  {name:20s} : {cnt}")
        print("-"*45)
        print(f"  TOTAL OBJECTS      : {sum(counts.values())}")
    else:
        print("  No objects detected")
    print("="*45)

    os.makedirs("output", exist_ok=True)
    out_path = f"output/counted_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, annotated)
    print(f"\nResult saved to: {out_path}")

    cv2.imshow("Object Count", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_from_image.py <image_path>")
    else:
        count_objects(sys.argv[1])
