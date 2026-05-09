import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# ========================
# Utility Functions
# ========================

def calculate_iou(boxA, boxB):
    """Compute IoU for two boxes in [x_min, y_min, x_max, y_max]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def convert_yolo_to_xyxy(box, img_width, img_height):
    """Convert YOLO [x_center, y_center, w, h] → [x_min, y_min, x_max, y_max]."""
    x_c, y_c, w, h = box
    x_min = (x_c - w / 2) * img_width
    y_min = (y_c - h / 2) * img_height
    x_max = (x_c + w / 2) * img_width
    y_max = (y_c + h / 2) * img_height
    return [x_min, y_min, x_max, y_max]


def parse_yolo_labels(label_file, img_width, img_height):
    """Parse YOLO label file into [{'bbox': [...], 'class_id': int}, ...]."""
    annotations = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            bbox = convert_yolo_to_xyxy(parts[1:], img_width, img_height)
            annotations.append({'bbox': bbox, 'class_id': class_id})
    return annotations


# ========================
# Benchmark Function
# ========================

def benchmark_yolo_sahi(model_path, dataset_path,
                        conf_threshold=0.25,
                        slice_height=512,
                        slice_width=512,
                        overlap_ratio=0.2,
                        iou_thresholds=None):

    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    class_names = model.names

    # Wrap model for SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device
    )

    image_dir = Path(dataset_path) / "images"
    label_dir = Path(dataset_path) / "labels"
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    print(f"\nRunning YOLOv11 + SAHI benchmarking on {len(image_files)} images...\n")

    total_TP = defaultdict(int)
    total_FP = defaultdict(int)
    total_FN = defaultdict(int)
    iou_results = defaultdict(list)

    for image_path in tqdm(image_files):
        image_name = image_path.stem
        label_path = label_dir / f"{image_name}.txt"
        if not label_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        gt_annotations = parse_yolo_labels(label_path, img_w, img_h)

        # SAHI prediction
        result = get_sliced_prediction(
            image=str(image_path),
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=False,
        )

        pred_boxes = [obj.bbox.to_xyxy() for obj in result.object_prediction_list]
        pred_classes = [obj.category.id for obj in result.object_prediction_list]
        pred_scores = [obj.score.value for obj in result.object_prediction_list]

        for iou_t in iou_thresholds:
            matched_gt = set()
            matched_pred = set()

            for i, (pb, pc, ps) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
                best_iou = 0
                best_j = -1

                for j, gt in enumerate(gt_annotations):
                    if gt['class_id'] != pc or j in matched_gt:
                        continue
                    iou = calculate_iou(pb, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= iou_t:
                    total_TP[pc] += 1
                    matched_gt.add(best_j)
                    matched_pred.add(i)
                    iou_results[pc].append(best_iou)
                else:
                    total_FP[pc] += 1

            # False negatives
            for j, gt in enumerate(gt_annotations):
                if j not in matched_gt:
                    total_FN[gt['class_id']] += 1

    # ========================
    # Metrics Summary
    # ========================
    print("\n" + "="*50)
    print("YOLOv11 + SAHI Benchmarking Results")
    print("="*50)

    metrics = []
    for cid, cname in class_names.items():
        TP, FP, FN = total_TP[cid], total_FP[cid], total_FN[cid]
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        mean_iou = np.mean(iou_results[cid]) if len(iou_results[cid]) > 0 else 0
        metrics.append((cname, TP, FP, FN, precision, recall, f1, mean_iou))
        print(f"\nClass '{cname}':")
        print(f"  TP={TP}, FP={FP}, FN={FN}")
        print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Mean IoU={mean_iou:.3f}")

    # ========================
    # Confusion Matrix
    # ========================
    num_classes = len(class_names)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for cid in range(num_classes):
        conf_matrix[cid, cid] = total_TP[cid]

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(class_names.values()),
                yticklabels=list(class_names.values()))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("YOLOv11 + SAHI Confusion Matrix (IoU 0.5–0.95)")
    plt.tight_layout()
    plt.savefig("yolov11_sahi_confusion_matrix.png")
    print("\nConfusion matrix saved as 'yolov11_sahi_confusion_matrix.png'.\n")

    return metrics


# ========================
# Example Usage
# ========================
if __name__ == "__main__":
    MODEL_PATH = "./best_per_video.pt"          # path to YOLOv11 model
    DATASET_PATH = "./total_dataset_per_video/val"  # dataset folder with /images and /labels

    benchmark_yolo_sahi(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        conf_threshold=0.25,
        slice_height=512,
        slice_width=512,
        overlap_ratio=0.2,
    )
