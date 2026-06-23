"""
===========================================================================
POSE TRACKING - trích xuất clip pose theo TỪNG NGƯỜI để tạo dataset training
===========================================================================

Best-practice pipeline (YOLO pose + BoT-SORT/ReID):

  PASS 1  (stream tracking)
    - model.track(stream=True) đọc & track toàn video, ID ổn định nhờ ReID
    - Với mỗi người ở mỗi frame: lưu bbox + keypoints
    - Chấm điểm chất lượng pose, lọc occlusion theo CONFIDENCE keypoint,
      lọc bbox quá nhỏ / dính mép khung / ảnh mờ

  CHỌN FRAME
    - Gom các khoảnh khắc đẹp của từng người thành cửa sổ (min_gap),
      chọn frame có ĐIỂM CAO NHẤT trong mỗi cửa sổ (không lấy đại frame đầu)

  PASS 2  (export)
    - Mỗi khoảnh khắc -> 1 clip riêng, crop bám người + letterbox vuông
    - Tuỳ chọn: xuất nhãn keypoints JSON kèm theo clip để train

Output:
    out_dir/person_<id>_clip_<n>.mp4
    out_dir/person_<id>_clip_<n>.json   (nếu export_labels=True)
===========================================================================
"""

import os
import json

import cv2
import numpy as np
from ultralytics import YOLO

# ===========================================================================
# CẤU HÌNH
# ===========================================================================

CFG = dict(
    # ----- model & tracker -----
    model_path="yolo11s-pose.pt",   # mạnh hơn yolov8n nhiều. Đổi yolo11m-pose.pt nếu cần chính xác hơn
    tracker="botsort_reid.yaml",    # BoT-SORT + ReID (file cùng thư mục)
    det_conf=0.5,                   # ngưỡng detect người
    imgsz=960,                      # ảnh lớn -> keypoint xa chính xác hơn
    device=None,                    # None=auto, "cpu", 0 (GPU id)...

    # ----- clip -----
    clip_seconds=5,
    scan_interval=0.4,              # chấm điểm pose mỗi 0.4s
    out_size=384,                   # cạnh ô vuông output khi crop
    pad_ratio=0.25,                 # nới rộng crop quanh người
    crop=True,                      # True: crop bám người (data sạch)
    export_labels=True,             # xuất JSON keypoints theo clip

    # ----- bộ lọc chất lượng pose -----
    kp_vis_conf=0.5,                # keypoint coi là "thấy rõ" khi conf > giá trị này
    min_visible_kp=12,              # cần >= 12/17 keypoint rõ
    core_joints=(5, 6, 11, 12),     # vai + hông bắt buộc phải rõ
    core_conf=0.5,
    max_tilt_sin=0.45,              # sin(góc nghiêng vai) tối đa (~27 độ). Tăng để nhận pose nghiêng
    min_bbox_h_ratio=0.20,          # bbox cao >= 20% khung -> loại người quá xa/nhỏ
    edge_margin=4,                  # bbox dính mép trong margin px -> coi là bị cắt -> loại
    blur_thresh=60.0,               # variance of Laplacian < ngưỡng -> ảnh mờ -> loại
)

# COCO-17 keypoint index:
# 0 mũi 1-2 mắt 3-4 tai 5-6 vai 7-8 khuỷu 9-10 cổtay
# 11-12 hông 13-14 gối 15-16 cổ chân


# ===========================================================================
# CHẤM ĐIỂM & LỌC CHẤT LƯỢNG POSE
# ===========================================================================

def laplacian_blur(img):
    """Độ nét: variance of Laplacian (cao = nét)."""
    if img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pose_score(kp, cfg):
    """
    Trả về điểm chất lượng pose (float) nếu ĐẠT, hoặc None nếu loại.
    Occlusion được đánh giá bằng CONFIDENCE keypoint (đúng bản chất hơn
    là xét vị trí hình học tay-trong-thân).
    """
    conf = kp[:, 2]

    # 1) các khớp lõi (vai, hông) phải rõ
    if np.any(conf[list(cfg["core_joints"])] < cfg["core_conf"]):
        return None

    # 2) đủ số keypoint nhìn thấy -> không bị che quá nhiều
    n_visible = int(np.sum(conf > cfg["kp_vis_conf"]))
    if n_visible < cfg["min_visible_kp"]:
        return None

    # 3) người tương đối thẳng (chuẩn hoá theo độ dài vai -> bất biến tỉ lệ)
    l_sh, r_sh = kp[5, :2], kp[6, :2]
    sh_vec = r_sh - l_sh
    sh_len = np.linalg.norm(sh_vec) + 1e-6
    tilt_sin = abs(sh_vec[1]) / sh_len
    if tilt_sin > cfg["max_tilt_sin"]:
        return None

    # điểm = độ rõ trung bình toàn thân + thưởng số keypoint rõ
    return float(np.mean(conf) + 0.02 * n_visible)


def bbox_ok(box, frame_w, frame_h, cfg):
    """Loại người quá nhỏ hoặc bị cắt ở mép khung."""
    x1, y1, x2, y2 = box
    if (y2 - y1) < cfg["min_bbox_h_ratio"] * frame_h:
        return False
    m = cfg["edge_margin"]
    if x1 <= m or y1 <= m or x2 >= frame_w - m or y2 >= frame_h - m:
        return False
    return True


# ===========================================================================
# HELPERS
# ===========================================================================

def pad_clip_box(box, w, h, pad_ratio):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    return (
        int(max(0, x1 - px)), int(max(0, y1 - py)),
        int(min(w, x2 + px)), int(min(h, y2 + py)),
    )


def letterbox_square(img, size, color=(0, 0, 0)):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


# ===========================================================================
# MAIN
# ===========================================================================

def export_pose_clips(video_path, out_dir="pose_clips", cfg=CFG):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    step = max(1, int(fps * cfg["scan_interval"]))

    model = YOLO(cfg["model_path"])

    # qualified[pid] = [(frame_idx, score), ...]
    # boxes[pid][frame_idx] = (x1,y1,x2,y2)
    # kpts[pid][frame_idx]  = (17,3)   (chỉ lưu khi export_labels)
    qualified, boxes, kpts = {}, {}, {}

    print("PASS 1: tracking + scoring...")

    # stream=True: ultralytics tự đọc video tuần tự -> tracking liên tục, đúng cách
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=cfg["tracker"],
        conf=cfg["det_conf"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        verbose=False,
    )

    for frame_idx, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue

        ids = r.boxes.id.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()
        kps = r.keypoints.data.cpu().numpy()      # (N,17,3)
        scan = (frame_idx % step == 0)

        for pid, box, kp in zip(ids, xyxy, kps):
            boxes.setdefault(pid, {})[frame_idx] = box
            if cfg["export_labels"]:
                kpts.setdefault(pid, {})[frame_idx] = kp

            if not scan or not bbox_ok(box, W, H, cfg):
                continue

            s = pose_score(kp, cfg)
            if s is None:
                continue

            # lọc mờ: crop nhanh quanh người để đo độ nét
            x1, y1, x2, y2 = [int(v) for v in box]
            if laplacian_blur(r.orig_img[y1:y2, x1:x2]) < cfg["blur_thresh"]:
                continue

            qualified.setdefault(pid, []).append((frame_idx, s))

    if not qualified:
        print("❌ Không tìm thấy pose đạt chuẩn")
        return

    # ----- CHỌN FRAME ĐẸP NHẤT TRONG MỖI CỬA SỔ -----
    min_gap = int(fps * cfg["clip_seconds"])
    half = int(fps * cfg["clip_seconds"] / 2)

    clips = []  # (pid, center_frame)
    for pid in sorted(qualified):
        items = sorted(qualified[pid])                 # theo frame_idx
        window = [items[0]]
        for f, s in items[1:]:
            if f - window[0][0] > min_gap:
                best = max(window, key=lambda t: t[1])  # frame điểm cao nhất
                clips.append((pid, best[0]))
                window = [(f, s)]
            else:
                window.append((f, s))
        best = max(window, key=lambda t: t[1])
        clips.append((pid, best[0]))

    n_people = len({c[0] for c in clips})
    print(f"✅ {len(clips)} clip từ {n_people} người")

    # ----- PASS 2: EXPORT -----
    print("PASS 2: export clips...")
    per_person = {}

    for pid, center in clips:
        n = per_person.get(pid, 0)
        per_person[pid] = n + 1

        start = max(0, center - half)
        end = min(total - 1, center + half)
        base = os.path.join(out_dir, f"person_{pid:03d}_clip_{n:03d}")

        size = (cfg["out_size"], cfg["out_size"]) if cfg["crop"] else (W, H)
        writer = cv2.VideoWriter(
            base + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        label_frames = []
        last_box = None
        cur = start
        while cur <= end:
            ret, frame = cap.read()
            if not ret:
                break

            box = boxes.get(pid, {}).get(cur, last_box)

            if cfg["crop"]:
                if box is not None:
                    last_box = box
                    cx1, cy1, cx2, cy2 = pad_clip_box(box, W, H, cfg["pad_ratio"])
                    sub = frame[cy1:cy2, cx1:cx2]
                    if sub.size > 0:
                        writer.write(letterbox_square(sub, cfg["out_size"]))
            else:
                writer.write(frame)

            if cfg["export_labels"] and box is not None:
                kp = kpts.get(pid, {}).get(cur)
                label_frames.append({
                    "frame": cur,
                    "bbox_xyxy": [float(v) for v in box],
                    # keypoints ở toạ độ ẢNH GỐC (x,y,conf) - dễ transform về crop sau
                    "keypoints": kp.tolist() if kp is not None else None,
                })

            cur += 1

        writer.release()
        cap.release()

        if cfg["export_labels"]:
            with open(base + ".json", "w") as f:
                json.dump({
                    "person_id": int(pid),
                    "video": os.path.basename(video_path),
                    "fps": fps,
                    "frame_range": [start, end],
                    "crop": cfg["crop"],
                    "out_size": cfg["out_size"],
                    "pad_ratio": cfg["pad_ratio"],
                    "frames": label_frames,
                }, f)

        print(f"  -> {base}.mp4")

    print("\n🎉 DONE:", out_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="đường dẫn video đầu vào")
    ap.add_argument("-o", "--out", default="pose_clips", help="thư mục output")
    ap.add_argument("--no-crop", action="store_true", help="xuất full-frame thay vì crop")
    ap.add_argument("--no-labels", action="store_true", help="không xuất JSON nhãn")
    ap.add_argument("--model", default=None, help="ghi đè model (vd yolo11m-pose.pt)")
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.no_crop:
        cfg["crop"] = False
    if args.no_labels:
        cfg["export_labels"] = False
    if args.model:
        cfg["model_path"] = args.model

    export_pose_clips(args.video, args.out, cfg)
