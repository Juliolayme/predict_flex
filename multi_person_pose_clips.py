"""
Trích xuất clip pose theo TỪNG NGƯỜI để tạo dataset huấn luyện.

Pipeline:
  1. Tracking toàn video -> mỗi người có 1 ID ổn định.
  2. Quét frame, chấm điểm pose cho TỪNG người, lưu các khoảnh khắc "đẹp".
  3. Gom khoảnh khắc theo ID người, loại các frame quá gần nhau.
  4. Xuất 1 clip riêng cho mỗi khoảnh khắc, crop bám sát người đó.

Kết quả: pose_clips/person_<id>_clip_<n>.mp4
"""

import os

import cv2
import numpy as np
from ultralytics import YOLO

# =====================================
# LOAD MODEL
# =====================================

model = YOLO("yolov8n-pose.pt")

# COCO keypoint index:
# 5 vai trái, 6 vai phải, 7 khuỷu trái, 8 khuỷu phải,
# 9 cổ tay trái, 10 cổ tay phải, 11 hông trái, 12 hông phải


# =====================================
# FILTERS (giữ nguyên logic, hoạt động trên keypoints của 1 người)
# =====================================

def is_aesthetic_pose(kp, conf_threshold=0.65, max_tilt=0.3):
    """Pose đẹp: các khớp chính rõ + vai không nghiêng quá."""
    main_joints = kp[[5, 6, 7, 8, 11, 12], 2]
    if np.mean(main_joints) < conf_threshold:
        return False

    l_sh, r_sh = kp[5], kp[6]
    shoulder_tilt = abs(l_sh[1] - r_sh[1]) / (abs(l_sh[0] - r_sh[0]) + 1e-6)
    if shoulder_tilt > max_tilt:
        return False

    return True


def is_occluded(kp):
    """Bị che: cổ tay nằm trong vùng thân (vai -> hông)."""
    l_w, r_w = kp[9], kp[10]
    l_sh, r_sh = kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]

    x1 = min(l_sh[0], r_sh[0])
    x2 = max(l_sh[0], r_sh[0])
    y1 = min(l_sh[1], r_sh[1])
    y2 = max(l_hip[1], r_hip[1])

    def inside(pt):
        return x1 < pt[0] < x2 and y1 < pt[1] < y2

    return inside(l_w) or inside(r_w)


# =====================================
# HELPERS
# =====================================

def pad_and_clip_box(box, width, height, pad_ratio=0.25):
    """Nới rộng bbox theo pad_ratio và kẹp trong khung hình."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio

    x1 = int(max(0, x1 - px))
    y1 = int(max(0, y1 - py))
    x2 = int(min(width, x2 + px))
    y2 = int(min(height, y2 + py))
    return x1, y1, x2, y2


# =====================================
# MAIN
# =====================================

def export_pose_clips(
    video_path,
    output_dir="pose_clips",
    clip_seconds=5,
    scan_interval=0.5,
    crop=True,          # True: crop bám người (data huấn luyện). False: full-frame.
    pad_ratio=0.25,     # nới rộng crop quanh người
    out_size=256,       # cạnh ô vuông output khi crop (letterbox)
    max_people=None,    # giới hạn số người xử lý (None = tất cả)
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step = max(1, int(fps * scan_interval))

    # ----- PASS 1: TRACKING + CHẤM ĐIỂM TỪNG NGƯỜI -----
    # qualified[id] = [frame_idx, ...]   các khoảnh khắc đẹp của người id
    # boxes[id][frame_idx] = (x1,y1,x2,y2)  bbox để crop ở pass 2
    qualified = {}
    boxes = {}

    frame_idx = 0
    print("Scanning video (tracking)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # persist=True giữ ID liên tục giữa các frame
        results = model.track(
            frame, persist=True, verbose=False, conf=0.6
        )
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            kps = r.keypoints.data.cpu().numpy()  # (N, 17, 3)

            for pid, box, kp in zip(ids, xyxy, kps):
                # luôn lưu bbox để pass 2 có thể crop kể cả frame không quét pose
                boxes.setdefault(pid, {})[frame_idx] = box

                if frame_idx % step != 0:
                    continue

                if is_aesthetic_pose(kp) and not is_occluded(kp):
                    qualified.setdefault(pid, []).append(frame_idx)

        frame_idx += 1

    cap.release()

    if not qualified:
        print("❌ Không tìm thấy pose đẹp")
        return

    # ----- GOM KHOẢNH KHẮC THEO NGƯỜI, BỎ FRAME QUÁ GẦN -----
    min_gap = int(fps * clip_seconds)
    half_clip = int(fps * clip_seconds / 2)

    # mỗi phần tử: (track_id, center_frame)
    clips = []
    person_ids = sorted(qualified.keys())
    if max_people is not None:
        person_ids = person_ids[:max_people]

    for pid in person_ids:
        last = None
        for f in qualified[pid]:
            if last is None or f - last > min_gap:
                clips.append((pid, f))
                last = f

    print(f"✅ Tìm thấy {len(clips)} clip từ {len(person_ids)} người")

    # ----- PASS 2: XUẤT CLIP TỪNG NGƯỜI -----
    per_person_count = {}

    for pid, center_frame in clips:
        n = per_person_count.get(pid, 0)
        per_person_count[pid] = n + 1

        start_frame = max(0, center_frame - half_clip)
        end_frame = min(total_frames - 1, center_frame + half_clip)

        output_path = os.path.join(
            output_dir, f"person_{pid:03d}_clip_{n:03d}.mp4"
        )

        if crop:
            frame_size = (out_size, out_size)
        else:
            frame_size = (width, height)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        last_box = None
        current = start_frame
        while current <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if crop:
                # bbox của đúng người này tại frame hiện tại (fallback: bbox trước đó)
                box = boxes.get(pid, {}).get(current, last_box)
                if box is not None:
                    last_box = box
                    x1, y1, x2, y2 = pad_and_clip_box(
                        box, width, height, pad_ratio
                    )
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size > 0:
                        out_frame = letterbox_square(crop_img, out_size)
                        writer.write(out_frame)
                # nếu chưa có bbox nào -> bỏ qua frame
            else:
                writer.write(frame)

            current += 1

        writer.release()
        cap.release()

        print(f"  Saved -> {output_path}")

    print("\n🎉 DONE")
    print(output_dir)


def letterbox_square(img, size, color=(0, 0, 0)):
    """Resize giữ tỉ lệ rồi pad thành ô vuông size x size."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


if __name__ == "__main__":
    export_pose_clips(
        video_path="input.mp4",
        output_dir="pose_clips",
        clip_seconds=5,
        crop=True,        # đổi False nếu muốn clip full-frame
    )
