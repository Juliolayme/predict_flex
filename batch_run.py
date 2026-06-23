"""
Batch driver: chạy pose_tracking trên TẤT CẢ video trong 1 thư mục.

- Lặp qua mọi *.mp4/*.mov/*.mkv trong --input-dir
- Mỗi video xuất vào  <out-dir>/<tên_video>/
- Ghi manifest.json để RESUME: chạy lại sẽ bỏ qua video đã xong
- Mọi tham số tốc độ lấy từ biến môi trường POSE_* (xem pose_tracking.apply_env_overrides)

Dùng:
    python batch_run.py --input-dir videos --out-dir out
"""

import os
import json
import glob
import argparse
import traceback

from pose_tracking import CFG, apply_env_overrides, auto_device, export_pose_clips

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm")


def list_videos(input_dir):
    files = []
    for ext in VIDEO_EXT:
        files += glob.glob(os.path.join(input_dir, "**", "*" + ext), recursive=True)
        files += glob.glob(os.path.join(input_dir, "**", "*" + ext.upper()), recursive=True)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="thư mục chứa video")
    ap.add_argument("--out-dir", default="out", help="thư mục kết quả")
    ap.add_argument("--manifest", default=None, help="file manifest (mặc định <out-dir>/manifest.json)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.json")

    # nạp manifest cũ để resume
    done = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            done = json.load(f)

    cfg = auto_device(apply_env_overrides(dict(CFG)))
    print("CONFIG:", {k: cfg[k] for k in
          ("model_path", "imgsz", "vid_stride", "device", "half", "clip_seconds")})

    videos = list_videos(args.input_dir)
    print(f"Tìm thấy {len(videos)} video trong {args.input_dir}")

    for vid in videos:
        key = os.path.relpath(vid, args.input_dir)
        if done.get(key, {}).get("status") == "ok":
            print(f"SKIP (đã xong): {key}")
            continue

        stem = os.path.splitext(os.path.basename(vid))[0]
        out_sub = os.path.join(args.out_dir, stem)
        print(f"\n==== {key} ====")
        try:
            export_pose_clips(vid, out_sub, cfg)
            n_clips = len(glob.glob(os.path.join(out_sub, "*.mp4")))
            done[key] = {"status": "ok", "clips": n_clips, "out": out_sub}
        except Exception as e:
            traceback.print_exc()
            done[key] = {"status": "error", "error": str(e)}

        # lưu manifest sau MỖI video -> crash vẫn resume được
        with open(manifest_path, "w") as f:
            json.dump(done, f, indent=2)

    ok = sum(1 for v in done.values() if v.get("status") == "ok")
    err = sum(1 for v in done.values() if v.get("status") == "error")
    print(f"\n🎉 BATCH DONE: {ok} ok, {err} lỗi -> {args.out_dir}")


if __name__ == "__main__":
    main()
