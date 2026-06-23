"""
Batch driver: chạy pose_tracking trên TẤT CẢ video trong 1 thư mục.

- Lặp qua mọi video trong --input-dir (đệ quy)
- Mỗi video xuất vào  <out-dir>/<tên_video>/   (mỗi video 1 thư mục riêng)
- ĐA LUỒNG: --workers N  xử lý N video song song (mỗi process 1 model riêng)
- RESUME: manifest.json, chạy lại bỏ qua video đã xong
- --delete-source: xoá file video sau khi xử lý để tiết kiệm disk
- Tham số tốc độ lấy từ biến môi trường POSE_* (xem pose_tracking.apply_env_overrides)

Dùng:
    python batch_run.py --input-dir videos --out-dir out --workers 2 --delete-source
"""

import os
import json
import glob
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from pose_tracking import CFG, apply_env_overrides, auto_device, export_pose_clips

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm")


def list_videos(input_dir):
    files = []
    for ext in VIDEO_EXT:
        files += glob.glob(os.path.join(input_dir, "**", "*" + ext), recursive=True)
        files += glob.glob(os.path.join(input_dir, "**", "*" + ext.upper()), recursive=True)
    return sorted(set(files))


def process_one(vid, out_sub, cfg, delete_source):
    """Chạy 1 video (gọi trong process con). Trả (status, n_clips, error)."""
    try:
        export_pose_clips(vid, out_sub, cfg)
        n = len(glob.glob(os.path.join(out_sub, "*.mp4")))
        if delete_source:
            try:
                os.remove(vid)
            except OSError:
                pass
        return ("ok", n, None)
    except Exception as e:
        traceback.print_exc()
        return ("error", 0, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="thư mục chứa video")
    ap.add_argument("--out-dir", default="out", help="thư mục kết quả")
    ap.add_argument("--workers", type=int, default=1, help="số video xử lý song song")
    ap.add_argument("--delete-source", action="store_true", help="xoá video sau khi xử lý")
    ap.add_argument("--manifest", default=None, help="file manifest (mặc định <out-dir>/manifest.json)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.json")

    done = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            done = json.load(f)

    cfg = auto_device(apply_env_overrides(dict(CFG)))
    print("CONFIG:", {k: cfg[k] for k in
          ("model_path", "imgsz", "vid_stride", "device", "half", "clip_seconds")})

    videos = list_videos(args.input_dir)
    print(f"Tìm thấy {len(videos)} video | workers={args.workers}")

    # lập danh sách việc cần làm (bỏ video đã xong)
    tasks = []
    for vid in videos:
        key = os.path.relpath(vid, args.input_dir)
        if done.get(key, {}).get("status") == "ok":
            print(f"SKIP (đã xong): {key}")
            continue
        stem = os.path.splitext(os.path.basename(vid))[0]
        tasks.append((vid, key, os.path.join(args.out_dir, stem)))

    def save_manifest():
        with open(manifest_path, "w") as f:
            json.dump(done, f, indent=2)

    if args.workers <= 1:
        for vid, key, out_sub in tasks:
            print(f"\n==== {key} ====")
            status, n, err = process_one(vid, out_sub, cfg, args.delete_source)
            done[key] = {"status": status, "clips": n, "out": out_sub, "error": err}
            save_manifest()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(process_one, vid, out_sub, cfg, args.delete_source): (key, out_sub)
                for vid, key, out_sub in tasks
            }
            for fut in as_completed(futs):
                key, out_sub = futs[fut]
                status, n, err = fut.result()
                done[key] = {"status": status, "clips": n, "out": out_sub, "error": err}
                print(f"[{status}] {key} ({n} clip)")
                save_manifest()   # lưu sau mỗi video xong -> crash vẫn resume

    ok = sum(1 for v in done.values() if v.get("status") == "ok")
    err = sum(1 for v in done.values() if v.get("status") == "error")
    print(f"\n🎉 BATCH DONE: {ok} ok, {err} lỗi -> {args.out_dir}")


if __name__ == "__main__":
    main()
