# build_cache.py
# Run this once before training to pre-build the DINOv2 embedding cache.
# Usage:
#   python build_cache.py --data /path/to/data --cache /path/to/cache.h5 --device cuda:0
#   python build_cache.py --data /path/to/data --cache /path/to/cache.h5 --device cuda:1 --frameskip 6

import os
import sys
import argparse
import subprocess
import shutil
import threading
import queue
import torch
import numpy as np
import h5py
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD  = [0.229, 0.224, 0.225]
_SENTINEL   = object()


def get_max_frame(txt_file: str) -> int:
    max_frame = -1
    with open(txt_file, "r") as f:
        for row in f:
            item = row.strip().split()
            if not item:
                continue
            try:
                frame = int(item[5])
                if frame > max_frame:
                    max_frame = frame
            except (IndexError, ValueError):
                continue
    return max_frame


def collect_videos(data_dirs):
    video_map    = {}
    frame_counts = {}

    for data_dir in data_dirs:
        for root, _, files in os.walk(data_dir):
            for f in files:
                if not f.endswith(".txt"):
                    continue

                txt_file   = os.path.join(root, f)
                video_name = os.path.basename(root)

                mp4_path = os.path.join(root, f"{video_name}.mp4")
                if not os.path.exists(mp4_path):
                    candidates = [c for c in os.listdir(root) if c.endswith(".mp4")]
                    mp4_path   = os.path.join(root, candidates[0]) if candidates else None

                if mp4_path is None:
                    continue

                max_frame = get_max_frame(txt_file)
                if max_frame < 0:
                    continue

                if video_name not in frame_counts or max_frame > frame_counts[video_name]:
                    frame_counts[video_name] = max_frame
                    video_map[video_name]    = mp4_path

    return video_map, frame_counts


def extract_frames_ffmpeg(mp4_path, out_dir, max_frame, frameskip, map_size):
    """
    Extract every frameskip-th frame up to max_frame using ffmpeg.
    Files named by actual frame index: {frame_idx}.jpg
    """
    os.makedirs(out_dir, exist_ok=True)

    select_filter = (
        f"select='not(mod(n\\,{frameskip}))*lte(n\\,{max_frame})',"
        f"scale={map_size}:{map_size}"
    )

    cmd = [
        "ffmpeg", "-i", mp4_path,
        "-vf", select_filter,
        "-vsync", "0",
        "-frame_pts", "1",
        "-q:v", "2",
        os.path.join(out_dir, "%d.jpg"),
        "-y",
        "-loglevel", "error"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ffmpeg] Error: {result.stderr}")
        return False
    return True


def _jpeg_reader(out_dir, needed_sorted, chunk_size, frame_queue):
    """Background thread: reads JPEG frames from disk in chunks."""
    for chunk_start in range(0, len(needed_sorted), chunk_size):
        chunk_idxs   = needed_sorted[chunk_start:chunk_start + chunk_size]
        frames_chunk = {}

        for t in chunk_idxs:
            jpg_path = os.path.join(out_dir, f"{t}.jpg")
            if os.path.exists(jpg_path):
                img = Image.open(jpg_path).convert("RGB")
                frames_chunk[t] = np.array(img)

        frame_queue.put((chunk_idxs, frames_chunk))

    frame_queue.put(_SENTINEL)


def build_cache(video_map, frame_counts, cache_path, map_size, device, batch_size,
                chunk_size, frameskip, frames_root):

    print(f"Loading DINOv2 ViT-B/14 on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model.eval().to(device)

    normalize = transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD)
    resize    = transforms.Resize((map_size, map_size))

    sample_saved = False
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    os.makedirs(frames_root, exist_ok=True)

    with h5py.File(cache_path, "a") as h5:
        for video_idx, (video_name, mp4_path) in enumerate(video_map.items()):
            max_frame = frame_counts[video_name]

            all_needed = list(range(0, max_frame + 1, frameskip))
            needed     = [t for t in all_needed if f"{video_name}_{t}" not in h5]

            if not needed:
                print(f"[{video_idx+1}/{len(video_map)}] {video_name}: already cached, skipping.")
                continue

            print(f"[{video_idx+1}/{len(video_map)}] {video_name}: extracting {len(all_needed)} frames with ffmpeg...")

            out_dir = os.path.join(frames_root, video_name)
            ok = extract_frames_ffmpeg(mp4_path, out_dir, max_frame, frameskip, map_size)
            if not ok:
                print(f"  Skipping {video_name} due to ffmpeg error.")
                continue

            if not sample_saved:
                first_jpg = os.path.join(out_dir, f"{all_needed[0]}.jpg")
                if os.path.exists(first_jpg):
                    sample_path = os.path.join(os.path.dirname(cache_path), "sample_frame.png")
                    Image.open(first_jpg).save(sample_path)
                    print(f"  Sample frame saved to {sample_path}")
                    sample_saved = True

            print(f"  Embedding {len(needed)} frames...")
            needed_sorted = sorted(needed)
            done_total    = 0

            frame_queue = queue.Queue(maxsize=2)
            reader = threading.Thread(
                target=_jpeg_reader,
                args=(out_dir, needed_sorted, chunk_size, frame_queue),
                daemon=True
            )
            reader.start()

            while True:
                item = frame_queue.get()
                if item is _SENTINEL:
                    break

                chunk_idxs, frames_chunk = item

                for batch_start in range(0, len(chunk_idxs), batch_size):
                    batch_idxs = chunk_idxs[batch_start:batch_start + batch_size]

                    tensors = []
                    for t in batch_idxs:
                        if t not in frames_chunk:
                            continue
                        img_t = TF.to_tensor(Image.fromarray(frames_chunk[t])).float()
                        img_t = resize(img_t)
                        img_t = normalize(img_t)
                        tensors.append((t, img_t))

                    if not tensors:
                        continue

                    ts, imgs     = zip(*tensors)
                    batch_tensor = torch.stack(imgs).to(device)
                    with torch.no_grad():
                        embeddings = model(batch_tensor)
                    embeddings_np = embeddings.cpu().numpy().astype(np.float32)

                    for i, t in enumerate(ts):
                        h5.create_dataset(f"{video_name}_{t}", data=embeddings_np[i], compression="gzip")

                    done_total += len(ts)
                    sys.stdout.write(f"\r  {done_total}/{len(needed)} frames embedded")
                    sys.stdout.flush()

                del frames_chunk

            reader.join()
            print()

            # delete extracted frames immediately to free disk space
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"  Cleaned up frames for {video_name}")

    print(f"\nDone. Cache saved to {cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-build DINOv2 embedding cache.")
    parser.add_argument("--data",        nargs="+", required=True,  help="Data directories containing .txt and MP4 files")
    parser.add_argument("--cache",       type=str,  required=True,  help="Path to output .h5 cache file")
    parser.add_argument("--frames_root", type=str,  default="/tmp/sdd_frames", help="Temp directory for extracted frames (default: /tmp/sdd_frames)")
    parser.add_argument("--device",      type=str,  default="cuda", help="Device for DINOv2 (default: cuda)")
    parser.add_argument("--batch_size",  type=int,  default=64,     help="Frames per DINOv2 batch (default: 64)")
    parser.add_argument("--chunk_size",  type=int,  default=500,    help="Frames per read chunk (default: 500)")
    parser.add_argument("--map_size",    type=int,  default=224,    help="Resize frames to this size (default: 224)")
    parser.add_argument("--frameskip",   type=int,  default=6,      help="Extract every nth frame (default: 6, matching 30fps->5fps)")
    parser.add_argument("--cpu_cores",   type=int,  default=4,      help="Number of CPU cores for torch (default: 4)")
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_cores)
    torch.set_num_interop_threads(args.cpu_cores)
    try:
        os.sched_setaffinity(0, set(range(args.cpu_cores)))
        print(f"Pinned to {args.cpu_cores} core(s).")
    except AttributeError:
        print(f"torch thread cap applied ({args.cpu_cores} thread(s)).")

    device = torch.device(args.device)

    print(f"Scanning {args.data} for videos...")
    video_map, frame_counts = collect_videos(args.data)
    print(f"Found {len(video_map)} video(s).")

    if not video_map:
        print("No videos found. Check your --data path.")
        sys.exit(1)

    for name, path in video_map.items():
        print(f"  {name}: {path} (max frame {frame_counts[name]})")

    build_cache(
        video_map    = video_map,
        frame_counts = frame_counts,
        cache_path   = args.cache,
        map_size     = args.map_size,
        device       = device,
        batch_size   = args.batch_size,
        chunk_size   = args.chunk_size,
        frameskip    = args.frameskip,
        frames_root  = args.frames_root,
    )