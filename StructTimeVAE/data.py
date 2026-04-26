from typing import Optional, List, Sequence

import os, sys
import torch
import numpy as np
import io, pickle

from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

import cv2
import h5py


DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD  = [0.229, 0.224, 0.225]


def _get_max_frame(txt_file: str) -> int:
    """Return the maximum frame index listed in an annotation .txt file."""
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


def _build_dino_cache(
    video_map: dict,          # {video_name: mp4_path}
    frame_counts: dict,       # {video_name: max_frame_index}
    cache_path: str,
    map_size: int,
    device: torch.device,
    batch_size: int = 64,
):
    """
    For each video, extract frames up to max_frame_index, embed with DINOv2,
    and write to an h5py cache keyed by "{video_name}_{timestep}".
    Skips videos/frames already present in the cache.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model.eval().to(device)

    normalize = transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD)
    resize    = transforms.Resize((map_size, map_size))

    sample_saved = False  # save one resized frame before any embedding

    with h5py.File(cache_path, "a") as h5:
        for video_name, mp4_path in video_map.items():
            max_frame = frame_counts.get(video_name, -1)
            if max_frame < 0:
                continue

            cap = cv2.VideoCapture(mp4_path)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # we only need frames up to max_frame (annotation-driven upper bound)
            n_frames = min(max_frame + 1, total_video_frames)

            # collect which frame indices still need embedding
            needed = []
            for t in range(n_frames):
                key = f"{video_name}_{t}"
                if key not in h5:
                    needed.append(t)

            if not needed:
                cap.release()
                sys.stdout.write(f"\r[DINOv2 cache] {video_name}: all {n_frames} frames already cached, skipping.")
                sys.stdout.flush()
                continue

            # read all needed frames from video
            frames_by_idx = {}
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            vid_pos = 0
            needed_set = set(needed)
            while vid_pos <= max(needed):
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if vid_pos in needed_set:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames_by_idx[vid_pos] = frame_rgb
                vid_pos += 1
            cap.release()

            # save the first frame of the first video as a sanity check image
            if not sample_saved and frames_by_idx:
                first_idx = sorted(frames_by_idx.keys())[0]
                img = Image.fromarray(frames_by_idx[first_idx])
                img_resized = img.resize((map_size, map_size), Image.BILINEAR)
                sample_path = os.path.join(os.path.dirname(cache_path), "sample_frame.png")
                img_resized.save(sample_path)
                print(f"\n[DINOv2 cache] Sample frame saved to {sample_path}")
                sample_saved = True

            # embed in batches
            frame_indices = sorted(frames_by_idx.keys())
            for batch_start in range(0, len(frame_indices), batch_size):
                batch_idxs = frame_indices[batch_start:batch_start + batch_size]
                tensors = []
                for t in batch_idxs:
                    img = Image.fromarray(frames_by_idx[t])
                    img_t = TF.to_tensor(img).float()  # [3, H, W]
                    img_t = resize(img_t)
                    img_t = normalize(img_t)
                    tensors.append(img_t)

                batch_tensor = torch.stack(tensors).to(device)  # [B, 3, 224, 224]
                with torch.no_grad():
                    embeddings = model(batch_tensor)             # [B, 768]
                embeddings_np = embeddings.cpu().numpy().astype(np.float32)

                for i, t in enumerate(batch_idxs):
                    key = f"{video_name}_{t}"
                    h5.create_dataset(key, data=embeddings_np[i], compression="gzip")

                done = batch_start + len(batch_idxs)
                sys.stdout.write(
                    f"\r[DINOv2 cache] {video_name}: {done}/{len(frame_indices)} frames embedded"
                )
                sys.stdout.flush()

    print(f"\n[DINOv2 cache] Done. Cache saved to {cache_path}")


class Dataloader(torch.utils.data.Dataset):

    class FixedNumberBatchSampler(torch.utils.data.sampler.BatchSampler):
        def __init__(self, n_batches, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_batches = n_batches
            self.sampler_iter = None
        def __iter__(self):
            counter = 0
            batch = []
            while True:
                if counter >= self.n_batches:
                    break
                if self.sampler_iter is None:
                    self.sampler_iter = iter(self.sampler)
                try:
                    idx = next(self.sampler_iter)
                except StopIteration:
                    self.sampler_iter = None
                    if self.drop_last: batch = []
                    continue
                batch.append(idx)
                if len(batch) == self.batch_size:
                    counter += 1
                    yield batch
                    batch = []

    def __init__(self,
        files: List[str], ob_horizon: int, pred_horizon: int,
        batch_size: int, drop_last: bool=False, shuffle: bool=False, batches_per_epoch=None,
        frameskip: int=1, inclusive_groups: Optional[Sequence]=None,
        batch_first: bool=False, seed: Optional[int]=None,
        device: Optional[torch.device]=None,
        flip: bool=False, rotate: bool=False, scale: bool=False,

        min_ob_horizon: Optional[int]=None, traj_max_overlap: Optional[int]=None,
        ob_radius: Optional[int]=None,
        map_dir: Optional[str]=None, map_size: int=224, map_scale: int=1, preload_map: bool=True,

        # --- new args ---
        embedding_dir: Optional[str]=None,   # if set, load pre-computed embeddings from here
        cache_path: Optional[str]=None,       # path to h5py cache file (default: alongside data)
        dino_device: Optional[torch.device]=None,  # device to run DINOv2 on (defaults to `device`)
        dino_batch_size: int=64,
    ):
        super().__init__()

        # --- observation/prediction horizons ---
        self.min_ob_horizon = ob_horizon if min_ob_horizon is None else min_ob_horizon
        self.ob_horizon = ob_horizon
        if self.ob_horizon < self.min_ob_horizon:
            self.min_ob_horizon, self.ob_horizon = self.ob_horizon, self.min_ob_horizon
        self.pred_horizon = pred_horizon
        self.frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1
        self.batch_first = batch_first

        # --- map options ---
        self.use_map = map_dir is not None or embedding_dir is not None
        self.map_scale = map_scale
        self.map_size = map_size
        self.flip = flip and not self.use_map
        self.rotate = rotate and not self.use_map
        self.scale = scale and not self.use_map

        # --- device & trajectory options ---
        self.device = device
        self.dino_device = dino_device if dino_device is not None else device
        self.traj_max_overlap = traj_max_overlap
        self.ob_radius = ob_radius
        self.preload_map = preload_map
        self.embedding_dir = embedding_dir
        self.dino_batch_size = dino_batch_size

        # --- inclusive groups handling ---
        if inclusive_groups is None:
            inclusive_groups = []
        self.inclusive_groups = inclusive_groups

        # --- scan .txt files ---
        print("Scanning files...")
        files_ = []
        for path in files:
            if os.path.isdir(path):
                files_.extend([os.path.join(root, f)
                            for root, _, fs in os.walk(path) for f in fs if f.endswith(".txt")])
            elif os.path.exists(path):
                files_.append(path)
        data_files = sorted(files_)
        assert len(data_files) > 0, f"No valid files found from {files}"

        # ---------------------------------------------------------------
        # MAP / EMBEDDING SETUP
        # ---------------------------------------------------------------
        self.map = dict()           # video_name -> h5py cache path (string) used at getitem
        self.cache_path = None      # single h5py file path

        # Store video_map so visualizer can resolve mp4 paths by video name
        self.video_map = {}         # {video_name: mp4_path}

        if embedding_dir is not None:
            # --- use pre-computed embeddings from directory ---
            # expect files named {video_name}.h5 or a single combined cache.h5
            print(f"Using pre-computed embeddings from {embedding_dir}")
            self.use_map = True
            # resolve cache: either one combined file or per-video files
            combined = os.path.join(embedding_dir, "cache.h5")
            if os.path.exists(combined):
                self.cache_path = combined
                print(f"  Found combined cache: {combined}")
            else:
                # per-video h5 files; we'll look them up at runtime
                self.embedding_dir = embedding_dir
                self.cache_path = None  # signals per-video lookup

        else:
            # --- video-based map: find MP4s and build DINOv2 cache ---

            # Always scan for MP4s so video_map is populated for the visualizer,
            # regardless of whether the cache already exists.
            video_map    = {}   # {video_name: mp4_path}
            frame_counts = {}   # {video_name: max_annotation_frame}

            for txt_file in data_files:
                video_folder = os.path.dirname(txt_file)
                video_name   = os.path.basename(video_folder)
                mp4_path     = os.path.join(video_folder, f"{video_name}.mp4")

                # fallback: any .mp4 in the same folder
                if not os.path.exists(mp4_path):
                    candidates = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
                    if candidates:
                        mp4_path = os.path.join(video_folder, candidates[0])
                    else:
                        mp4_path = None

                if mp4_path is None:
                    continue

                max_frame = _get_max_frame(txt_file)
                if max_frame < 0:
                    continue

                if video_name not in frame_counts or max_frame > frame_counts[video_name]:
                    frame_counts[video_name] = max_frame
                    video_map[video_name]    = mp4_path

            if video_map:
                self.video_map = video_map  # store for visualizer lookup

            # if cache already exists at the given path, just use it directly
            if cache_path is not None and os.path.exists(cache_path):
                print(f"Using existing cache at {cache_path}")
                self.cache_path = cache_path
                self.use_map = True

            else:
                if video_map:
                    self.use_map = True

                    # determine cache path
                    if cache_path is not None:
                        self.cache_path = cache_path
                    else:
                        # place cache next to the first data file
                        self.cache_path = os.path.join(
                            os.path.dirname(data_files[0]), "dino_embedding_cache.h5"
                        )

                    print(f"Found {len(video_map)} video(s). Building DINOv2 cache at {self.cache_path} ...")
                    _build_dino_cache(
                        video_map    = video_map,
                        frame_counts = frame_counts,
                        cache_path   = self.cache_path,
                        map_size     = self.map_size,
                        device       = self.dino_device if self.dino_device else torch.device("cpu"),
                        batch_size   = self.dino_batch_size,
                    )
                else:
                    print("No MP4 files found alongside .txt files; map disabled.")
                    self.use_map = False

        # NOTE: h5 handle is opened AFTER trajectory loading to avoid pickle errors
        # in ProcessPoolExecutor workers
        self._h5_handle = None

        # ---------------------------------------------------------------
        # LOAD TRAJECTORY DATA (parallel)
        # ---------------------------------------------------------------
        sys.stdout.write(f"\r\033[K Loading data files...0/{len(data_files)}")
        done = 0
        data = []
        max_workers = min(len(data_files), torch.get_num_threads(), 20)
        with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers) as p:
            futures = [p.submit(self.__class__.load, self, f, self.inclusive_groups) for f in data_files]
            for fut in as_completed(futures):
                done += 1
                sys.stdout.write(f"\r\033[K Loading data files...{done}/{len(data_files)}")
            for fut in futures:
                item = fut.result()
                if item:
                    data.extend(item)
        sys.stdout.write(f"\r\033[K Loading data files...{len(data_files)}/{len(data_files)} ")

        # --- safe augmentation check ---
        if len(data) > 0 and len(data[-1]) > 3 and data[-1][3] is not None:
            if isinstance(data[-1][3], (list, tuple)) and len(data[-1][3]) > 1:
                if data[-1][3][1] is not None:
                    self.flip = False
                    self.rotate = False
                    self.scale = False

        self.data = np.array(data, dtype=object)
        del data
        print(f"\nLoaded {len(self.data)} trajectories.")

        # open h5 handle now — safe since ProcessPoolExecutor is done
        if self.cache_path and os.path.exists(self.cache_path):
            self._h5_handle = h5py.File(self.cache_path, "r")

        self.rng = np.random.RandomState()
        if seed: self.rng.seed(seed)

        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(self)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self)
        if batches_per_epoch is None:
            self.batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
            self.batches_per_epoch = len(self.batch_sampler)
        else:
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(batches_per_epoch, sampler, batch_size, drop_last)
            self.batches_per_epoch = batches_per_epoch

    # ------------------------------------------------------------------
    # H5 HANDLE: reopen per worker (h5py handles are not fork-safe)
    # ------------------------------------------------------------------
    def _get_h5(self):
        if self._h5_handle is None and self.cache_path and os.path.exists(self.cache_path):
            self._h5_handle = h5py.File(self.cache_path, "r")
        return self._h5_handle

    def _lookup_embedding(self, video_name: str, timestep: int) -> Optional[np.ndarray]:
        """Return the DINOv2 embedding for (video_name, timestep), or None if missing."""
        key = f"{video_name}_{timestep}"

        if self.cache_path:
            h5 = self._get_h5()
            if h5 and key in h5:
                return h5[key][:]

        # fallback: per-video h5 in embedding_dir
        if self.embedding_dir:
            per_video_path = os.path.join(self.embedding_dir, f"{video_name}.h5")
            if os.path.exists(per_video_path):
                with h5py.File(per_video_path, "r") as f:
                    if key in f:
                        return f[key][:]

        return None

    # ------------------------------------------------------------------
    # COLLATE
    # ------------------------------------------------------------------
    def collate_fn(self, batch):
        if not batch:
            raise ValueError("Received an empty batch in collate_fn — cannot process.")

        X, Y, NEIGHBOR, EMBEDDINGS, L = [], [], [], [], []
        # NEW: collect per-item video metadata for the visualizer
        VIDEO_NAMES  = []   # list[str]  — one entry per batch item
        START_FRAMES = []   # list[int]  — one entry per batch item

        for item in batch:
            hist, future, neighbor = item[0], item[1], item[2]
            seq_len = item[-1]

            if seq_len is None:
                raise ValueError(f"Trajectory has no seq_len: {item}")

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
            L.append(seq_len)

            if self.use_map:
                # item[3] = (video_name, timestep_list)  — set in __getitem__
                video_name, timesteps = item[3]
                embs = []
                for t in timesteps:
                    emb = self._lookup_embedding(video_name, t)
                    if emb is None:
                        emb = np.zeros(768, dtype=np.float32)
                    embs.append(emb)
                EMBEDDINGS.append(np.stack(embs, axis=0))  # [T, 768]

                # NEW: store video name and start frame for visualizer
                VIDEO_NAMES.append(video_name)
                START_FRAMES.append(timesteps[0])  # first annotation timestep in obs window
            else:
                VIDEO_NAMES.append(None)
                START_FRAMES.append(0)

        # pad neighbors
        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors)
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0,0),(0,max_neighbors-n),(0,0)), "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]

        stack_dim = 0 if self.batch_first else 1
        x           = torch.tensor(np.stack(X,        stack_dim), dtype=torch.float32, device=self.device)
        y           = torch.tensor(np.stack(Y,        stack_dim), dtype=torch.float32, device=self.device)
        neighbor    = torch.tensor(np.stack(NEIGHBOR, stack_dim), dtype=torch.float32, device=self.device)
        seq_len_tensor = torch.tensor(L, dtype=torch.long, device=self.device)

        ret = [x, y, neighbor]
        if self.use_map:
            # always stack as [B, T, 768] — model expects batch-first for embeddings
            emb_array = np.stack(EMBEDDINGS, axis=0)
            ret.append(torch.tensor(emb_array, dtype=torch.float32, device=self.device))
        ret.append(seq_len_tensor)
        # NEW: append video metadata as plain Python lists (not tensors)
        ret.append(VIDEO_NAMES)    # list[str | None], length B
        ret.append(START_FRAMES)   # list[int],         length B

        return ret

    # ------------------------------------------------------------------
    # DATASET PROTOCOL
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        hist, future, neighbor = item[0], item[1], item[2]

        seq_len = hist.shape[0] if hist is not None else 0

        if future is not None:
            if future.ndim == 3:
                future = future.transpose(1, 0, 2).copy()
            elif future.ndim == 2:
                future = future[None, :, :].copy()
            else:
                raise ValueError(f"Unexpected future shape: {future.shape}")

        if self.use_map:
            map_info = item[3]
            video_name = map_info[0] if isinstance(map_info, tuple) else map_info
            # build list of annotation timesteps for the observation window
            # item[4] carries start_tid; if absent fall back to 0
            start_tid = item[5] if len(item) > 5 and item[5] is not None else 0
            timesteps = list(range(start_tid, start_tid + self.ob_horizon * self.frameskip, self.frameskip))
            return hist, future, neighbor, (video_name, timesteps), seq_len

        return hist, future, neighbor, seq_len

    # ------------------------------------------------------------------
    # STATIC HELPERS (called inside ProcessPoolExecutor workers)
    # ------------------------------------------------------------------
    @staticmethod
    def load(self, filename, inclusive_groups=None):
        import numpy as np
        import os

        if os.path.isdir(filename):
            return None

        min_horizon = (self.min_ob_horizon - 1 + self.pred_horizon) * self.frameskip

        with open(filename, "r") as f:
            data = self.load_traj(f)
        data = self.extend(data, self.frameskip)

        time = np.sort(list(data.keys()))
        if len(time) < min_horizon + 1:
            return None

        ob_radius = None if self.ob_radius is None else self.ob_radius + 0.5

        info_file = filename.replace(".txt", ".info")
        if os.path.exists(info_file):
            ts, maps = [], []
            with open(info_file, "r") as map_file:
                for line in map_file.readlines():
                    if self.use_map:
                        t, map_name = line.split()
                        maps.append(map_name)
                    else:
                        t = line.split()[0]
                    ts.append(np.where(time == int(t))[0][0])
            segments = [(t, ts[i] if i < len(ts) else len(time)) for i, t in enumerate(ts, 1)]
        else:
            segments = [(0, len(time))]
            video_folder = os.path.basename(os.path.dirname(filename))
            maps = [video_folder]

        items = []
        ob_gap = self.ob_horizon - self.min_ob_horizon

        for seg, (s, e) in enumerate(segments):
            tid_curr  = s + (self.min_ob_horizon - 1) * self.frameskip
            tid_final = e - self.pred_horizon * self.frameskip
            timestamp = dict()

            while tid_curr < tid_final:
                tid_start = tid_curr - (self.ob_horizon - 1) * self.frameskip
                tid_end   = tid_curr + self.pred_horizon * self.frameskip
                idx_ego   = []
                idx_all   = []
                first_frame = dict()

                for i, tid in enumerate(range(tid_start, tid_end + 1, self.frameskip)):
                    if tid < s:
                        continue
                    if tid not in data:
                        continue

                    t = time[tid]
                    idx_curr = list(data[t].keys())

                    if not idx_curr:
                        idx_ego = []
                        if i >= ob_gap:
                            break

                    idx_ego = np.intersect1d(idx_ego, idx_curr) if len(idx_ego) > 0 else idx_curr

                    for idx in idx_curr:
                        if idx not in first_frame:
                            first_frame[idx] = (i, tid)

                    if i <= ob_gap:
                        idx_ego = np.union1d(idx_ego, idx_curr)

                    if i >= ob_gap and len(idx_ego) == 0:
                        break

                    idx_all.extend(data[t].keys())

                if self.traj_max_overlap is not None and self.traj_max_overlap < self.ob_horizon + self.pred_horizon:
                    overlapped = []
                    for idx in idx_ego:
                        if idx in timestamp and first_frame[idx][1] < timestamp[idx]:
                            overlapped.append(idx)
                        else:
                            timestamp[idx] = tid_end - (self.traj_max_overlap - 1) * self.frameskip
                    idx_ego = np.setdiff1d(idx_ego, overlapped)

                if len(idx_ego) == 0:
                    tid_curr += 1
                    continue

                data_dim     = 6
                neighbor_idx = np.setdiff1d(idx_all, idx_ego)

                agents = np.array([
                    [
                        data[time[tid]][i][:data_dim] if time[tid] in data and i in data[time[tid]] else [1e9]*data_dim
                        for i in idx_ego
                    ] + [
                        data[time[tid]][j][:data_dim] if time[tid] in data and j in data[time[tid]] else [1e9]*data_dim
                        for j in neighbor_idx
                    ]
                    for tid in range(tid_start, tid_end + 1, self.frameskip)
                ])

                for i, idx in enumerate(idx_ego):
                    hist    = agents[:self.ob_horizon, i]
                    future  = agents[self.ob_horizon:, i, :2]
                    neighbor = agents[:, [d for d in range(agents.shape[1]) if d != i]]

                    if ob_radius is not None:
                        dp    = neighbor[:, :, :2] - agents[:, i:i+1, :2]
                        dist  = np.linalg.norm(dp, axis=-1)
                        valid = np.any(dist <= ob_radius, axis=0)
                        neighbor = neighbor[:, valid]

                    if first_frame[idx][0] > 0:
                        seq_len     = self.ob_horizon - first_frame[idx][0]
                        padded_hist = np.zeros_like(hist)
                        if seq_len > 0:
                            padded_hist[:seq_len] = hist[-seq_len:]
                        hist = padded_hist

                        padded_neighbor = np.zeros_like(neighbor)
                        if neighbor.shape[0] >= self.ob_horizon:
                            padded_neighbor[self.ob_horizon:] = neighbor[self.ob_horizon:]
                            if seq_len > 0:
                                padded_neighbor[:seq_len] = neighbor[self.ob_horizon - seq_len:self.ob_horizon]
                        neighbor = padded_neighbor
                    else:
                        seq_len = self.ob_horizon

                    # store actual annotation timestep for the start of the observation window
                    # so __getitem__ can look up the right video frames
                    start_annotation_tid = int(time[tid_start]) if tid_start >= 0 and tid_start < len(time) else 0

                    map_name = maps[seg] if self.use_map else None
                    items.append((
                        hist.astype(np.float32),
                        future.astype(np.float32),
                        neighbor.astype(np.float32),
                        map_name,
                        seq_len if self.min_ob_horizon != self.ob_horizon else None,
                        start_annotation_tid,   # actual video timestep for frame lookup
                    ))

                tid_curr += 1

        return items

    @staticmethod
    def load_map(map_file, compressed=False):
        with Image.open(map_file) as img:
            img = img.convert('RGB')
            semantic_map = np.array(img)
            H = semantic_map.shape[0]

        if compressed:
            buf = io.BytesIO()
            np.savez_compressed(buf, M=semantic_map, H=H)
            buf.seek(0)
            return buf

        semantic_map = torch.tensor(semantic_map, dtype=torch.float32)
        return semantic_map, H

    def extend(self, data, frameskip):
        time = np.sort(list(data.keys()))
        dts  = np.unique(time[1:] - time[:-1])
        dt   = dts.min()
        if np.any(dts % dt != 0):
            raise ValueError("Inconsistent frame interval:", dts)
        i = 0
        while i < len(time)-1:
            if time[i+1] - time[i] != dt:
                time = np.insert(time, i+1, time[i]+dt)
            i += 1
        for tid, t in enumerate(time):
            removed = []
            if t not in data: data[t] = {}
            for idx in data[t].keys():
                t0 = time[tid-frameskip] if tid >= frameskip else None
                t1 = time[tid+frameskip] if tid+frameskip < len(time) else None
                if (t0 is None or t0 not in data or idx not in data[t0]) and \
                (t1 is None or t1 not in data or idx not in data[t1]):
                    removed.append(idx)
            for idx in removed:
                data[t].pop(idx)
        for tid in range(len(time)-frameskip):
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                x0, y0 = data[t0][i][0], data[t0][i][1]
                x1, y1 = data[t1][i][0], data[t1][i][1]
                vx, vy = x1-x0, y1-y0
                data[t1][i].insert(2, vx)
                data[t1][i].insert(3, vy)
                if tid < frameskip or i not in data[time[tid-frameskip]]:
                    data[t0][i].insert(2, vx)
                    data[t0][i].insert(3, vy)
        for tid in range(len(time)-frameskip):
            t_1 = None if tid < frameskip else time[tid-frameskip]
            t0  = time[tid]
            t1  = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                if len(data[t0][i]) < 4 or len(data[t1][i]) < 4: continue
                vx0, vy0 = data[t0][i][2], data[t0][i][3]
                vx1, vy1 = data[t1][i][2], data[t1][i][3]
                ax, ay = vx1-vx0, vy1-vy0
                data[t1][i].insert(4, ax)
                data[t1][i].insert(5, ay)
                if t_1 is None or i not in data[t_1]:
                    data[t0][i].insert(4, ax)
                    data[t0][i].insert(5, ay)
        return data

    def load_traj(self, file):
        data = {}
        for row in file.readlines():
            item = row.strip().split()
            if not item:
                continue
            track_id = int(item[0])
            xmin  = float(item[1])
            ymin  = float(item[2])
            xmax  = float(item[3])
            ymax  = float(item[4])
            frame = int(item[5])
            label = item[9].replace('"','').replace("'",'').strip().lower()
            if label != "pedestrian":
                continue
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            if frame not in data:
                data[frame] = {}
            data[frame][track_id] = [x, y]
        return data