import torch
import numpy as np
from torch.utils.data import Dataset
import os
import re
from PIL import Image


class BatchGeneratorTCN(Dataset):
    def __init__(
        self, mode, actions_dict, goals_dict, sample_rate, vid_list_file, pred_perc, obs_perc, args
    ):
        print(f"Dataset: {args.ds}")
        print(f"Mode : {mode}")
        print(f"Vid list file : {vid_list_file}")
        print(f"Observation perc : {obs_perc}")

        # Set params
        self.mode = mode
        self.num_highlevel_classes = int(args.num_highlevel_classes)
        self.num_classes = args.num_classes
        self.actions_dict = actions_dict
        self.goals_dict = goals_dict
        self.dataset = args.ds

        self.sample_rate = sample_rate
        self.obs_perc = obs_perc
        self.pred_perc = pred_perc
        self.part_obs = args.part_obs

        # sanity check
        if self.mode != "train":
            assert self.obs_perc != 0

        self.features_path = args.features_path
        self.gt_path = args.gt_path

        # ---- NEW: frames root (breakfast_png) ----
        # expected structure:
        #   frames_root/P03/cam01/P03_cereals/00000.png ...
        self.frames_root = getattr(
            args,
            "frames_root",
            "/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff/breakfast_png",
        )
        self.qwen_max_side = getattr(args, "qwen_max_side", 336)
        self._img_cache = {}
        self._img_cache_cap = getattr(args, "img_cache_cap", 2048)

        # Annotations
        file_ptr = open(vid_list_file, "r")
        list_of_vid = file_ptr.read().split("\n")[:-1]
        file_ptr.close()

        self.list_of_examples = list()
        for vid in list_of_vid:
            if obs_perc != 0:
                self.list_of_examples.append([vid, obs_perc])
            else:
                assert self.mode == "train"
                self.list_of_examples.append([vid, 0.2])
                self.list_of_examples.append([vid, 0.3])
                self.list_of_examples.append([vid, 0.5])

    def __len__(self):
        return len(self.list_of_examples)

    def label_to_id(self, content):
        classes = np.zeros(len(content))
        for i in range(len(content)):
            classes[i] = self.actions_dict[content[i]]
        return classes

    def goal_label_to_id(self, content):
        classes = self.goals_dict[content]
        return np.array([classes])

    # ---------- NEW: image utils ----------
    def _resize_max_side(self, img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if s <= max_side:
            return img
        scale = max_side / float(s)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.BICUBIC)

    def _load_resized_image(self, path: str, max_side: int) -> Image.Image:
        if path in self._img_cache:
            return self._img_cache[path]
        img = Image.open(path).convert("RGB")
        img = self._resize_max_side(img, max_side)
        if len(self._img_cache) >= self._img_cache_cap:
            self._img_cache.pop(next(iter(self._img_cache)))
        self._img_cache[path] = img
        return img

    def _sorted_frame_paths(self, folder: str):
        if not os.path.isdir(folder):
            return []
        exts = (".png", ".jpg", ".jpeg")
        files = [os.path.join(folder, x) for x in os.listdir(folder) if x.lower().endswith(exts)]
        if len(files) == 0:
            return []
        num_re = re.compile(r"(\d+)")
        def key_fn(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            m = num_re.search(stem)
            return int(m.group(1)) if m else 10**18
        files.sort(key=key_fn)
        return files

    def __getitem__(self, idx):
        # Load feats and anns
        file_name = self.list_of_examples[idx][0].split(".")[0]
        features_name = os.path.join(self.features_path, file_name + ".npy")
        gt_name = os.path.join(self.gt_path, self.list_of_examples[idx][0])

        goals_label = (os.path.basename(gt_name)).split('_')[-1].split('.')[0]

        assert os.path.exists(features_name), "{} features file not found".format(features_name)
        assert os.path.exists(gt_name), "{} annot. file not found".format(gt_name)

        features = np.load(features_name)  # D x T
        file_ptr = open(gt_name, "r")
        content = file_ptr.read().split("\n")[:-1]
        vid_len = len(content)

        # Obs limit
        obs_percentage = self.list_of_examples[idx][1]
        if self.obs_perc == 0:
            assert self.mode == "train"
            if np.random.random() < 0.4:
                obs_percentage = 0.15 + 0.25 * np.random.random()
        else:
            assert obs_percentage in [0.2, 0.3]
        obs_lim = int(obs_percentage * len(content))

        # Pred limit
        pred_percentage = 1.0 - obs_percentage
        if self.part_obs:
            pred_percentage = 0.5
        pred_lim = int((obs_percentage + pred_percentage) * len(content))
        assert pred_lim <= len(content)

        """ ANNOTATIONS """
        # all labels
        content_past_future = self.label_to_id(content)
        content_past_future = content_past_future[:pred_lim]

        # goal label
        goal_label = self.goal_label_to_id(goals_label)

        # masks
        mask_past = np.zeros(len(content_past_future))
        mask_past[:obs_lim] = 1
        mask_past = mask_past[:: self.sample_rate]

        mask_future = np.zeros(len(content_past_future))
        mask_future[obs_lim:] = 1
        mask_future = mask_future[:: self.sample_rate]

        # classes (one hot)
        classes_one_hot = np.zeros((self.num_classes, len(content_past_future)))
        for i in range(len(content_past_future)):
            classes_one_hot[int(content_past_future[i])][i] = 1
        classes_one_hot = classes_one_hot[:, :: self.sample_rate]

        # goal (one hot)
        goal_label_extend = goal_label.repeat(len(content_past_future))
        goals_one_hot = np.zeros((int(self.num_highlevel_classes), len(content_past_future)))
        for i in range(len(content_past_future)):
            goals_one_hot[int(goal_label_extend[i])][i] = 1
        goals_one_hot = goals_one_hot[:, :: self.sample_rate]

        """ FEATURES """
        features_past = features[:, :obs_lim]
        features_past = features_past[:, ::self.sample_rate]

        # all classes (full)
        content_all = self.label_to_id(content)
        assert len(content_all) == vid_len

        # downsampled pred-window class ids
        content_past_future_ds = content_past_future[::self.sample_rate]

        # ---------- NEW: VIDEO FILES ----------
        # gt file name looks like: P03_cam01_P03_cereals.txt
        # frames folder: frames_root/P03/cam01/P03_cereals/
        base = os.path.basename(gt_name)  # e.g., P03_cam01_P03_cereals.txt
        parts = base.split("_")
        # robust parse
        participant = parts[0] if len(parts) >= 1 else "PXX"
        cam = parts[1] if len(parts) >= 2 else "cam01"
        stem = os.path.splitext(base)[0]
        if len(parts) >= 3:
            stem = "_".join(parts[2:]).split(".")[0]  # P03_cereals
        frames_dir = os.path.join(self.frames_root, participant, cam, stem)

        all_frames = self._sorted_frame_paths(frames_dir)

        # align frame count if needed (optional safety)
        # We assume 1 frame per GT line originally.
        if len(all_frames) < len(content):
            # if frames 부족하면 가능한 범위까지만 사용
            effective_len = len(all_frames)
        else:
            effective_len = len(content)

        # pred window frames: [0:pred_lim], then downsample by sample_rate
        # also ensure it matches classes_one_hot time length
        seq_paths = all_frames[:min(pred_lim, effective_len)]
        seq_paths = seq_paths[:: self.sample_rate]

        T = classes_one_hot.shape[1]
        if len(seq_paths) > T:
            seq_paths = seq_paths[:T]

        video_files = [self._load_resized_image(p, self.qwen_max_side) for p in seq_paths]

        sample = {
            "features": features_past,
            "classes": content_past_future_ds,
            "classes_all": content_all,
            "classes_one_hot": classes_one_hot,
            "mask_past": mask_past,
            "mask_future": mask_future,
            "vid_len": vid_len,
            "file_name": file_name,
            "goal": goal_label,
            "goal_one_hot": goals_one_hot,
            "video_files": video_files,  # <-- NEW
        }
        return sample

    def custom_collate(self, batch):
        """COLLECT"""
        batch_features = [item["features"] for item in batch]

        batch_classes = [item["classes"] for item in batch]
        batch_classes_all = [item["classes_all"] for item in batch]
        batch_classes_one_hot = [item["classes_one_hot"] for item in batch]

        batch_goal = [item["goal"] for item in batch]
        batch_goal_one_hot = [item["goal_one_hot"] for item in batch]

        batch_mask_past = [item["mask_past"] for item in batch]
        batch_mask_future = [item["mask_future"] for item in batch]

        # NEW
        batch_video_files = [item["video_files"] for item in batch]

        file_names = np.asarray([item["file_name"] for item in batch])
        batch_vid_len = [item["vid_len"] for item in batch]

        """ PAD """
        bz = len(batch_features)
        length_of_seq = list(map(len, batch_classes))
        length_of_seq_all = list(map(len, batch_classes_all))

        features_tensor = torch.zeros(
            bz, np.shape(batch_features[0])[0], max(length_of_seq), dtype=torch.float
        )

        goals_tensor = torch.ones(bz, 1, dtype=torch.long) * (self.num_highlevel_classes + 1)
        classes_tensor = torch.ones(bz, max(length_of_seq), dtype=torch.long) * (self.num_classes + 1)
        classes_tensor_all = torch.ones(bz, max(length_of_seq_all), dtype=torch.long) * (self.num_classes + 1)
        classes_tensor_one_hot = torch.zeros(bz, self.num_classes, max(length_of_seq), dtype=torch.float)
        goals_tensor_one_hot = torch.zeros(bz, self.num_highlevel_classes, max(length_of_seq), dtype=torch.float)

        mask_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)
        mask_past_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)
        mask_future_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)

        for i in range(bz):
            features_tensor[i, :, : np.shape(batch_features[i])[1]] = torch.from_numpy(batch_features[i])

            classes_tensor[i, : np.shape(batch_classes[i])[0]] = torch.from_numpy(batch_classes[i])
            classes_tensor_all[i, : np.shape(batch_classes_all[i])[0]] = torch.from_numpy(batch_classes_all[i])
            classes_tensor_one_hot[i, :, : np.shape(batch_classes_one_hot[i])[1]] = torch.from_numpy(batch_classes_one_hot[i])

            mask_tensor[i, 0, : np.shape(batch_classes[i])[0]] = torch.ones(np.shape(batch_classes[i])[0])
            mask_past_tensor[i, 0, : np.shape(batch_mask_past[i])[0]] = torch.from_numpy(batch_mask_past[i])
            mask_future_tensor[i, 0, : np.shape(batch_mask_future[i])[0]] = torch.from_numpy(batch_mask_future[i])

            goals_tensor[i, : np.shape(batch_goal[i])[0]] = torch.from_numpy(batch_goal[i])
            goals_tensor_one_hot[i, :, : np.shape(batch_goal_one_hot[i])[1]] = torch.from_numpy(batch_goal_one_hot[i])

        # SORT BY LENGTH and PERMUTE
        vid_lengths = torch.tensor(batch_vid_len)
        _, perm_idx = torch.sort(torch.tensor(length_of_seq), 0, descending=True)
        vid_lengths = vid_lengths[perm_idx]

        features_tensor = features_tensor[perm_idx]
        classes_tensor = classes_tensor[perm_idx]
        classes_tensor_all = classes_tensor_all[perm_idx]
        classes_tensor_one_hot = classes_tensor_one_hot[perm_idx]
        goals_tensor = goals_tensor[perm_idx]
        goals_tensor_one_hot = goals_tensor_one_hot[perm_idx]
        mask_tensor = mask_tensor[perm_idx]
        mask_past_tensor = mask_past_tensor[perm_idx]
        mask_future_tensor = mask_future_tensor[perm_idx]

        file_names = file_names[perm_idx.tolist()]
        meta_dict = {"file_names": file_names}

        # NEW: reorder video_files to match perm_idx
        video_files = [batch_video_files[i] for i in perm_idx.tolist()]

        return (
            features_tensor,            # (B, D, Tmax)
            classes_tensor,             # (B, Tmax)
            classes_tensor_all,         # (B, Tmax_all)
            classes_tensor_one_hot,     # (B, C, Tmax)
            mask_tensor,                # (B, 1, Tmax)
            mask_past_tensor,           # (B, 1, Tmax)
            mask_future_tensor,         # (B, 1, Tmax)
            vid_lengths,                # (B,)
            meta_dict,
            goals_tensor,               # (B, 1)
            goals_tensor_one_hot,       # (B, L1C, Tmax)
            video_files,                # list[list[PIL.Image]]
        )