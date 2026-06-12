import gymnasium as gym
import numpy as np
import cv2
import json
import os
from gymnasium import spaces

class COCOPatchEnv(gym.Env):
    """Environment for patch-based object detection.
    
    Agent selects patches from a grid. Reward is based on finding objects.
    """
    def __init__(self, img_dir, ann_file, grid_size=4, max_steps=8):
        super().__init__()
        self.img_dir = img_dir
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_patches = grid_size * grid_size  # 16 patches

        # Load COCO annotations
        with open(ann_file) as f:
            coco = json.load(f)
        self.images = coco['images']
        self.annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            self.annotations.setdefault(img_id, []).append(ann)

        # Action: pick one of 16 patches
        self.action_space = spaces.Discrete(self.n_patches)

        # State: for each patch — (inspected_flag + 3 color channels mean) = 4 values
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.n_patches * 4,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Pick random image
        img_info = self.images[np.random.randint(len(self.images))]
        self.img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        self.img = cv2.imread(img_path)
        self.img = cv2.resize(self.img, (640, 640))
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Ground truth boxes for this image
        self.gt_boxes = self.annotations.get(self.img_id, [])

        # Track which patches have been inspected
        self.inspected = np.zeros(self.n_patches, dtype=np.float32)
        self.steps = 0
        self.total_reward = 0

        return self._get_obs(), {}

    def _get_obs(self):
        patch_h = 640 // self.grid_size
        patch_w = 640 // self.grid_size
        obs = []
        for i in range(self.n_patches):
            row = i // self.grid_size
            col = i % self.grid_size
            patch = self.img_rgb[
                row*patch_h:(row+1)*patch_h,
                col*patch_w:(col+1)*patch_w
            ]
            mean_rgb = patch.mean(axis=(0,1)) / 255.0
            obs.extend([self.inspected[i]] + mean_rgb.tolist())
        return np.array(obs, dtype=np.float32)

    def _patch_has_object(self, patch_idx):
        """Check if any GT box overlaps with this patch"""
        row = patch_idx // self.grid_size
        col = patch_idx % self.grid_size
        patch_h = 640 // self.grid_size
        patch_w = 640 // self.grid_size

        px1, py1 = col * patch_w, row * patch_h
        px2, py2 = px1 + patch_w, py1 + patch_h

        for ann in self.gt_boxes:
            x, y, w, h = ann['bbox']
            # Scale to 640x640 (original sizes vary)
            if px1 < x+w and px2 > x and py1 < y+h and py2 > y:
                return True
        return False

    def step(self, action):
        self.steps += 1
        reward = -0.1  # step penalty (encourages efficiency)

        if self.inspected[action] == 0:  # new patch
            self.inspected[action] = 1
            if self._patch_has_object(action):
                reward += 1.0   # found an object!
            else:
                reward -= 0.2   # wasted a step

        done = self.steps >= self.max_steps
        truncated = False

        # Bonus reward at end if found most objects
        if done:
            n_obj_patches = sum(
                1 for i in range(self.n_patches)
                if self._patch_has_object(i)
            )
            if n_obj_patches > 0:
                found = sum(
                    1 for i in range(self.n_patches)
                    if self.inspected[i] and self._patch_has_object(i)
                )
                reward += 2.0 * (found / n_obj_patches)

        return self._get_obs(), reward, done, truncated, {}