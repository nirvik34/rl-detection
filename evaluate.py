from stable_baselines3 import DQN
from env import COCOPatchEnv
import numpy as np

IMG_DIR = "data/val2017"
ANN_FILE = "data/annotations/instances_val2017.json"

env = COCOPatchEnv(IMG_DIR, ANN_FILE)
model = DQN.load("models/rl_detector_final")

n_episodes = 200
total_patches_used = []
total_objects_found = []
total_objects_present = []

for _ in range(n_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)

    patches_used = int(env.inspected.sum())
    obj_patches = sum(1 for i in range(env.n_patches) if env._patch_has_object(i))
    found = sum(1 for i in range(env.n_patches) if env.inspected[i] and env._patch_has_object(i))

    total_patches_used.append(patches_used)
    total_objects_present.append(obj_patches)
    total_objects_found.append(found)

avg_patches = np.mean(total_patches_used)
recall = np.mean([f/p if p>0 else 1 for f,p in zip(total_objects_found, total_objects_present)])

print("=" * 40)
print(f"Avg patches inspected : {avg_patches:.1f} / 16")
print(f"Patch efficiency      : {(1 - avg_patches/16)*100:.1f}% fewer regions")
print(f"Object recall         : {recall*100:.1f}%")
print("=" * 40)