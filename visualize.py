from stable_baselines3 import DQN
from env import COCOPatchEnv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

IMG_DIR = "data/val2017"
ANN_FILE = "data/annotations/instances_val2017.json"

env = COCOPatchEnv(IMG_DIR, ANN_FILE)
model = DQN.load("models/rl_detector_final")

obs, _ = env.reset()
done = False
step_order = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    step_order.append(action)
    obs, _, done, _, _ = env.step(action)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(env.img_rgb)

patch_h = 640 // env.grid_size
patch_w = 640 // env.grid_size

for i, patch_idx in enumerate(step_order):
    row = patch_idx // env.grid_size
    col = patch_idx % env.grid_size
    has_obj = env._patch_has_object(patch_idx)
    color = 'lime' if has_obj else 'red'
    rect = mpatches.Rectangle(
        (col*patch_w, row*patch_h), patch_w, patch_h,
        linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
    )
    ax.add_patch(rect)
    ax.text(col*patch_w+5, row*patch_h+20, str(i+1),
            color='white', fontsize=12, fontweight='bold')

ax.set_title("RL Agent Inspection Order\n(Green=Object Found, Red=Empty)", fontsize=13)
plt.tight_layout()
plt.savefig("results/attention_map.png", dpi=150)
plt.show()
print("✅ Saved to results/attention_map.png")