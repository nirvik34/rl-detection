from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from env import COCOPatchEnv

IMG_DIR = "data/val2017"
ANN_FILE = "data/annotations/instances_val2017.json"

# Create environments
env = COCOPatchEnv(IMG_DIR, ANN_FILE)
eval_env = COCOPatchEnv(IMG_DIR, ANN_FILE)

# Validate environment
check_env(env)
print("Environment OK")

# Train DQN agent
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=500,
    batch_size=64,
    exploration_fraction=0.3,
    tensorboard_log="./tensorboard/"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=1000,
    n_eval_episodes=20,
    deterministic=True
)

print("Training started...")
model.learn(total_timesteps=50000, callback=eval_callback)
model.save("models/rl_detector_final")
print("Training complete. Model saved.")