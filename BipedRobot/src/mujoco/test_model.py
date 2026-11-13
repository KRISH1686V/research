import argparse
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import Config
from torch_rl_algorithms.algorithms.mpo.model import MPO
from torch_rl_algorithms.algorithms.d4pg.model import D4PG
from torch_rl_algorithms.algorithms.sac.model import SAC
from torch_rl_algorithms.algorithms.ddpg.model import DDPG
from envs.mujoco.mujoco_env import MujocoEnv


_ALGORITHM_MAP = {
    "sac": SAC,
    "ddpg": DDPG,
    "d4pg": D4PG,
    "mpo": MPO,
}


def test_model(config, checkpoint_path, episodes=10, episode_length=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = MujocoEnv(
        render_mode="human", 
        sim_frequency=config.train.sim_frequency,
        history_size=config.train.history_size,
    )

    algorithm = config.train.model.lower()
    ModelClass = _ALGORITHM_MAP.get(algorithm)
    if ModelClass is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(_ALGORITHM_MAP)}")

    model = ModelClass(env, model_path=checkpoint_path, device=device, config=config)

    mean_reward = 0.0
    for episode in range(episodes):
        obs, _ = env.reset()
        env.set_command(0.3, 0.0, 0.0)  # Walk straight forward at 0.3 m/s
        for step in range(episode_length):
            if isinstance(obs, dict):
                obs = {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0) if len(v.shape) == 1 else torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
            else:
                obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            action = model.step(obs)
            # Flatten action back if necessary
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).detach().cpu().numpy()
            
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            mean_reward += reward
            if done or truncated:
                break

    print(f"Mean reward over {episodes} episodes: {mean_reward / episodes:.4f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/final/train_config_sac.yaml",
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_final/sac/step_30000000.pt",
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=1000)
    args = parser.parse_args()
    config = Config(args.config)
    test_model(config, args.checkpoint, args.episodes, args.episode_length)
