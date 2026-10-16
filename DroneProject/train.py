import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch as th
import torch.nn as th_nn
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pybullet as p
from environment123456 import DroneEnvironment


# ══════════════════════════════════════════════
# الشبكة العصبية — CNN + MLP المتوافقة مع الكاميرا (3 قنوات)
# ══════════════════════════════════════════════
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=320)
        n_input_channels = 3

        self.cnn = th_nn.Sequential(
            th_nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            th_nn.BatchNorm2d(32), th_nn.ReLU(), th_nn.MaxPool2d(2),
            th_nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            th_nn.BatchNorm2d(64), th_nn.ReLU(), th_nn.MaxPool2d(2),
            th_nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            th_nn.BatchNorm2d(64), th_nn.ReLU(), th_nn.MaxPool2d(2),
            th_nn.Flatten(),
        )
        self.cnn_linear = th_nn.Sequential(
            th_nn.Linear(64 * 8 * 8, 256),
            th_nn.ReLU(),
            th_nn.Dropout(0.2),
        )

        n_sensors = observation_space.spaces["sensors"].shape[0]
        self.sensor_net = th_nn.Sequential(
            th_nn.Linear(n_sensors, 128), th_nn.ReLU(),
            th_nn.Linear(128, 64),        th_nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        image = observations["image"]
        if image.dim() == 4 and image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 3, 1, 2).contiguous()
        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        cnn_features    = self.cnn_linear(self.cnn(image))
        sensor_features = self.sensor_net(observations["sensors"])
        return th.cat((cnn_features, sensor_features), dim=1)


class DroneAI_Env(gym.Env):
    def __init__(self, render=False, level=1):
        super().__init__()
        self.env = DroneEnvironment(render=render, level=level)
        self.level = level
        self.episode_count  = 0
        self.success_count  = 0
        self.fail_count     = 0
        self.current_step   = 0
        self.steps_detected = 0

        self.REWARD_SUCCESS   = +50.0    
        self.REWARD_CRASH     = -15.0    
        self.REWARD_TIMEOUT   = -10.0    
        self.REWARD_TIME_PEN  = -0.1     
        self.REWARD_RADAR_PEN = -1.0     
        self.REWARD_APPROACH  = 15.0     

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image":   spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32),
            "sensors": spaces.Box(low=-100, high=100, shape=(19,), dtype=np.float32)
        })

    def set_level(self, new_level: int):
        self.level     = new_level
        self.env.level = new_level

    def get_observation(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.env.drone_id, physicsClientId=self.env.client)
        target_pos, _ = p.getBasePositionAndOrientation(self.env.target_id, physicsClientId=self.env.client)

        rel  = np.array(target_pos) - np.array(drone_pos)
        dist = float(np.linalg.norm(rel))
        radar_dists = self.env.get_radar_obs()
        is_det  = 1.0 if self.env.drone_detected_by_radar else 0.0
        wind_obs = self.env.get_wind_obs() if self.level >= 3 else [0.0, 0.0, 0.0]

        sensors = np.array([
            drone_pos[0], drone_pos[1], drone_pos[2],
            rel[0], rel[1], rel[2],
            dist,
            *radar_dists,   
            is_det,         
            *wind_obs       
        ], dtype=np.float32)

        camera_image = self.env.get_camera_obs()
        return {"image": camera_image, "sensors": sensors}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step   = 0
        self.steps_detected = 0
        self.env.reset()

        target_pos, _ = p.getBasePositionAndOrientation(self.env.target_id, physicsClientId=self.env.client)
        while True:
            start_x = random.uniform(-14.0, 14.0)
            start_y = random.uniform(-14.0, 14.0)
            start_z = random.uniform(2.0, 6.0)
            dist_to_target = np.linalg.norm(np.array([start_x, start_y]) - np.array([target_pos[0], target_pos[1]]))
            if dist_to_target > 8.0 and not (-3 < start_x < 3 and -3 < start_y < 3):
                break

        p.resetBasePositionAndOrientation(
            self.env.drone_id,
            [start_x, start_y, start_z],
            [0, 0, 0, 1],
            physicsClientId=self.env.client
        )
        return self.get_observation(), {}

    def step(self, action):
        self.current_step += 1
        old_obs  = self.get_observation()
        old_dist = float(old_obs["sensors"][6])
        is_detected = old_obs["sensors"][15] > 0.5

        if is_detected:
            self.steps_detected += 1
            action_to_env = action * 3.5
        else:
            self.steps_detected = max(0, self.steps_detected - 2)
            action_to_env = action * 2.5

        crashed = self.env.step(action_to_env)
        obs  = self.get_observation()
        dist = float(obs["sensors"][6])
        radar_obs = obs["sensors"][7:15]

        if self.current_step >= 500:
            self.episode_count += 1
            return obs, self.REWARD_TIMEOUT, False, True, {"status": "timeout"}

        if crashed:
            self.fail_count    += 1
            self.episode_count += 1
            return obs, self.REWARD_CRASH, True, False, {"status": "crash"}

        elif dist < 3.5:  
            self.success_count += 1
            self.episode_count += 1
            return obs, self.REWARD_SUCCESS, True, False, {"status": "success"}

        else:
            progress = old_dist - dist
            approach_reward = progress * self.REWARD_APPROACH

            building_dists = [r for r in radar_obs if r > 0]
            closest_obstacle = min(building_dists) if building_dists else 1.0
            obstacle_penalty = 0.0
            
            if closest_obstacle < 0.2:
                obstacle_penalty = -3.0 * (0.2 - closest_obstacle)

            if is_detected:
                reward = approach_reward + self.REWARD_TIME_PEN + self.REWARD_RADAR_PEN + obstacle_penalty
            else:
                reward = approach_reward + self.REWARD_TIME_PEN + obstacle_penalty

            reward = float(np.clip(reward, -15.0, 15.0))
            return obs, reward, False, False, {"status": "flying"}

    def close(self):
        self.env.close()

# ══════════════════════════════════════════════
# نظام الترفيع المطور (التعديلات المطلوبة)
# ══════════════════════════════════════════════
class CurriculumCallback(BaseCallback):
    WINDOW = 50

    def __init__(self, env: DroneAI_Env, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.episode_results = []
        self._prev_ep = 0

    def _on_step(self) -> bool:
        ep_now = self.env.episode_count
        if ep_now > self._prev_ep:
            self._prev_ep = ep_now
            for info in self.locals.get("infos", []):
                status = info.get("status", "")
                if status in ("success", "crash", "timeout"):
                    self.episode_results.append(1 if status == "success" else 0)

            if len(self.episode_results) > self.WINDOW:
                self.episode_results = self.episode_results[-self.WINDOW:]

            if len(self.episode_results) >= self.WINDOW:
                success_rate = np.mean(self.episode_results)
                
                # 💡 عتبات النجاح الذكية بناءً على طلبك
                if self.env.level == 1:
                    target_threshold = 0.70
                elif self.env.level == 2:
                    target_threshold = 0.65
                else:
                    target_threshold = 0.78

                if self.verbose:
                    print(f"\n📊 [Curriculum] Level {self.env.level} | نجاح آخر {self.WINDOW} جولة: {success_rate*100:.1f}% (الهدف: {target_threshold*100:.0f}%)")

                if self.env.level < 3 and success_rate >= target_threshold:
                    new_level = self.env.level + 1
                    print(f"\n🚀 ترقية! Level {self.env.level} → {new_level}")
                    self.model.save(f"drone_level{self.env.level}_done")
                    self.env.set_level(new_level)
                    self.episode_results.clear()

                elif self.env.level == 3 and success_rate >= target_threshold:
                    print(f"\n🏆 Level 3 مُتقن بنسبة {success_rate*100:.1f}%! (البيانات جاهزة)")
                    self.model.save("vision_smart_drone_level_3")
                    return False # 🛑 إيقاف التدريب فوراً!
        return True

if __name__ == "__main__":
    TOTAL_STEPS = 800_000
    env = DroneAI_Env(level=1, render=False)
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)

    print("📌 بناء موديل جديد نقي للتدريب النهائي الصارم — البداية من Level 1")
    model = PPO(
        "MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
        learning_rate=2e-4, ent_coef=0.02, clip_range=0.2, n_steps=2048,        
        batch_size=128, n_epochs=10, gamma=0.995, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95,
    )

    curriculum_cb = CurriculumCallback(env=env, verbose=1)
    print(f"\n🎓 بدء التدريب الذكي (800 ألف خطوة - ترقيات: 70% ⬅️ 65% ⬅️ 78%)")
    model.learn(total_timesteps=TOTAL_STEPS, callback=curriculum_cb)
    
    model.save("vision_smart_drone_level_3")
    env.close()
    print("\n✅ انتهى التدريب وتم حفظ الموديل النهائي (vision_smart_drone_level_3)!")