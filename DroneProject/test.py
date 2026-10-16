import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import numpy as np
import pybullet as p
import traceback
os.chdir("DroneProject")

from train import DroneAI_Env, CustomCombinedExtractor
from stable_baselines3 import PPO

print("=" * 60)
print("Testing Drone Vision - Level 3 (Safe Load Mode)")
print("=" * 60)

model_name = "vision_smart_drone_level_3.zip"

print("جاري تهيئة البيئة والشبكة العصبية...")
env = DroneAI_Env(render=True, level=3)

print("جاري بناء عقل فارغ بنفس المواصفات...")
policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, device="cpu")

print(f"جاري حقن الأوزان والخبرة من الملف ({model_name})...")
try:
    model.set_parameters(model_name)
    print("✅ تم حقن الموديل بنجاح! العقل جاهز للطيران.")
except Exception as e:
    print(f"\n❌ لم أتمكن من العثور على ملف {model_name}. تأكدي من إكمال التدريب أولاً.")
    exit()

obs, _ = env.reset()

print("✅ بدء الاختبار...")
print("اضغط Ctrl+C للإيقاف\n")

NUM_EPISODES = 50
results = []
ep_reward = 0.0

try:
    ep = 0
    while ep < NUM_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward

        dist = obs["sensors"][6]
        sleep_time = max(1/120, min(1/30, dist / 200))
        time.sleep(float(sleep_time))

        if done or truncated:
            ep += 1
            status = info.get("status", "unknown")
            results.append({"ep": ep, "status": status, "reward": ep_reward})

            icon = "🎯" if status == "success" else "💥" if status == "crash" else "⏱"
            print(f"جولة {ep:2d}/{NUM_EPISODES} | {icon} {status:8s} | "
                  f"مكافأة: {ep_reward:8.1f} | "
                  f"✅{sum(1 for r in results if r['status']=='success')} "
                  f"💥{sum(1 for r in results if r['status']=='crash')} "
                  f"⏱{sum(1 for r in results if r['status']=='timeout')}")

            ep_reward = 0.0
            time.sleep(0.5)
            obs, _ = env.reset()

except KeyboardInterrupt:
    print("\n⚠ تم الإيقاف المبكر من قبل المستخدم.")
except Exception:
    print("\n❌ حدث خطأ مفاجئ أثناء الطيران:")
    traceback.print_exc()

finally:
    env.close()

    if results:
        n = len(results)
        successes = sum(1 for r in results if r["status"] == "success")
        crashes   = sum(1 for r in results if r["status"] == "crash")
        timeouts  = sum(1 for r in results if r["status"] == "timeout")
        avg_r     = np.mean([r["reward"] for r in results])

        print(f"\n{'='*60}")
        print(f"📊 ملخص الاختبار — {n} جولة")
        print(f"{'='*60}")
        print(f"  🎯 نجاح   : {successes:3d}  ({successes/n*100:5.1f}%)")
        print(f"  💥 اصطدام : {crashes:3d}  ({crashes/n*100:5.1f}%)")
        print(f"  ⏱ timeout : {timeouts:3d}  ({timeouts/n*100:5.1f}%)")
        print(f"  متوسط المكافأة : {avg_r:.1f}")
        print(f"{'='*60}")