import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import csv
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO

from train import DroneAI_Env, CustomCombinedExtractor

# ─────────────────────────────────────────
# إعدادات جمع البيانات
# ─────────────────────────────────────────
IMAGES_DIR = "dataset_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# ✅ 10,000 صورة موزعة بالتوازن
TARGET_FRAMES = {
    "Normal":      3000,
    "Evading":     3000,
    "Approaching": 3000,
    "Crash":       1000
}

# ✅ هدف صفوف الـ CSV = مليون صف (كل خطوة = صف)
TARGET_CSV_ROWS = 1_000_000

CLASS_TO_INT = {
    "Normal":      0,
    "Evading":     1,
    "Approaching": 2,
    "Crash":       3
}

collected_frames = {k: 0 for k in TARGET_FRAMES}
MAX_STEPS_PER_EP = 500

# ─────────────────────────────────────────
# الكاميرا مع Augmentation متكيف
# ─────────────────────────────────────────
def get_training_camera_image(env_wrapper, dist_to_target=None, augment=True):
    inner = env_wrapper.env
    try:
        img_float = inner.get_camera_obs()
        if augment:
            if dist_to_target is not None and dist_to_target < 4.0:
                brightness = random.uniform(0.90, 1.10)
                contrast   = random.uniform(0.95, 1.05)
            else:
                brightness = random.uniform(0.75, 1.25)
                contrast   = random.uniform(0.85, 1.15)
            img_aug = img_float * brightness
            mean    = img_aug.mean()
            img_aug = (img_aug - mean) * contrast + mean
            return np.clip(img_aug, 0.0, 1.0)
        return img_float
    except Exception as e:
        print(f"⚠ خطأ في الكاميرا: {e}")
        return np.zeros((64, 64, 3), dtype=np.float32)

# ─────────────────────────────────────────
# دالة استخراج كل بيانات الـ obs
# ─────────────────────────────────────────
def extract_features(obs, action, reward, step, ep_step_count):
    s = obs["sensors"]

    drone_x, drone_y, drone_z = float(s[0]), float(s[1]), float(s[2])
    rel_x,   rel_y,   rel_z   = float(s[3]), float(s[4]), float(s[5])
    target_dist = float(s[6])

    radar = [float(s[7 + i]) for i in range(8)]
    closest_obstacle = min((r for r in radar if r > 0), default=1.0)
    is_detected = int(s[15] > 0.5)

    wind_x   = float(s[16])
    wind_y   = float(s[17])
    wind_str = float(s[18])

    bearing_to_target = float(np.degrees(np.arctan2(rel_y, rel_x)))

    ax = round(float(action[0]), 4)
    ay = round(float(action[1]), 4)
    az = round(float(action[2]), 4)
    action_magnitude = round(float(np.linalg.norm(action)), 4)

    return {
        "step":             step,
        "ep_step_count":    ep_step_count,
        "drone_x":          round(drone_x,  4),
        "drone_y":          round(drone_y,  4),
        "drone_z":          round(drone_z,  4),
        "rel_x":            round(rel_x,    4),
        "rel_y":            round(rel_y,    4),
        "rel_z":            round(rel_z,    4),
        "target_dist":      round(target_dist, 4),
        "bearing_deg":      round(bearing_to_target, 2),
        "radar_detected":   is_detected,
        "radar_0":          round(radar[0], 4),
        "radar_1":          round(radar[1], 4),
        "radar_2":          round(radar[2], 4),
        "radar_3":          round(radar[3], 4),
        "radar_4":          round(radar[4], 4),
        "radar_5":          round(radar[5], 4),
        "radar_6":          round(radar[6], 4),
        "radar_7":          round(radar[7], 4),
        "closest_obstacle": round(closest_obstacle, 4),
        "wind_x":           round(wind_x,   4),
        "wind_y":           round(wind_y,   4),
        "wind_strength":    round(wind_str, 4),
        "action_x":         ax,
        "action_y":         ay,
        "action_z":         az,
        "action_magnitude": action_magnitude,
        "reward":           round(float(reward), 4),
    }

# ─────────────────────────────────────────
# أعمدة CSV
# ─────────────────────────────────────────
COLUMNS = [
    # row_id  = رقم صف فريد (يصل لمليون)
    # frame_id = رقم الصورة (يصل لـ 10,000)، فارغ إذا لم تُلتقط صورة في هذه الخطوة
    "row_id", "frame_id", "episode", "step", "ep_step_count",
    "frame_class", "frame_class_int",
    "image_path",
    "drone_x", "drone_y", "drone_z",
    "rel_x", "rel_y", "rel_z", "target_dist", "bearing_deg",
    "radar_detected",
    "radar_0", "radar_1", "radar_2", "radar_3",
    "radar_4", "radar_5", "radar_6", "radar_7",
    "closest_obstacle",
    "wind_x", "wind_y", "wind_strength",
    "action_x", "action_y", "action_z", "action_magnitude",
    "reward",
    "episode_outcome",
]

# ─────────────────────────────────────────
# التهيئة والتحميل
# ─────────────────────────────────────────
print("🚀 تهيئة بيئة SkyCipher لجمع البيانات...")
env = DroneAI_Env(render=False, level=3)

model_name    = "vision_smart_drone_level_3"
policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)
model         = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, device="cpu")

print(f"جاري حقن العقل المدبر من ({model_name}.zip)...")
try:
    model.set_parameters(model_name + ".zip")
    print("✅ تم تحميل أوزان الموديل بنجاح!")
except Exception as e:
    print(f"❌ فشل تحميل الموديل: {e}")
    env.close()
    exit()

# ─────────────────────────────────────────
# إعداد ملف CSV
# ─────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path  = f"skycipher_dataset_{timestamp}.csv"

total_images_target = sum(TARGET_FRAMES.values())
print(f"\n🎯 الأهداف:")
print(f"   صفوف CSV  : {TARGET_CSV_ROWS:,} صف  (كل خطوة = صف واحد)")
print(f"   صور       : {total_images_target:,} صورة موزعة:")
for k, v in TARGET_FRAMES.items():
    print(f"      - {k}: {v:,}")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()

    ep               = 0
    row_id_counter   = 0    # عداد الصفوف → يصل لمليون
    frame_id_counter = 0    # عداد الصور  → يصل لـ 10,000

    images_done = lambda: all(collected_frames[k] >= TARGET_FRAMES[k] for k in TARGET_FRAMES)

    # ✅ نستمر حتى يكتمل المليون صف وتكتمل الصور
    while row_id_counter < TARGET_CSV_ROWS or not images_done():
        obs, _ = env.reset()
        ep     += 1
        ep_step = 0

        recent_history = []
        ep_rows        = []  # نجمع صفوف الجولة لنكتب episode_outcome بعد نهايتها

        for step in range(1, MAX_STEPS_PER_EP + 1):
            action, _ = model.predict(obs, deterministic=True)
            new_obs, reward, done, truncated, info = env.step(action)
            ep_step += 1

            is_detected = bool(obs["sensors"][15] > 0.5)
            target_dist = float(obs["sensors"][6])

            current_class = "Normal"
            if is_detected:
                current_class = "Evading"
            elif target_dist < 4.5:
                current_class = "Approaching"

            features = extract_features(obs, action, reward, step, ep_step)

            # ─── منطق حفظ الصور (10,000 صورة موزعة) ───
            img_path_to_save  = ""
            frame_id_to_write = ""

            save_image = False
            if not images_done() and collected_frames[current_class] < TARGET_FRAMES[current_class]:
                if current_class == "Normal":
                    if step % 5 == 0:
                        save_image = True
                else:
                    save_image = True

            if save_image:
                img = get_training_camera_image(env, dist_to_target=target_dist, augment=True)
                frame_entry = {
                    "frame_class": current_class,
                    "img":         img,
                    "features":    features,
                    "step":        step,
                }
                recent_history.append(frame_entry)

                if len(recent_history) > 5:
                    safe_entry = recent_history.pop(0)
                    safe_class = safe_entry["frame_class"]

                    if collected_frames[safe_class] < TARGET_FRAMES[safe_class]:
                        frame_id_counter += 1
                        img_filename      = f"frame_{frame_id_counter:05d}_{safe_class}.png"
                        saved_img_path    = os.path.join(IMAGES_DIR, img_filename)
                        plt.imsave(saved_img_path, safe_entry["img"])
                        collected_frames[safe_class] += 1
                        img_path_to_save  = saved_img_path
                        frame_id_to_write = frame_id_counter

            # ─── تسجيل كل خطوة في الـ CSV (= مليون صف) ───
            row_id_counter += 1
            row = {
                "row_id":           row_id_counter,
                "frame_id":         frame_id_to_write,
                "episode":          ep,
                "frame_class":      current_class,
                "frame_class_int":  CLASS_TO_INT[current_class],
                "image_path":       img_path_to_save,
                "episode_outcome":  "",   # يُكمَل بعد نهاية الجولة
                **features,
            }
            ep_rows.append(row)
            obs = new_obs

            if done or truncated:
                status = info.get("status", "unknown")

                # ✅ Crash: نصنف آخر 5 إطارات كـ Crash
                if status == "crash":
                    for crash_entry in recent_history:
                        if collected_frames["Crash"] < TARGET_FRAMES["Crash"]:
                            frame_id_counter += 1
                            img_filename      = f"frame_{frame_id_counter:05d}_Crash.png"
                            crash_img_path    = os.path.join(IMAGES_DIR, img_filename)
                            plt.imsave(crash_img_path, crash_entry["img"])
                            collected_frames["Crash"] += 1
                            # نربط الصورة بالصف الصحيح
                            for r in reversed(ep_rows):
                                if r["step"] == crash_entry["step"] and r["frame_id"] == "":
                                    r["frame_id"]        = frame_id_counter
                                    r["image_path"]      = crash_img_path
                                    r["frame_class"]     = "Crash"
                                    r["frame_class_int"] = CLASS_TO_INT["Crash"]
                                    break

                # ✅ نكتب episode_outcome الصحيح على كل صفوف الجولة دفعة واحدة
                for r in ep_rows:
                    r["episode_outcome"] = status
                    writer.writerow(r)

                break

        # تقرير كل 10 جولات
        if ep % 10 == 0:
            img_pct = (sum(collected_frames.values()) / total_images_target) * 100
            row_pct = (row_id_counter / TARGET_CSV_ROWS) * 100
            print(
                f"جولة {ep:4d} | "
                f"صفوف: {row_id_counter:>8,}/{TARGET_CSV_ROWS:,} ({row_pct:5.1f}%) | "
                f"صور: {sum(collected_frames.values()):>5,}/{total_images_target:,} ({img_pct:5.1f}%) "
                f"[N:{collected_frames['Normal']} E:{collected_frames['Evading']} "
                f"A:{collected_frames['Approaching']} C:{collected_frames['Crash']}]"
            )
            f.flush()

env.close()

print(f"\n{'='*65}")
print(f"✅ اكتمل جمع البيانات!")
print(f"   ملف CSV         : {csv_path}")
print(f"   إجمالي الصفوف   : {row_id_counter:,}")
print(f"   إجمالي الصور    : {frame_id_counter:,}")
print(f"   مجلد الصور      : {IMAGES_DIR}/")
print(f"{'='*65}")