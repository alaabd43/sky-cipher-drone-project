import airsim
import cv2
import numpy as np
from ultralytics import YOLO

# 1. تحميل نموذج YOLOv8
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

# 2. الاتصال بمحاكي AirSim
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 3. إقلاع الدرون
print("Taking off...")
client.takeoffAsync().join()
client.moveToZAsync(-10, 5).join() # الارتفاع 10 أمتار
print("Drone is airborne. Initiating Patrol and Attack mode...")

def is_red_color(img, box):
    """دالة لقص صورة السيارة من الإطار وفحص هل لونها أحمر"""
    x1, y1, x2, y2 = map(int, box)
    car_roi = img[y1:y2, x1:x2]
    
    if car_roi.size == 0:
        return False
        
    hsv = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
    # اللون الأحمر في HSV يقع في نطاقين
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    red_ratio = cv2.countNonZero(mask) / (car_roi.shape[0] * car_roi.shape[1])
    return red_ratio > 0.15 # إذا كان أكثر من 15% أحمر، نعتبرها سيارة حمراء

def get_camera_image_and_depth():
    """دالة مخصصة للاتصال بالمحاكي وجلب الصورة العادية وصورة العمق معاً"""
    req_scene = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    req_depth = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
    responses = client.simGetImages([req_scene, req_depth])
    
    # صورة BGR للرؤية
    response_scene = responses[0]
    img1d = np.frombuffer(response_scene.image_data_uint8, dtype=np.uint8)
    img_bgr = img1d.reshape(response_scene.height, response_scene.width, 3)
    
    # صورة العمق (Depth) لتجنب الأشجار
    response_depth = responses[1]
    depth_img = np.array(response_depth.image_data_float, dtype=np.float32)
    depth_img = depth_img.reshape(response_depth.height, response_depth.width)
    
    return img_bgr, response_scene.width, depth_img

def check_obstacle(depth_img, threshold=4.0):
    """فحص إذا كان هناك عائق (شجرة مثلاً) في المنتصف قريباً جداً"""
    h, w = depth_img.shape
    # أخذ عينة من وسط الشاشة (المسار الأمامي)
    center_region = depth_img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    # تجاهل الأصفار (البعيدة جدا أو المعطوبة)
    valid_depths = center_region[center_region > 0]
    if len(valid_depths) == 0:
        return False
    min_dist = np.min(valid_depths)
    return min_dist < threshold

def move_drone_sync(vx, vy, vz, duration, yaw_rate):
    """تحريك الدرون باستخدام أوامر متزامنة لمنع انهيار المحاكي (IOLoop Error)"""
    yaw_mode = airsim.YawMode(True, yaw_rate)
    client.client.call('moveByVelocityBodyFrame', float(vx), float(vy), float(vz), float(duration), 
                       airsim.DrivetrainType.ForwardOnly, yaw_mode, "")

# 4. تجهيز نافذة العرض لتكون كبيرة وقابلة للتعديل
cv2.namedWindow("Drone FPV View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drone FPV View", 800, 600)

# متغيرات للتحكم السلس لتجنب الاهتزازات
current_yaw_rate = 0.0
current_forward_speed = 3.0
mission_completed = False

# 5. حلقة المراقبة والتتبع
try:
    while not mission_completed:
        # التقاط الصورة العادية والعمق
        img_bgr, img_width, depth_img = get_camera_image_and_depth()
        target_found = False
        
        # 1. التحقق من وجود أشجار أو عوائق أمام الدرون أولاً
        is_obstacle = check_obstacle(depth_img, threshold=4.0) # مسافة 4 أمتار كتحذير لتجنب الشجر
        
        # تمرير الصورة لنموذج YOLO
        results = model(img_bgr, verbose=False)
        annotated_frame = results[0].plot()
        
        if is_obstacle:
            print("Obstacle (Tree) detected! Evading...")
            # إيقاف التقدم والالتفاف بسرعة لتجنب الشجرة
            current_forward_speed = 0.0
            current_yaw_rate = 40.0 # الالتفاف بقوة لليمين للهروب
            move_drone_sync(0, 0, 0, 0.2, current_yaw_rate)
            
            # إظهار رسالة تحذير على الشاشة
            cv2.putText(annotated_frame, "OBSTACLE EVASION!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # 2. إذا لم يكن هناك عائق، نبحث عن سيارة حمراء
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    
                    # رقم 2 يمثل "سيارة" (Car) في نموذج YOLO
                    if class_id == 2: 
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # التأكد من أن السيارة حمراء
                        if is_red_color(img_bgr, bbox):
                            target_found = True
                            print("Target Locked: RED CAR! Engaging...")
                            
                            x1, y1, x2, y2 = map(int, bbox)
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            img_height = img_bgr.shape[0]
                            
                            # حساب الانحراف عن مركز الصورة الأفقية
                            img_center_x = img_width / 2
                            error_x = center_x - img_center_x
                            
                            # التحكم السلس للالتفاف (Smooth Yaw PID)
                            target_yaw_rate = float(error_x * 0.1)
                            # دمج السرعة السابقة مع الحالية لتقليل الاهتزاز (Smoothing)
                            current_yaw_rate = (0.7 * current_yaw_rate) + (0.3 * target_yaw_rate)
                            
                            # 3. الهبوط إذا كانت السيارة أسفل الشاشة تماماً
                            if center_y > img_height * 0.75:
                                cv2.putText(annotated_frame, "LANDING ON RED CAR...", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                cv2.imshow("Drone FPV View", annotated_frame)
                                cv2.waitKey(1)
                                
                                print("Hovering successful. Commencing Landing sequence...")
                                # إيقاف الحركة الأفقية تماما ثم الهبوط
                                client.moveByVelocityAsync(0, 0, 0, 1).join()
                                client.landAsync().join()
                                mission_completed = True
                            else:
                                current_forward_speed = 8.0 # سرعة الهجوم
                            
                            break # ركز على أول سيارة حمراء تلاقيها
                
                if target_found:
                    break 
                    
            if not mission_completed:
                if target_found:
                    # توجيه الدرون نحو الهدف
                    cv2.putText(annotated_frame, "TARGET LOCKED: RED CAR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    move_drone_sync(current_forward_speed, 0, 0, 0.2, current_yaw_rate)
                else:
                     # 4. وضع الدورية: الطيران للأمام ببطء للبحث عن سيارة حمراء
                     current_forward_speed = 4.0
                     current_yaw_rate = 0.0
                     move_drone_sync(current_forward_speed, 0, 0, 0.2, 0)
             
        if not mission_completed:
            # عرض الكاميرا المباشرة للمراقبة
            cv2.imshow("Drone FPV View", annotated_frame)
            
            # اضغط 'q' لإيقاف الكود
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"Error occurred: {e}")
    
except KeyboardInterrupt:
    print("Mission Aborted by user.")
    
finally:
    if not mission_completed:
        # هبوط وإيقاف المحركات بأمان فقط إذا لم يكمل المهمة بعد
        client.moveByVelocityAsync(0, 0, 0, 1).join()
    cv2.destroyAllWindows()
    print("Drone stopped safely.")
