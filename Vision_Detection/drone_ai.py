import airsim
import cv2
import numpy as np
from ultralytics import YOLO

# 1. تحميل الموديل
model = YOLO('yolov8n.pt') 

# 2. الاتصال
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("جاري الإقلاع...")
client.takeoffAsync().join() 
client.moveToZAsync(-3, 1).join() 

try:
    while True:
        # التعديل: استدعاء الـ RPC مباشرة بدون المرور بدالة المكتبة اللي فيها المشكلة
        # هيك إحنا بنبعث الباراميترز "عالبلاطة" للمحاكي
        raw_image = client.client.call('simGetImage', "0", airsim.ImageType.Scene)
        
        if not raw_image:
            continue

        img1d = np.frombuffer(raw_image, dtype=np.uint8)
        img_rgb = cv2.imdecode(img1d, cv2.IMREAD_COLOR)

        if img_rgb is None:
            continue

        # تشغيل YOLO
        results = model(img_rgb, stream=True, conf=0.4)
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("Drone AI Eye - YOLOv8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"حدث خطأ: {e}")
finally:
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    cv2.destroyAllWindows()