"""
手势识别测试脚本
使用本地图片测试 HAND_KEYPOINT_CLS 类
"""
import cv2
from walnutpi_kpu.HAND_KEYPOINT_CLS import HAND_KEYPOINT_CLS

# 可修改为你的测试图片路径
IMG_PATH = "./test.jpg"
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.5

# 读取图片
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"无法读取图片: {IMG_PATH}")
    exit(1)

print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

# 初始化手势识别器
hkc = HAND_KEYPOINT_CLS()

# 执行检测
results = hkc.run(img, reliability_threshold=CONFIDENCE_THRESHOLD,
                   nms_threshold=NMS_THRESHOLD)

print(f"\n检测到 {len(results)} 个手掌")
for result in results:
    print(f"  {result}")
    print(f"    手势: {result.gesture}")

# 绘制结果
img_draw = img.copy()
GESTURE_COLORS = {
    "fist": (0, 0, 255), "five": (0, 255, 0), "gun": (255, 0, 0),
    "love": (255, 0, 255), "one": (255, 255, 0), "six": (0, 255, 255),
    "three": (128, 128, 0), "thumbUp": (128, 0, 128), "yeah": (0, 128, 128),
}

for result in results:
    HAND_KEYPOINT_CLS.draw_keypoints(img_draw, result)
    color = GESTURE_COLORS.get(result.gesture, (255, 255, 255))
    cv2.putText(img_draw, result.gesture, (result.x, result.y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

# 保存结果
OUTPUT_PATH = "./.result.jpg"
cv2.imwrite(OUTPUT_PATH, img_draw)
print(f"\n结果已保存至: {OUTPUT_PATH}")
