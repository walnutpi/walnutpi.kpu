import os
import numpy as np
import cv2
from typing import List
import time
import threading
import queue
from walnutpi_kpu import *
from typing import Literal


class YOLO_RESULT_DET:
    x: int
    y: int
    w: int
    h: int
    xywh: np.ndarray
    label: int  # 类别索引
    reliability: float  # 置信度
    index_in_all_boxes: int


class YOLO_RESULT_OBB(YOLO_RESULT_DET):
    angle: float  # 旋转角度

    def _rotate_point(self, cx, cy, x, y, angle):
        """旋转点(x, y)围绕中心点(cx, cy)旋转angle弧度"""
        s, c = np.sin(angle), np.cos(angle)
        x_new = c * (x - cx) - s * (y - cy) + cx
        y_new = s * (x - cx) + c * (y - cy) + cy
        return int(x_new), int(y_new)

    def get_top_left(self):
        """获取旋转后的左上角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x - half_w, self.y - half_h, self.angle
        )

    def get_bottom_left(self):
        """获取旋转后的左下角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x - half_w, self.y + half_h, self.angle
        )

    def get_top_right(self):
        """获取旋转后的右上角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x + half_w, self.y - half_h, self.angle
        )

    def get_bottom_right(self):
        """获取旋转后的右下角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x + half_w, self.y + half_h, self.angle
        )


class YOLO_RESULT_SEG(YOLO_RESULT_DET):
    contours: list  # 边界点的坐标，形式是 ((x1, y1)，....)
    mask: np.ndarray  # 一张单通道图片，被识别为物体的区域为255，背景为0
    _raw_mask: np.ndarray


class _YOLO_KEYPOINT:
    xy = (0, 0)
    visibility: float


class YOLO_RESULT_POSE(YOLO_RESULT_DET):
    keypoints: List[_YOLO_KEYPOINT] = []  # 各个关键点的坐标


class _YOLO_RESULT_CLS_INDEX:
    label: int  # 类别索引
    reliability: float  # 置信度

    def __init__(self, label=0, reliability=0):
        self.label = label
        self.reliability = reliability


class YOLO_RESULT_CLS:
    # TOP5包含了置信度排名前5的类别
    top5 = [
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
    ]
    # all是一个数组，包含所有类别的置信度，all[35]代表类别35的置信度，以此类推
    all = np.zeros(1)


class _YOLO_BASE:
    has_result = False
    is_running = False
    model_size: int
    nms_threshold = 0.45  # nms阈值
    results = []
    thread = None

    class _speed:
        ms_post_process: float = 0  # 后处理耗时
        ms_inference: float = 0  # 推理耗时

    speed = _speed()

    def __init__(
        self, kmodel_path: str, size: int, nncase_version: NNCASEVersionType = "2.10"
    ):
        """
        初始化
        @kmodel_path: 模型路径
        @size: 模型输入尺寸
        @nncase_version: nncase版本
        """
        self.model_h = size
        self.model_w = size

        self.nn = get_nncase(nncase_version)
        self.kpu = self.nn.Interpreter()
        self.ai2d = self.nn.AI2D()

        self.kpu.load_model(kmodel_path)
        # 创建一个临时输入张量用于绑定输入
        tmp_tensor = self.nn.RuntimeTensor.from_numpy(
            np.ones((1, 3, self.model_h, self.model_w), dtype=np.uint8)
        )

        self.kpu.set_input_tensor(0, tmp_tensor)
        # 创建任务队列和工作线程
        self._task_queue = queue.Queue()
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    ai2d_2d_w = -1
    ai2d_2d_h = -1

    def ai2d_init(self, model_w, model_h, img_w, img_h):
        """
        初始化AI2D
        @model_w: 模型输入尺寸,宽度
        @model_h: 模型输入尺寸,高度
        @img_w: 图片宽度
        @img_h: 图片高度
        """

        if self.ai2d_2d_w == model_w and self.ai2d_2d_h == model_h:
            return
        self.ai2d_2d_w, self.ai2d_2d_h = model_w, model_h

        # 计算输入图像缩放比例，保持纵横比
        self.ratio = min(model_w / img_w, model_h / img_h)
        new_w, new_h = int(img_w * self.ratio), int(img_h * self.ratio)
        dw, dh = (model_w - new_w), (model_h - new_h)
        pad_left, pad_right = 0, int(dw)
        pad_top, pad_bottom = 0, int(dh)

        # ===============================
        # 配置 AI2D 预处理流水线
        # ===============================
        self.ai2d.set_datatype(
            self.nn.AI2D_FORMAT.NCHW_FMT,  # 输入格式
            self.nn.AI2D_FORMAT.NCHW_FMT,  # 输出格式
            np.uint8,
            np.uint8,  # 输入输出数据类型
        )
        # 设置 resize 参数（使用 tf_bilinear 双线性插值）
        self.ai2d.set_resize_param(
            True,
            self.nn.AI2D_INTERP_METHOD.tf_bilinear,
            self.nn.AI2D_INTERP_MODE.half_pixel,
        )
        # 设置 padding 参数（补边）
        self.ai2d.set_pad_param(
            True,
            [0, 0, 0, 0, pad_top, pad_bottom, pad_left, pad_right],
            0,
            [114, 114, 114],
        )  # 用灰色填充
        # 构建 AI2D pipeline（输入、输出 shape）
        self.ai2d.build([1, 3, img_h, img_w], [1, 3, model_h, model_w])

    def post_process(self, reliability_threshold, nms_threshold):
        """后处理，子类需要重写这个函数"""
        return []

    def run(self, img, reliability_threshold=0.5, nms_threshold=0.5):
        """
        检测图片，阻塞直到检测完成，返回检测结果
        @img: 图片
        @reliability_threshold: 置信度阈值
        @nms_threshold: nms阈值
        """
        self.is_running = True
        self.has_result = False

        time_point = time.time() * 1000

        try:
            self.img_w, self.img_h = img.shape[1], img.shape[0]
            self.ai2d_init(self.model_w, self.model_h, self.img_w, self.img_h)

            self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_nchw = np.array([self.img_rgb.transpose((2, 0, 1))])  # 转换为 NCHW

            # -------------------------------
            # 执行 AI2D 预处理（resize + pad）
            # -------------------------------
            ai2d_input_tensor = self.nn.RuntimeTensor.from_numpy(img_nchw)
            kpu_input_tensor = self.kpu.get_input_tensor(0)
            self.ai2d.run(ai2d_input_tensor, kpu_input_tensor)

            # -------------------------------
            # 模型推理
            # -------------------------------
            self.kpu.run()

            self.speed.ms_inference = time.time() * 1000 - time_point
            time_point = time.time() * 1000

            self.results = self.post_process(reliability_threshold, nms_threshold)
            self.speed.ms_post_process = time.time() * 1000 - time_point
            time_point = time.time() * 1000
        except Exception as e:
            print(e)

        self.has_result = True
        self.is_running = False
        return self.results

    def run_async(self, img, reliability_threshold=0.5, nms_threshold=0.5):
        """
        检测图片，立即返回，不阻塞
        @img: 图片路径或图像数据
        @reliability_threshold: 置信度阈值
        @nms_threshold: NMS阈值
        """
        if not self.is_running:
            self.nms_threshold = nms_threshold
            self.is_running = True
            # 将任务放入队列
            self._task_queue.put((img, reliability_threshold, nms_threshold))
        else:
            print("模型正在运行中，请等待当前任务完成")

    def _worker_loop(self):
        """工作线程,检测信号随时启动异步任务"""
        while not self._shutdown_event.is_set():
            try:
                # 等待任务，超时后检查是否需要退出
                task_data = self._task_queue.get(timeout=0.1)
                if task_data is None:  # 停止信号
                    break

                img, reliability_threshold, nms_threshold = task_data
                self.thread_async_run(img, reliability_threshold, nms_threshold)
                self._task_queue.task_done()
            except queue.Empty:
                continue  # 超时继续检查退出信号
            except Exception:
                if not self._task_queue.empty():
                    self._task_queue.task_done()

    def thread_async_run(self, img, reliability_threshold, nms_threshold):
        """线程异步任务"""
        try:
            self.run(img, reliability_threshold, nms_threshold)
        except:
            pass

    def get_result(self):
        self.has_result = False
        return self.results

    def __del__(self):
        """析构函数，清理线程"""
        # 使用 getattr 安全地访问属性，如果属性不存在则返回 None
        shutdown_event = getattr(self, "_shutdown_event", None)
        worker_thread = getattr(self, "_worker_thread", None)

        if shutdown_event:
            shutdown_event.set()

        # 等待工作线程结束
        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=1.0)  # 设置超时避免无限等待


class YOLO11_DET(_YOLO_BASE):
    results: List[YOLO_RESULT_DET] = []
    _result_type = YOLO_RESULT_DET

    def get_result(self) -> List[YOLO_RESULT_DET]:
        return super().get_result()

    def run(
        self, img, reliability_threshold=0.5, nms_threshold=0.5
    ) -> List[YOLO_RESULT_DET]:
        return super().run(img, reliability_threshold, nms_threshold)

    def post_process(self, reliability_threshold, nms_threshold):
        # 获取模型输出
        model_output = self.kpu.get_output_tensor(0).to_numpy()
        predictions = model_output[0].transpose()  # (8400, 84)

        boxes = predictions[:, :4]  # [x_center, y_center, w, h]
        class_scores = predictions[:, 4:]  # 各类别置信度

        # 取每个候选框的最大类别得分
        scores = np.max(class_scores, axis=1)
        class_ids = class_scores.argmax(axis=1)

        # 过滤低置信度目标
        mask = scores > reliability_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 将归一化坐标还原到原图尺寸
        boxes_scaled = boxes.copy()
        boxes_scaled[:, 0] /= self.ratio  # x_center
        boxes_scaled[:, 1] /= self.ratio  # y_center
        boxes_scaled[:, 2] /= self.ratio  # width
        boxes_scaled[:, 3] /= self.ratio  # height

        scores = scores.astype(np.float32)

        # -------------------------------
        # 执行 NMS
        # -------------------------------
        boxes_xy = boxes_scaled[:, :2] - boxes_scaled[:, 2:4] / 2  # top-left corner
        boxes_xy2 = (
            boxes_scaled[:, :2] + boxes_scaled[:, 2:4] / 2
        )  # bottom-right corner
        boxes_xyxy = np.concatenate([boxes_xy, boxes_xy2], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), scores.tolist(), reliability_threshold, nms_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            res: List[YOLO_RESULT_DET] = []
            for i in indices:
                # 直接使用缩放后的xywh格式数据，避免重复转换
                box_xywh = boxes_scaled[i]

                score = scores[i]
                class_id = class_ids[i]
                re = self._result_type()
                re.index_in_all_boxes = i

                # 直接从xywh格式获取宽高和中心坐标
                re.x = int(box_xywh[0])  # x_center
                re.y = int(box_xywh[1])  # y_center
                re.w = int(box_xywh[2])  # width
                re.h = int(box_xywh[3])  # height
                re.xywh = box_xywh
                re.reliability = score
                re.label = class_id
                res.append(re)
            return res

        return []
