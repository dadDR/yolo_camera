import sys
sys.path.append("./src")
import platform
import random
import cv2
from copy import copy
import numpy as np
from rknnlite.api import RKNNLite
import struct
import serial
import threading
import time
import serial
from uservo import UartServoManager

# 参数配置
# 角度定义
SERVO_PORT_NAME = '/dev/ttyUSB0' # 舵机串口号
SERVO_BAUDRATE = 115200 # 舵机的波特率
SERVO_ID = 1 # 舵机的ID号


# 初始化串口
uart = serial.Serial(port=SERVO_PORT_NAME, baudrate=SERVO_BAUDRATE,\
parity=serial.PARITY_NONE, stopbits=1,\
bytesize=8,timeout=0)



Servo1_angle = 0 #定义第一个舵机的角度
message = "S180!"


# 初始化舵机管理器
uservo = UartServoManager(uart, is_debug=True)

print("[单圈模式]设置舵机角度为90.0°")
uservo.set_servo_angle(SERVO_ID, 90.0, interval=0) # 设置舵机角度 极速模式
uservo.wait() # 等待舵机静止
print("-> {}".format(uservo.query_servo_angle(SERVO_ID)))



# 舵机通讯检测
is_online = uservo.ping(SERVO_ID)
print("舵机ID={} 是否在线: {}".format(SERVO_ID, is_online))


OBJ_THRESH = 0.55
NMS_THRESH = 0.45

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

# device tree for RK356x/RK3576/RK3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

RK3566_RK3568_RKNN_MODEL_PATH = './yolov10_rk356x.rknn'
RK3588_RKNN_MODEL_PATH = './yolov10_rk3588.rknn'
RK3562_RKNN_MODEL_PATH = './yolov10_rk3562.rknn'
RK3576_RKNN_MODEL_PATH = './yolov10_rk3576.rknn'


# Video capture device
video_device = '/dev/video1'  # 或者 '/dev/video2'

CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
           "scissors", "teddy bear", "hair drier", "toothbrush")

CLASS_COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASSES))]

def get_host():
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                elif 'rk3576' in device_compatible_str:
                    host = 'RK3576'
                elif 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = e_y / np.sum(e_y, axis=2, keepdims=True)
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy



def post_process(input_data):
    # 初始化结果列表
    boxes, scores, classes_conf = [], [], []
    # 定义默认的分支数目
    defualt_branch = 3
    # 计算每个分支的数据对数
    pair_per_branch = len(input_data) // defualt_branch

    # 遍历每个分支
    for i in range(defualt_branch):
        # 处理框数据
        boxes.append(box_process(input_data[pair_per_branch * i]))
        # 获取类别置信度
        classes_conf.append(input_data[pair_per_branch * i + 1])
        # 初始化分数为1，形状与类别置信度相同
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    # 定义一个函数将数据进行展平处理
    def sp_flatten(_in):
        ch = _in.shape[1]  # 获取通道数
        _in = _in.transpose(0, 2, 3, 1)  # 转置维度，使通道维度在最后
        return _in.reshape(-1, ch)  # 展平数据

    # 对每个分支的数据进行展平处理
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    # 合并所有分支的数据
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # 过滤框、类别和分数
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    return boxes, classes, scores


def draw(img, boxes, scores, classes):
    global Servo1_angle
    global message
    # 获取图像宽度
    img_width = img.shape[1]
    img_height = img.shape[0]
    # 计算图像中心的 x 坐标及允许的偏差范围
    center_x_threshold = img_width * 0.08  # 中心的 10% 偏差，即 10% 向左和 10% 向右
    center_x_left_limit = int(img_width * 0.5 - center_x_threshold)
    center_x_right_limit = int(img_width * 0.5 + center_x_threshold)


    # 在图像的左右限制位置画红色竖线
    cv2.line(img, (center_x_left_limit, 0), (center_x_left_limit, img_height), (0, 0, 255), 2)
    cv2.line(img, (center_x_right_limit, 0), (center_x_right_limit, img_height), (0, 0, 255), 2)

    # 遍历每个检测框、分数和类别
    for box, score, cls in zip(boxes, scores, classes):
        # 仅检测类别为“person”的目标
        if CLASSES[int(cls)] == "person":
            x0, y0, x1, y1 = map(int, box)  # 将检测框坐标转换为整数
            color = np.array(CLASS_COLORS[int(cls)])  # 确保 color 是一个 NumPy 数组

            # 绘制目标检测框
            cv2.rectangle(img, (x0, y0), (x1, y1), tuple(color.tolist()), 2)


            # 计算矩形框的质心
            centroid_x = int((x0 + x1) / 2)
            centroid_y = int((y0 + y1) / 2)

            # 绘制质心
            cv2.circle(img, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # 质心用红色圆点标记


            # 绘制标签
            text = '{} {:.1f}%'.format(CLASSES[int(cls)], score * 100)  # 标签文本
            txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)  # 选择文本颜色
            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]  # 计算文本大小
            txt_bk_color = (color * 0.7).astype(np.uint8)  # 计算文本背景颜色
            txt_bk_color = tuple(txt_bk_color.tolist())  # 将背景颜色转换为元组
            # 绘制文本背景矩形
            cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(txt_size[1] * 1.5)), txt_bk_color, -1)
            # 绘制文本
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

            # 计算目标的中心 x 坐标
            person_center_x = (x0 + x1) / 2


            if centroid_x > center_x_left_limit and  centroid_x < center_x_right_limit:
                uservo.set_servo_angle(SERVO_ID, Servo1_angle, velocity=50.0, t_acc=200, t_dec=200) # 设置舵机角度 极速模式
            if centroid_x < center_x_left_limit:
                if Servo1_angle >= 180:
                    Servo1_angle = 180
                if Servo1_angle < 180:
                    Servo1_angle = Servo1_angle - 1.5
                uservo.set_servo_angle(SERVO_ID,Servo1_angle, velocity=50.0, t_acc=200, t_dec=200) # 设置舵机角度 极速模式
            if centroid_x > center_x_right_limit:
                if Servo1_angle <= -180:
                    Servo1_angle = -180
                if Servo1_angle > -180:
                    Servo1_angle = Servo1_angle + 1.5
                uservo.set_servo_angle(SERVO_ID, Servo1_angle, velocity=50.0, t_acc=200, t_dec=200) # 设置舵机角度 极速模式


            # 根据目标位置选择合适的方向并打印到控制台
          #  if person_center_x > center_x_left_limit:
                #让舵机向左转
           #     if Servo1_angle >= 180:
           #         Servo1_angle = 180
           #     if Servo1_angle < 180:
            #        Servo1_angle = Servo1_angle + 1.5
                    #调整舵机角度
            #        Sertemp = Servo1_angle*2
            #        message = "S%d!" % Sertemp
             #       uservo.set_servo_angle(SERVO_ID, Servo1_angle, interval=0) # 设置舵机角度 极速模式
#                    uart.write(message.encode('utf-8'))
           #     print('Direction: Left')  # 向左
           # elif person_center_x < center_x_right_limit:
               # if Servo1_angle <= -180:
               #     Servo1_angle = -1800
               # if Servo1_angle > -180:
                  #  Servo1_angle = Servo1_angle - 1.5
                  #  Sertemp = Servo1_angle*2
                 #   message = "S%d!" % Sertemp
                #    uservo.set_servo_angle(SERVO_ID, Servo1_angle, interval=0) # 设置舵机角度 极速模式
           # else :
            #    uservo.set_servo_angle(SERVO_ID, Servo1_angle, interval=0) # 设置舵机角度 极速模式
 #                   uart.write(message.encode('utf-8'))
           #     print('Direction: Right')  # 向右
           # else:
           #     print('Direction: Center')  # 中心
            #uart.write(message.encode('utf-8'))
            

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # 增加一个维度以匹配模型期望的输入形状（四维）
    img = np.expand_dims(img, axis=0)

    return img

def main():
    global Servo1_angle
    global message
    host = get_host()

    if host == 'RK3562':
        rknn_model_path = RK3562_RKNN_MODEL_PATH
    elif host == 'RK3576':
        rknn_model_path = RK3576_RKNN_MODEL_PATH
    elif host == 'RK3588':
        rknn_model_path = RK3588_RKNN_MODEL_PATH
    else:
        rknn_model_path = RK3566_RK3568_RKNN_MODEL_PATH

    rknn_lite = RKNNLite()
    # Load model
    print('--> Load model')
    ret = rknn_lite.load_rknn(rknn_model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # 打开摄像头
    cap = cv2.VideoCapture(video_device)
    if not cap.isOpened():
        print('Failed to open camera!')
        return

    #初始化舵机角度(180/2 = 90 度)
    uart.write(b"S180!")
    Servo1_angle = 90
    message = "S180!"


    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read frame from camera!')
            break

        img = process_image(frame)

        outputs = rknn_lite.inference(inputs=[img])
        boxes, classes, scores = post_process(outputs)
        draw(frame, boxes, scores, classes)
        
        frame_resized = cv2.resize(frame, (640, 480))
        cv2.imshow("frame", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    uart.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

