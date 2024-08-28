import cv2

def main():
    # 尝试打开摄像头
    video_device = '/dev/video1'  # 也可以尝试 '/dev/video2'，根据具体情况调整
    cap = cv2.VideoCapture(video_device)

    # 检查摄像头是否打开
    if not cap.isOpened():
        print(f"无法打开摄像头设备 {video_device}!")
        return

    print("摄像头已打开，按 'q' 键退出")

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法从摄像头读取帧!")
            break

        # 显示帧
        cv2.imshow("Camera", frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
