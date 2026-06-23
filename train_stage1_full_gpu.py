from ultralytics import YOLO


def main():
    model = YOLO('yolo11n-pose.pt')
    model.train(
        data='C:/home/ubuntu/hand_keypoints_full.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        pretrained=True,
        project='C:/home/ubuntu/runs/hand_pose',
        name='baseline_full_gpu_10e',
        exist_ok=True
    )


if __name__ == '__main__':
    main()
