from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

model.train(
    data='C:/home/ubuntu/hand_keypoints_smoke.yaml',
    epochs=1,
    imgsz=640,
    batch=4,
    device='cpu',
    workers=0,
    project='C:/home/ubuntu/runs/hand_pose',
    name='smoke_test',
    exist_ok=True
)
