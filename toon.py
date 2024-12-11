import torch
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('trained_model.pt')  # 또는 YOLOv8의 모델 파일 경로

# 더미 입력 데이터 (ONNX 변환을 위한 샘플 데이터)
dummy_input = torch.randn(1, 3, 640, 640)  # 이미지 크기 (1, 3, 640, 640)

# 모델을 ONNX 형식으로 변환
onnx_model_path = 'yolov8_model.onnx'
torch.onnx.export(model.model, dummy_input, onnx_model_path, opset_version=12)
