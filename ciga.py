import cv2
from ultralytics import YOLO
import pygame
import threading
import time

# YOLOv8 모델 로드 (학습한 모델 파일 경로)
model = YOLO('trained_model.pt')

# 웹캠 열기 (기본 카메라 0번 사용)
cap = cv2.VideoCapture(0)

# 알림 소리 파일 경로 (소리 파일을 프로젝트 폴더에 넣고 사용)
alert_sound = 'alert_sound.mp3'

# pygame 초기화
pygame.mixer.init()

# 소리 재생을 위한 비동기 함수
def play_alert_sound():
    while True:
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()
        print("소리 재생됨!")  # 소리 재생 시 출력
        time.sleep(1)  # 너무 자주 반복되지 않도록 잠시 대기

# 소리 재생을 위한 스레드 시작
alert_thread = threading.Thread(target=play_alert_sound, daemon=True)
alert_thread.start()

# 현재 소리가 재생 중인지 확인
is_playing = False

while True:
    ret, frame = cap.read()  # 웹캠으로부터 프레임 읽기

    if not ret:
        break

    # YOLOv8 모델로 객체 탐지
    results = model(frame)

    # 결과에서 탐지된 객체 확인
    cigarette_detected = False

    # results.pred[0]는 YOLO 모델이 반환하는 첫 번째 이미지에 대한 탐지 결과입니다
    for result in results[0].boxes:
        # result.xyxyn[5]는 탐지된 클래스 ID입니다.
        class_id = int(result.cls.item())  # 결과 클래스 ID 추출
        if model.names[class_id] == 'cigarette':  # 'cigarette' 클래스가 탐지되었는지 확인
            cigarette_detected = True
            break

    # 'cigarette'가 탐지되었을 때
    if cigarette_detected:
        print("Cigarette detected!")  # 탐지 시 출력
        if not is_playing:  # 소리가 재생 중이 아니면
            pygame.mixer.music.set_volume(1.0)  # 볼륨을 최대값으로 설정
            is_playing = True
    else:
        # 'cigarette'가 탐지되지 않았을 때
        print("No cigarette detected!")  # 탐지되지 않으면 출력
        pygame.mixer.music.set_volume(0.0)  # 볼륨을 0으로 설정
        is_playing = False

    # 탐지된 결과를 영상에 표시
    annotated_frame = results[0].plot()  # 탐지된 객체 표시

    # 실시간 웹캠 영상 출력
    cv2.imshow('Webcam Feed', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
