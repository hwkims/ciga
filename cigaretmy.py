import cv2
from ultralytics import YOLO
from playsound import playsound
import threading

# YOLOv8 모델 로드 (학습한 모델 파일 경로)
model = YOLO('trained_model.pt')

# 웹캠 열기 (기본 카메라 0번 사용)
cap = cv2.VideoCapture(0)

# 알림 소리 파일 경로 (소리 파일을 프로젝트 폴더에 넣고 사용)
alert_sound = 'alert_sound.mp3'

# 소리 재생을 위한 비동기 함수
def play_alert_sound():
    while True:
        playsound(alert_sound)
        print("소리 재생됨!")  # 소리 재생 시 출력

# 소리 재생을 위한 스레드 시작
alert_thread = threading.Thread(target=play_alert_sound, daemon=True)
alert_thread.start()

while True:
    ret, frame = cap.read()  # 웹캠으로부터 프레임 읽기

    if not ret:
        break

    # YOLOv8 모델로 객체 탐지
    results = model(frame)

    # 결과에서 탐지된 객체 확인
    cigarette_detected = False
    for result in results:
        # 'cigarette' 객체를 탐지했을 때
        if 'cigarette' in result.names:
            cigarette_detected = True
            break

    # 'cigarette'가 탐지되었을 때만 소리 계속 재생
    if cigarette_detected:
        print("Cigarette detected!")  # 탐지 시 출력

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
