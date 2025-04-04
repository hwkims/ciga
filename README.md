📺 **[동영상 보기](https://www.youtube.com/watch?v=9ECnFEY9Sg0)** 🎬

🔗 **[블로그 프로젝트 링크](https://velog.io/@hwkims/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EB%8B%B4%EB%B0%B0%EA%BD%81%EC%B4%88)** 🌐

https://hwkims.github.io/ciga/demo.html
https://v0-ciga-h3ywai.vercel.app/


# ciga
Cigarette Detection yolo

### **README - YOLOv8 모델을 사용한 실시간 객체 탐지 및 알림 시스템**

---

#### **1. 개요**
이 프로젝트는 **YOLOv8 (You Only Look Once version 8)** 모델을 사용하여 **실시간 객체 탐지**를 수행하고, 특정 객체 (예: 담배)가 탐지되었을 때 **알림 소리**를 울리도록 구현한 시스템입니다. 이 시스템은 **OpenCV**, **Ultralytics YOLO**, 그리고 **pygame** 라이브러리를 활용하여 웹캠에서 실시간 비디오 스트림을 캡처하고, 이를 처리하여 탐지된 객체에 대한 시각적 표시 및 소리 알림을 제공합니다.

이 시스템은 다음과 같은 구성 요소로 이루어져 있습니다:
- **YOLOv8 모델**을 사용한 객체 탐지
- **웹캠 비디오 스트림**을 사용하여 실시간 영상 입력
- **알림 소리**: 특정 객체(예: 담배)가 탐지되면 소리를 재생
- **스레드 기반의 비동기 처리**: 알림 소리와 객체 탐지 처리 분리

---

#### **2. 주요 기능**

1. **YOLOv8 모델로 실시간 객체 탐지**:
   - 웹캠에서 입력된 영상을 YOLOv8 모델로 처리하여 객체를 탐지합니다.
   - 탐지된 객체는 **Bounding Box**와 함께 화면에 표시됩니다.

2. **알림 소리**:
   - 특정 객체(예: 담배)가 탐지되면 알림 소리를 재생합니다.
   - 소리는 `pygame` 라이브러리를 사용하여 재생되며, 소리가 반복되지 않도록 관리합니다.

3. **스레드를 활용한 비동기 소리 재생**:
   - 소리 재생은 별도의 스레드에서 처리되어, 객체 탐지와 동시에 소리가 반복적으로 재생될 수 있도록 구현됩니다.

4. **실시간 웹캠 스트리밍**:
   - OpenCV를 이용하여 웹캠에서 실시간으로 영상을 캡처하고, 이를 표시합니다.

---

#### **3. 동작 원리**

이 시스템은 두 주요 프로세스를 병행하여 처리합니다: 객체 탐지와 소리 재생. 전체 시스템의 동작 흐름은 아래와 같습니다.

1. **YOLOv8 모델 로드**:
   - 학습된 YOLOv8 모델 파일(`trained_model.pt`)을 로드합니다.
   - 이 모델은 웹캠에서 실시간으로 캡처된 이미지에서 객체를 탐지하는 데 사용됩니다.

2. **웹캠 영상 캡처**:
   - `cv2.VideoCapture(0)`을 사용하여 기본 카메라(Webcam)에서 비디오 스트림을 가져옵니다.
   - 각 프레임을 YOLOv8 모델에 전달하여 객체 탐지를 수행합니다.

3. **객체 탐지 및 알림 소리 재생**:
   - YOLOv8 모델은 객체를 탐지하여 **Bounding Box**를 이미지에 그립니다.
   - 탐지된 객체가 'cigarette'라면 알림 소리(`alert_sound.mp3`)가 재생됩니다.
   - 소리는 `pygame.mixer.music`을 사용하여 재생되며, 소리 재생 여부는 별도의 스레드에서 처리됩니다.

4. **객체 탐지 결과 출력**:
   - 객체 탐지 결과(탐지된 클래스 및 해당 위치)는 **Bounding Box** 형태로 실시간 비디오에 오버레이됩니다.
   - 탐지된 객체가 없다면, 탐지되지 않은 상태로 비디오가 출력됩니다.

---

#### **4. 코드 분석**

##### **4.1. 모델 로드 및 웹캠 설정**

```python
# YOLOv8 모델 로드 (학습한 모델 파일 경로)
model = YOLO('trained_model.pt')

# 웹캠 열기 (기본 카메라 0번 사용)
cap = cv2.VideoCapture(0)
```

- `YOLO('trained_model.pt')`: YOLOv8 모델을 로드합니다. 이 모델은 **자신이 학습한 객체 인식 모델**을 사용하여 실시간 탐지를 수행합니다.
- `cv2.VideoCapture(0)`: OpenCV를 사용하여 기본 웹캠에서 비디오 스트림을 캡처합니다.

##### **4.2. 알림 소리 처리**

```python
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
```

- `pygame.mixer.init()`: `pygame`의 믹서 모듈을 초기화합니다. 소리 파일을 재생할 준비를 합니다.
- `play_alert_sound()`: 알림 소리를 무한 반복하여 재생하는 함수입니다. 소리가 재생될 때마다 1초 간격을 두고 반복됩니다.
- `threading.Thread(target=play_alert_sound, daemon=True)`: 별도의 스레드를 사용하여 소리를 비동기적으로 처리합니다. 이는 객체 탐지와 소리 재생이 동시에 이루어질 수 있게 합니다.

##### **4.3. 객체 탐지 및 소리 알림**

```python
# YOLOv8 모델로 객체 탐지
results = model(frame)

# 결과에서 탐지된 객체 확인
cigarette_detected = False

# results.pred[0]는 YOLO 모델이 반환하는 첫 번째 이미지에 대한 탐지 결과입니다
for result in results[0].boxes:
    class_id = int(result.cls.item())  # 결과 클래스 ID 추출
    if model.names[class_id] == 'cigarette':  # 'cigarette' 클래스가 탐지되었는지 확인
        cigarette_detected = True
        break
```

- `results = model(frame)`: 모델을 사용하여 입력된 `frame`에서 객체 탐지를 수행합니다.
- `results[0].boxes`: 탐지된 객체들의 정보를 가져옵니다. 각 객체는 **클래스 ID**와 **경계 상자 좌표** 등의 정보를 포함합니다.
- `if model.names[class_id] == 'cigarette'`: 탐지된 객체가 'cigarette'인지 확인하고, 해당 객체가 발견되면 `cigarette_detected`를 `True`로 설정합니다.

##### **4.4. 소리 재생 제어**

```python
if cigarette_detected:
    print("Cigarette detected!")  # 탐지 시 출력
    if not is_playing:  # 소리가 재생 중이 아니면
        pygame.mixer.music.set_volume(1.0)  # 볼륨을 최대값으로 설정
        is_playing = True
else:
    print("No cigarette detected!")  # 탐지되지 않으면 출력
    pygame.mixer.music.set_volume(0.0)  # 볼륨을 0으로 설정
    is_playing = False
```

- `pygame.mixer.music.set_volume(1.0)`: 담배가 탐지된 경우 알림 소리를 **최대 볼륨**으로 설정하여 재생합니다.
- `pygame.mixer.music.set_volume(0.0)`: 담배가 탐지되지 않은 경우 소리를 **음소거**로 설정합니다.

##### **4.5. 실시간 웹캠 출력**

```python
# 탐지된 결과를 영상에 표시
annotated_frame = results[0].plot()  # 탐지된 객체 표시

# 실시간 웹캠 영상 출력
cv2.imshow('Webcam Feed', annotated_frame)
```

- `results[0].plot()`: 탐지된 객체에 대한 **Bounding Box**를 그려서 반환합니다.
- `cv2.imshow('Webcam Feed', annotated_frame)`: 실시간으로 처리된 프레임을 화면에 출력합니다.

---

#### **5. 종료 및 리소스 해제**

```python
# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
```

- `cap.release()`: 웹캠 스트리밍을 종료합니다.
- `cv2.destroyAllWindows()`: OpenCV 창을 닫습니다.

---

#### **6. 결론**

이 시스템은 **YOLOv8 모델**을 사용하여 실시간으로 객체를 탐지하고, **특정 객체**가 탐지되면 **알림 소리**를 재생하는 간단한 객체 탐지 시스템입니다. `pygame`과 `threading`을 활용하여 소리 재생을 비동기적으로 처리하고, **OpenCV**를 사용하여 실시간으로 웹캠 영상에 탐지 결과를 표시합니다.
![confusion_matrix_normalized](https://github.com/user-attachments/assets/8c02c9d4-bc14-4321-872f-f42b272f3781)
![confusion_matrix](https://github.com/user-attachments/assets/18b9ee93-cd06-4923-98ef-8feff489b164)
![val_batch2_pred](https://github.com/user-attachments/assets/0ce19e7e-1848-4bb2-9cb9-dbcfca138bd2)
![val_batch2_labels](https://github.com/user-attachments/assets/0ce8484e-5a82-4c02-b554-42c58a80c7e4)
![val_batch1_pred](https://github.com/user-attachments/assets/63766593-0375-472d-b908-4c613e7a2c9e)
![val_batch1_labels](https://github.com/user-attachments/assets/fd07b9a7-9248-4a45-a0b5-bd5384f0b2b2)
![val_batch0_pred](https://github.com/user-attachments/assets/de2af524-ee95-4796-b73a-967e78706b1c)
![val_batch0_labels](https://github.com/user-attachments/assets/0b9126e6-d72a-417c-9f6b-6fdbfef83f18)
![train_batch2](https://github.com/user-attachments/assets/22bf7c68-b6bc-4e1c-a10a-3b14f2e3fb16)
![train_batch1](https://github.com/user-attachments/assets/93c75e15-e50c-4cdb-bcd5-660170fbb303)
![train_batch0](https://github.com/user-attachments/assets/669c1d8e-b318-4cd4-bb2b-efd21354b49b)
![results](https://github.com/user-attachments/assets/275ddf62-b244-440a-9640-db57091a3d89)
[results.csv](https://github.com/user-attachments/files/18089864/results.csv)
![R_curve](https://github.com/user-attachments/assets/4d476c3c-6fc3-49b0-851f-797652f1dbc4)
![PR_curve](https://github.com/user-attachments/assets/46386b4b-cb14-4ab1-89db-02b6b0ea138b)
![P_curve](https://github.com/user-attachments/assets/88cc3689-970c-46d1-bbe5-41e16fc7e6e8)
![labels_correlogram](https://github.com/user-attachments/assets/07fadfe8-3624-4682-8eea-ab892711b599)
![labels](https://github.com/user-attachments/assets/64dd8031-7755-4046-9df2-c4829a68c394)
![F1_curve](https://github.com/user-attachments/assets/a58e0c54-3f51-4589-b31d-eb07ff8a3819)
