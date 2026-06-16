import cv2
import mediapipe as mp
import Sound as s
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#수면 카운트
sleep_cnt = 1

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요
            continue

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 이미지 위에 얼굴 그물망 주석을 그립니다. https://puleugo.tistory.com/5
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #좌표 구하는 파트
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #왼눈 위
                coordinates = face_landmarks.landmark[159]
                leftup_x = coordinates.x * 10000
                leftup_y = coordinates.y * 10000
                leftup_z = coordinates.z *1000
                #왼눈 아래
                coordinates = face_landmarks.landmark[145]
                leftdown_x = coordinates.x * 10000
                leftdown_y = coordinates.y * 10000
                leftdown_z = coordinates.z *1000
                #오른눈 위
                coordinates = face_landmarks.landmark[386]
                rightup_x = coordinates.x * 10000
                rightup_y = coordinates.y * 10000
                rightup_z = coordinates.z *1000
                #오른눈 아래
                coordinates = face_landmarks.landmark[374]
                rightdown_x = coordinates.x * 10000
                rightdown_y = coordinates.y * 10000
                rightdown_z = coordinates.z *1000
                cv2.putText( 
                    image, text = 'LU = (%d,%d), LD = (%d,%d), RU = (%d,%d), RD = (%d,%d)' %(leftup_x, leftdown_y, leftdown_x,leftdown_y,rightup_x, rightup_y ,rightdown_x , rightdown_y), org = (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=255, thickness=2)
                #딜레이 200밀리초
                if cv2.waitKey(200) > 0 :
                    break
                sleep_cnt += 1
                #Left 두점의 거리 측정
                LRan = math.sqrt((leftup_x - leftdown_x)**2 + (leftup_y - leftdown_y)**2)
                RRan = math.sqrt((rightup_x - rightdown_x)**2 + (rightup_y - rightdown_y)**2)

                if (LRan < 100):
                    print("졸림 상태")
                    

                    if sleep_cnt >= 30:
                        sleep_cnt = 1
                        print("30초간 졸고 있네요!!!")
                        s.beepsound()
                        s.send_music_link()
                        break ##1번만 알림이 오면 프로그램 정지 시킴
                else:
                    print("깨어있는 상태")
                    sleep_cnt = 1

                #얼굴에 매쉬 생성.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Face Mesh(Puleugo)', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
