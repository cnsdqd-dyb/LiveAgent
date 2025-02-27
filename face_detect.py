import cv2
import face_recognition
import os
import random
import threading


class FaceDetector:
    def __init__(self, cap_name=""):
        self.known_face_encodings = []
        self.known_face_names = []
        self.faces_dir = "faces/"
        self.cap_name = cap_name
        self.interval = 10
        self.current_faces = []

        self.load_known_faces()
        
    def load_known_faces(self):
        for filename in os.listdir(self.faces_dir):
            if filename.endswith(".jpg"):
                image = face_recognition.load_image_file(os.path.join(self.faces_dir, filename))

                print(f"加载: {filename}")
                try:
                    encoding = face_recognition.face_encodings(image)[0]
                except:
                    os.remove(os.path.join(self.faces_dir, filename))
                    continue
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(filename[:-4])
                
    def detect_faces(self, frame):
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        names = []
        centers = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.3)
            name = ""
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                names.append(name.split("_")[0])

                face_center = (int((left + right) / 2)/frame.shape[1], int((top + bottom) / 2)/frame.shape[0])
                centers.append(face_center)
            elif self.cap_name != "":
                self.known_face_encodings.append(face_encoding)
                face_image = frame[top:bottom, left:right]
                rand_id = random.randint(0, 100000)
            
                cv2.imwrite(f"{self.faces_dir}{self.cap_name}_{rand_id}.jpg", face_image)
                self.known_face_names.append(f"{self.cap_name}_{rand_id}")
                print("已保存新的人脸！")
        
        self.current_faces = names
        return names, centers
            
        
# # 初始化摄像头
# video_capture = cv2.VideoCapture(0)

# # 初始化变量
# known_face_encodings = []
# known_face_names = []

# # 加载已知人脸
# faces_dir = "faces/"
# for filename in os.listdir(faces_dir):
#     if filename.endswith(".jpg"):
#         # 加载人脸图像并编码
#         image = face_recognition.load_image_file(os.path.join(faces_dir, filename))
#         print(f"加载已知人脸: {filename}")
#         try:
#             encoding = face_recognition.face_encodings(image)[0]
#         except:
#             # remove invalid image
#             os.remove(os.path.join(faces_dir, filename))
#             continue
#         known_face_encodings.append(encoding)
#         known_face_names.append(filename[:-4])  # 去掉文件扩展名

# cap_name = "dongyubo"
# while True:
#     # 捕获一帧视频
#     ret, frame = video_capture.read()

#     # 将图像从BGR转换为RGB
#     rgb_frame = frame[:, :, ::-1]

#     # 查找当前帧中所有人脸及其编码
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # 与已知人脸进行比较
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.3)
#         name = "Unknown"

#         # 如果找到匹配的人脸，则获取其名称
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]
#         else:
#             # 保存新检测到的人脸
#             known_face_encodings.append(face_encoding)
#             face_image = frame[top:bottom, left:right]
#             rand_id = random.randint(0, 100000)
#             cv2.imwrite(f"{faces_dir}{cap_name}_{rand_id}.jpg", face_image)  # 保存为已知人脸
#             known_face_names.append(f"{cap_name}_{rand_id}")
#             print("已保存新的人脸！")

#         # 在图像上绘制人脸框和名称
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#     # 显示结果
#     cv2.imshow('Video', frame)

#     # 按'q'键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放摄像头和关闭窗口
# video_capture.release()
# cv2.destroyAllWindows()
