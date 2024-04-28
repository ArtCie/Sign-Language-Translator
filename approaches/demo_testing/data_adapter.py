from typing import Optional

import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.system("export CUDA_VISIBLE_DEVICES=\"\"")

FPS_SAVED = 3


class RecordDecoder:

    def __init__(self):
        self.hands_handler = mp_hands.Hands()

    # def record_2_pandas(self, record: np.ndarray) -> Optional[dict]:
    #     img_rgb = cv2.cvtColor(record, cv2.COLOR_BGR2RGB)
    #     results = self.hands_handler.process(img_rgb)
    #
    #     if not results.multi_hand_landmarks:
    #         return
    #
    #     result_dict = {}
    #     for hand_landmarks, hand_data in zip(results.multi_hand_landmarks, results.multi_handedness):
    #         hand = hand_data.classification[0].label.lower()
    #         if hand != "left":
    #             continue
    #         for counter, hand_landmark in enumerate(hand_landmarks.landmark):
    #             result_dict[f"{hand}_landmark_{counter}_x"] = hand_landmark.x
    #             result_dict[f"{hand}_landmark_{counter}_y"] = hand_landmark.y
    #             result_dict[f"{hand}_landmark_{counter}_z"] = hand_landmark.z
    #         mp_drawing.draw_landmarks(record, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
    #     return result_dict

    def record_2_jpg(self, record: np.ndarray):
        img_rgb = cv2.cvtColor(record, cv2.COLOR_BGR2RGB)
        results = self.hands_handler.process(img_rgb)

        if not results.multi_hand_landmarks:
            print("No hands found")
            return

        for hand_landmarks, hand_data in zip(results.multi_hand_landmarks, results.multi_handedness):

            hand = hand_data.classification[0].label.lower()
            if hand != "left" or hand_data.classification[0].score < 0.94:
                continue
            bounding_box = self.calculate_bounding_box(record, hand_landmarks)

            hand_image = self.crop_hand(record, bounding_box)

            if hand_image.size == 0:
                continue

            gray_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
            # thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)[1]

            # blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
            # # _, otsu_thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_OTSU)
            # eq_image = cv2.equalizeHist(blurred_image)
            #
            # v = np.median(eq_image)
            #
            # # Apply automatic Canny edge detection using the computed median
            # lower = int(max(0, (1.0 - 0.5) * v))
            # upper = int(min(255, (1.0 + 0.5) * v))
            # print(f"{lower} {upper}")
            # edges_images = cv2.Canny(eq_image, lower, upper)

            # morphological gradient
            edgemap = cv2.dilate(gray_image, None) - cv2.erode(gray_image, None)
            # blurred_image = cv2.GaussianBlur(edgemap, (3, 3), 0)
            # _, otsu_thresh_image = cv2.threshold(edgemap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            resized_image = cv2.resize(edgemap, (32, 32), interpolation=cv2.INTER_CUBIC)
            # pixelated_image = cv2.resize(resized_image, (640, 640), interpolation=cv2.INTER_NEAREST)

            return resized_image

            # mp_drawing.draw_landmarks(record, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
            # cv2.imshow('Hand', otsu_thresh_image)
            # cv2.waitKey(0)

    @staticmethod
    def calculate_bounding_box(image, landmarks):
        image_height, image_width, _ = image.shape
        landmark_array = np.array([(landmark.x * image_width, landmark.y * image_height)
                                   for landmark in landmarks.landmark])
        x, y, w, h = cv2.boundingRect(landmark_array.astype(int))
        return [x, y, x + w, y + h]

    @staticmethod
    def crop_hand(image, bounding_box):
        x1, y1, x2, y2 = bounding_box
        const = 0.3
        size_y = int((y2 - y1) * const)
        size_x = int((x2 - x1) * const)
        return image[max(0, y1 - size_y):y2 + size_y, max(0, x1 - size_x):x2 + size_x]


class DataHandler:
    DATA_LOCATION = "data/"

    def __init__(self, display=False):
        self._data = pd.DataFrame()
        self._record_decoder = RecordDecoder()
        self.DISPLAY = display

    def add_record(self, letter: str, record: np.ndarray, video_id, frame_id):
        print(f"{letter} {video_id} {frame_id}")
        hands_data = self._record_decoder.record_2_jpg(record)
        # hands_data = self._record_decoder.record_2_pandas(record)

        if hands_data is not None:
            df = pd.DataFrame(
                {
                    "label": [letter],
                    "video_id": [video_id],
                    "frame_id": [frame_id],
                    "data": [hands_data.flatten().tolist()]
                }
            )


            self._data = pd.concat([self._data, df], ignore_index=True)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def load(self):
        global_video_counter = 0
        for directory in os.listdir(self.DATA_LOCATION):
            labels_directories_path = self.DATA_LOCATION + directory
            if ".DS_Store" in labels_directories_path:
                continue
            for labels_file in os.listdir(labels_directories_path):
                self.load_video(directory, labels_directories_path, labels_file, global_video_counter)
                global_video_counter += 1

    def load_video(self, label, labels_directories_path, labels_file, video_id):
        if not labels_file.endswith(".mp4"):
            return
        full_path = labels_directories_path + "/" + labels_file
        cap = cv2.VideoCapture(full_path)

        # fps = cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0
        # frame_interval = round(fps / FPS_SAVED)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # if frame_id % frame_interval == 0:
            #     print(" SAVE ", end="")

            # if frame_id % 20 == 0:
            self.add_record(label, frame, video_id, frame_id)

            if self.DISPLAY:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1
        cap.release()

    def save_to_csv(self):
        file_name = "full_hand_data_big_set.csv"
        self._data.to_csv(file_name, index=False)


data_handler = DataHandler(display=False)
data_handler.load()
data_handler.save_to_csv()
