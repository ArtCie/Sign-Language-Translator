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


class RecordDecoder:
    def __init__(self):
        self.hands_handler = mp_hands.Hands()

    def record_2_pandas(self, record: np.ndarray) -> Optional[dict]:
        img_rgb = cv2.cvtColor(record, cv2.COLOR_BGR2RGB)
        # results = mp_hands.Hands().process(img_rgb)
        results = self.hands_handler.process(img_rgb)

        if not results.multi_hand_landmarks:
            return

        result_dict = {}
        for hand_landmarks, hand_data in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand = hand_data.classification[0].label.lower()

            if hand != "left":
                continue

            for counter, hand_landmark in enumerate(hand_landmarks.landmark):
                result_dict[f"{hand}_landmark_{counter}_x"] = hand_landmark.x
                result_dict[f"{hand}_landmark_{counter}_y"] = hand_landmark.y
                result_dict[f"{hand}_landmark_{counter}_z"] = hand_landmark.z
            mp_drawing.draw_landmarks(record, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
        return result_dict


class DataHandler:
    DATA_LOCATION = "data/"

    def __init__(self):
        self._data = pd.DataFrame()
        self._record_decoder = RecordDecoder()

    def add_record(self, letter: str, record: np.ndarray, video_id, frame_id):
        print(f"{letter} {video_id} {frame_id}")
        hands_data = self._record_decoder.record_2_pandas(record)
        if hands_data is not None:
            hands_data["label"] = letter
            hands_data["video_id"] = video_id
            hands_data["frame_id"] = frame_id
            self._data = pd.concat([self._data, pd.DataFrame(hands_data, index=[0])], ignore_index=True)

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
        if not labels_file.endswith((".mp4", ".mov")):
            return
        frame_id = 0
        full_path = labels_directories_path + "/" + labels_file
        cap = cv2.VideoCapture(full_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.add_record(label, frame, video_id, frame_id)
            frame_id += 1

            if video_id % 5 == 0:
                self._record_decoder.record_2_pandas(frame)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()


data_handler = DataHandler()
data_handler.load()
