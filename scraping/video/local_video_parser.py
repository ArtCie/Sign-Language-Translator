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

    def record_2_dict(self, record: np.ndarray) -> Optional[dict]:
        img_rgb = cv2.cvtColor(record, cv2.COLOR_BGR2RGB)
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
            # mp_drawing.draw_landmarks(record, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
        return result_dict


class DataHandler:
    DATA_LOCATION = "../../dane/"
    COUNTER_START = 20

    def __init__(self):
        self._record_decoder = RecordDecoder()
        self.current_movie_counter = self.COUNTER_START

    def transform_to_individual(self):
        for directory in os.listdir(self.DATA_LOCATION):
            labels_directories_path = self.DATA_LOCATION + directory
            if ".DS_Store" in labels_directories_path:
                continue
            for labels_file in os.listdir(labels_directories_path):
                self.load_video(directory, labels_directories_path, labels_file)
                self.current_movie_counter = self.COUNTER_START

    def load_video(self, label, labels_directories_path, labels_file):
        if not labels_file.endswith(".mov"):
            return

        full_path = labels_directories_path + "/" + labels_file
        cap = cv2.VideoCapture(full_path)
        frames_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hands_data = self._record_decoder.record_2_dict(frame)

            if hands_data is not None:
                frames_buffer.append(frame)
            elif frames_buffer:
                if len(frames_buffer) > 11:
                    self.save_frames_to_video(frames_buffer, label)
                frames_buffer = []
                self.current_movie_counter += 1

    def save_frames_to_video(self, frames, label):
        save_path = f"{self.DATA_LOCATION}{label}/{self.current_movie_counter}.mp4"
        frames = frames[5:-5]  # cut first five and last five frames
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'))

        for frame in frames:
            video.write(frame)

        video.release()
        print(f"Saved video {save_path}")
        return


data_handler = DataHandler()
data_handler.transform_to_individual()
