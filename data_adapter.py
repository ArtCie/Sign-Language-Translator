from typing import Optional

import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage import io, color, morphology


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.system("export CUDA_VISIBLE_DEVICES=\"\"")

sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])


# FPS_SAVED = 15


class RecordDecoder:
    def __init__(self, display=False):
        self.hands_handler = mp_hands.Hands()
        self.DISPLAY = display

    def record_2_pandas(self, record: np.ndarray) -> Optional[dict]:
        img_rgb = cv2.cvtColor(record, cv2.COLOR_BGR2RGB)
        # results = mp_hands.Hands().process(img_rgb)
        results = self.hands_handler.process(img_rgb)

        if not results.multi_hand_landmarks:
            return

        result_dict = {}
        for hand_landmarks, hand_data in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand = hand_data.classification[0].label.lower()
            if hand != "left" or hand_data.classification[0].score < 0.94:
                continue
            for counter, hand_landmark in enumerate(hand_landmarks.landmark):
                result_dict[f"{hand}_landmark_{counter}_x"] = hand_landmark.x
                result_dict[f"{hand}_landmark_{counter}_y"] = hand_landmark.y
                result_dict[f"{hand}_landmark_{counter}_z"] = hand_landmark.z
            mp_drawing.draw_landmarks(record, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
        return result_dict

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

            # hsv_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2HSV)
            #
            # lower = np.array([0, 80, 100])  # Lower range of HSV color
            # upper = np.array([22, 255, 255])  # Upper range of HSV color
            # mask = cv2.inRange(hsv_image, lower, upper)
            #
            # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            # if not contours:
            #     continue
            #
            # largest_contour = max(contours, key=cv2.contourArea)
            # largest_contour_mask = np.zeros_like(mask)
            # cv2.drawContours(largest_contour_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
            #
            # masked_hand_image = cv2.bitwise_and(hand_image, hand_image, mask=largest_contour_mask)
            #
            # blurred_image = cv2.bilateralFilter(masked_hand_image, d=3, sigmaColor=75, sigmaSpace=75)
            #
            # if self.DISPLAY:
            #     cv2.imshow("bilateral filter", blurred_image)
            #
            # kernel = np.ones((5, 5), np.uint8)
            # opened = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel, iterations=1)
            #
            # opened = cv2.medianBlur(opened, 5)
            #
            # canny_edges = cv2.Canny(opened, threshold1=50, threshold2=120)
            #
            # if self.DISPLAY:
            #     cv2.imshow("opened", opened)
            #     cv2.imshow("opened", opened)
            #     cv2.imshow("canny", canny_edges)

            # part 2 fin
            #
            hsv_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2HSV)

            lower = np.array([0, 80, 100])  # Lower range of HSV color
            upper = np.array([22, 255, 255])  # Upper range of HSV color
            mask = cv2.inRange(hsv_image, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour_mask = np.zeros_like(mask)
            cv2.drawContours(largest_contour_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)

            # thinned_image = morphology.skeletonize(
            #     largest_contour_mask // 255)
            #
            # thinned_display = (thinned_image * 255).astype(np.uint8)

            gray_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)

            edgemap = (cv2.dilate(gray_image, None) - cv2.erode(gray_image, None)) * 3

            # combined_image = cv2.add(edgemap, thinned_display)

            ret, thresh = cv2.threshold(edgemap, 25, 255, cv2.THRESH_TOZERO)

            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            #
            # dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
            # eroded_image = cv2.erode(gray_image, kernel, iterations=1)
            # edgemap = dilated_image - eroded_image

            # canny_edges = cv2.Canny(edgemap, threshold1=20, threshold2=150)

            # masked_hand_image = cv2.bitwise_and(edgemap, edgemap, mask=largest_contour_mask)

            res_16 = cv2.resize(thresh, (16, 16), interpolation=cv2.INTER_CUBIC)
            res_32 = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
            res_80 = cv2.resize(thresh, (80, 80), interpolation=cv2.INTER_CUBIC)

            if self.DISPLAY:
                cv2.imshow("edgemap", edgemap)
                # cv2.imshow("canny_edges", canny_edges)
                cv2.imshow("res_80", res_80)
                cv2.imshow("thresh", thresh)

            return res_16, res_32, res_80

    def record_2_jpg_skel(self, record: np.ndarray):
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

            hsv_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2HSV)

            lower = np.array([0, 80, 100])  # Lower range of HSV color
            upper = np.array([22, 255, 255])  # Upper range of HSV color

            mask = cv2.inRange(hsv_image, lower, upper)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)

            largest_contour_mask = np.zeros_like(mask)
            cv2.drawContours(largest_contour_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)

            thinned_image = morphology.skeletonize(
                largest_contour_mask // 255)

            thinned_display = (thinned_image * 255).astype(np.uint8)

            if self.DISPLAY:
                cv2.imshow("mask", mask)
                cv2.imshow("hand_image", hand_image)
                cv2.imshow("orange_objects", mask)
                cv2.imshow("mask", largest_contour_mask)
                cv2.imshow("thinned", thinned_display)

            resized_image_16 = cv2.resize(thinned_display, (16, 16), interpolation=cv2.INTER_CUBIC)
            resized_image_32 = cv2.resize(thinned_display, (32, 32), interpolation=cv2.INTER_CUBIC)
            resized_image_80 = cv2.resize(thinned_display, (80, 80), interpolation=cv2.INTER_CUBIC)
            return resized_image_16, resized_image_32, resized_image_80

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
        const = 0.1
        size_y = int((y2 - y1) * const)
        size_x = int((x2 - x1) * const)
        return image[max(0, y1 - size_y):y2 + size_y, max(0, x1 - size_x):x2 + size_x]


class DataHandler:
    DATA_LOCATION = "dane/"
    OUTPUT_VIDEO_LOCATION = f"{os.getcwd()}/approaches/canny_edges/data_fir"

    def __init__(self, display=False, save_video=False):
        self._data = pd.DataFrame()
        self._record_decoder = RecordDecoder(display=display)
        self.DISPLAY = display
        self.save_video = save_video

    def add_record(self, letter: str, record: np.ndarray, video_id, frame_id):
        print(f"{letter} {video_id} {frame_id}")
        hands_data = self._record_decoder.record_2_pandas(record)
        if hands_data is not None:
            hands_data["label"] = letter
            hands_data["video_id"] = video_id
            hands_data["frame_id"] = frame_id
            self._data = pd.concat([self._data, pd.DataFrame(hands_data, index=[0])], ignore_index=True)

    def add_record_jpg(self, letter: str, record: np.ndarray, video_id, frame_id, jpg_method):
        print(f"{letter} {video_id} {frame_id}")
        if jpg_method == "image_canny":
            res = self._record_decoder.record_2_jpg(record)
        else:
            res = self._record_decoder.record_2_jpg_skel(record)

        if res is not None:
            df = pd.DataFrame(
                {
                    "label": [letter],
                    "video_id": [video_id],
                    "frame_id": [frame_id],
                    "data": [res[1].flatten().tolist()]
                }
            )
            self._data = pd.concat([self._data, df], ignore_index=True)

        return res

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def load(self, method="hand"):
        global_video_counter = 0
        for directory in os.listdir(self.DATA_LOCATION):
            labels_directories_path = self.DATA_LOCATION + directory
            if ".DS_Store" in labels_directories_path:
                continue
            for labels_file in os.listdir(labels_directories_path):
                self.load_video(directory, labels_directories_path, labels_file, global_video_counter, method)
                global_video_counter += 1

    def load_video(self, label, labels_directories_path, labels_file, video_id, method="hand"):
        if not labels_file.endswith(".mp4"):
            return

        full_path = labels_directories_path + "/" + labels_file
        cap = cv2.VideoCapture(full_path)

        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))

        # size = (frame_width, frame_height)
        # fps = cap.get(cv2.CAP_PROP_FPS)

        if self.save_video:
            for i in ["16", "32", "80"]:
                label_path = f"{self.OUTPUT_VIDEO_LOCATION}_{i}/{label}"
                output_file_dir = f"{label_path}/{video_id}/"
                # output_file_dir = f"{label_path}/{video_id}"
                if not os.path.exists(label_path):
                    os.mkdir(label_path)
                if not os.path.exists(output_file_dir):
                    os.mkdir(output_file_dir)

        frame_id = 0

        new_frame = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if method == "image":
                new_frame = self.add_record_jpg(label, frame, video_id, frame_id, jpg_method="image_canny")
            elif method == "image_skel":
                new_frame = self.add_record_jpg(label, frame, video_id, frame_id, jpg_method="image_skel")
            else:
                self.add_record(label, frame, video_id, frame_id)

            if self.DISPLAY and new_frame is not None:
                # cv2.imshow('frame', new_frame)
                if cv2.waitKey() & 0xFF == ord('q'):
                    break

            if self.save_video and new_frame is not None:
                _16, _32, _80 = new_frame
                cv2.imwrite(f"{self.OUTPUT_VIDEO_LOCATION}_16/{label}/{video_id}/{frame_id}.jpg", _16)
                cv2.imwrite(f"{self.OUTPUT_VIDEO_LOCATION}_32/{label}/{video_id}/{frame_id}.jpg", _32)
                cv2.imwrite(f"{self.OUTPUT_VIDEO_LOCATION}_80/{label}/{video_id}/{frame_id}.jpg", _80)
                # cv2.imwrite(f"{output_file_dir}{frame_id}.jpg", new_frame)
                # cv2.imwrite(f"{output_file_dir}{frame_id}.jpg", new_frame)

            frame_id += 1

        cap.release()

    def save_to_csv(self):
        file_name = f"{os.getcwd()}/approaches/canny_edges/csvki/32_base_thresh_bez_tla.csv"
        self._data.to_csv(file_name, index=False)


data_handler = DataHandler(display=False, save_video=False)
data_handler.load(method="image")
data_handler.save_to_csv()
