import cv2
import easyocr
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context


PATH_OUT = "../../../dane/"
PATH_IN = "videos/"


def process_video(input_video):
    ocr_manager = easyocr.Reader(['pl'], gpu=True)
    cap = cv2.VideoCapture(PATH_IN + input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_counter = 0
    current_letter = "A"
    frames = []
    video_number = 1

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_counter += 1
        # if frame_counter % 10 != 0:
        #     frames.append(frame)
        #     continue

        height, width, _ = frame.shape
        crop_size_width = width // 4
        crop_size_height = height // 2

        cropped_frame = frame[height - crop_size_height:height, width - crop_size_width:width]
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        ret, mask = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY_INV)

        # cv2.imshow('Processed Frame', new_img)
        # cv2.waitKey(0)

        res = ocr_manager.readtext(new_img)

        if not res:
            continue

        letter = res[0][-2].upper()

        if current_letter != letter and res[0][-1] > 0.5:
            print(f"Changed letter! {current_letter=} {letter}")
            if frames:
                save_video(frames, f'{current_letter}/11.mp4', fps)
                video_number += 1
                frames = []
            current_letter = letter
        frames.append(frame)

    save_video(frames, f'Ż/11.mp4', fps)

    cap.release()


def save_video(frames, output_video, fps):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if os.path.exists(PATH_OUT + output_video):
        raise FileExistsError

    if not os.path.exists(PATH_OUT + output_video[:output_video.find("/")]):
        os.mkdir(PATH_OUT + output_video[:output_video.find("/")])

    out = cv2.VideoWriter(PATH_OUT + output_video, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


# Użycie funkcji
process_video('10.mp4')