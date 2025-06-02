import cv2 as cv
import numpy as np
from ultralytics import YOLO
from PIL import Image


def process_video(path):
    cap = cv.VideoCapture(path)
    count = 500
    prev_frame = None
    model = YOLO("../yolo11n-seg.pt")

    width = 1920
    height = 1080
    count_inc = 280
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv.resize(frame, (width, height))
            count += count_inc
            cap.set(cv.CAP_PROP_POS_FRAMES, count)

            if prev_frame is None:
                prev_frame = frame

            frames.append(frame)
        else:
            cap.release()
            break

    results = model.predict(frames)

    masks = []
    for result in results:
        if result.masks is not None and result.masks.data.shape[0] > 0:
            masks.append(result.masks)
        else:
            masks.append(None)

    count = 500
    count_inc = 280
    a = 0.5

    getv1 = 0
    getv2 = 0
    y = 1
    cur_max = 0
    frame_save = []
    delta = 400
    for i, maskobj in enumerate(masks):
        if maskobj is not None:
            mask = maskobj.data[0].cpu().numpy().astype(np.uint8)
            mask = cv.dilate(mask, kernel=np.ones((17, 17), dtype=np.uint8))
            mask = cv.resize(mask, (width, height))
            mask_inv = cv.bitwise_not(mask) // 255

            add = cv.bitwise_and(prev_frame, prev_frame, mask=mask)
            bg = cv.bitwise_and(frames[i], frames[i], mask=mask_inv)
            res = cv.bitwise_or(bg, add)
        else:
            res = frames[i]

        prev_frame = res

        if i == 0:
            getv2 = get_value(res)
            y = getv2
        else:
            getv1, getv2 = getv2, get_value(res)
            y = (1 - a) * getv1 + a * getv2

        if y > cur_max:
            cur_max = y

        if cur_max - y > delta:
            frame_save.append(res)
            cur_max = 0


        count += count_inc


    frame_save = [Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)) for img in frame_save]
    frame_save[0].save('static/save1.pdf', "PDF", resolution=100.0, save_all=True, append_images=frame_save[1:])


def get_value(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 50, 50)
    canny = cv.dilate(canny, kernel=np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    return len(contours)



if __name__ == "__main__":
    path = 'lecture.mp4'
    process_video(path)
