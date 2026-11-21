"""
live_test_oakd.py  -- trained model ko OAK-D pe live chalana
=============================================================
Peanut dekhta hai -> crop -> model -> GOOD / BAD label screen pe.

Zaroori:
  - 'peanut.weights.h5' isi folder mein ho (Drive se download kiya hua).
  - tensorflow + depthai + opencv installed ho.

Keys:  q = quit
"""

import os
import cv2
import numpy as np
import depthai as dai
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

WEIGHTS = "peanut.weights.h5"
IMG = 224
TOP, BOT = 0.24, 0.80
MARGIN = 0.25
MIN_AREA = 2000
START_FOCUS = 130
THRESH = 0.40          # P(good) > THRESH => GOOD. Neeche = good aasaan, Upar = good sakht
LOCK_AFTER = 30        # itne frames baad exposure+WB lock (light settle hone ke baad)

# label mapping (training alphabetical): bad=0, good=1
# sigmoid output = P(good).  > THRESH => GOOD,  warna BAD


# ---------- SAME crop jaise preprocessing mein tha (domain gap na ho) ----------
def crop_peanut(im):
    h, w = im.shape[:2]
    y1, y2 = int(h * TOP), int(h * BOT)
    belt = im[y1:y2, :]
    gray = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((9, 9), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, None
    x, y, bw, bh = cv2.boundingRect(c)
    cx, cy = x + bw / 2, y + bh / 2
    side = int(max(bw, bh) * (1 + MARGIN))
    bx1 = max(0, int(cx - side / 2)); by1 = max(0, int(cy - side / 2))
    bx2 = min(belt.shape[1], int(cx + side / 2)); by2 = min(belt.shape[0], int(cy + side / 2))
    crop = belt[by1:by2, bx1:bx2]
    if crop.size == 0:
        return None, None
    box_on_belt = (y1, bx1, by1, bx2, by2)   # overlay dikhane ke liye
    return cv2.resize(crop, (IMG, IMG)), box_on_belt


# ---------- SAME architecture jaise training mein tha ----------
def build_model():
    augment = tf.keras.Sequential([
        layers.RandomRotation(0.3),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ])
    base = MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights=None)
    inputs = layers.Input(shape=(IMG, IMG, 3))
    x = augment(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs)


print("Model ban raha hai + weights load ho rahe hain...")
model = build_model()
model.load_weights(WEIGHTS)
print("Model ready.\n")

# ---------- OAK-D ----------
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setVideoSize(1920, 1080)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setFps(30)
xout = pipeline.create(dai.node.XLinkOut); xout.setStreamName("rgb")
cam.video.link(xout.input)
ctrl_in = pipeline.create(dai.node.XLinkIn); ctrl_in.setStreamName("control")
ctrl_in.out.link(cam.inputControl)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_ctrl = device.getInputQueue("control")
    c = dai.CameraControl(); c.setManualFocus(START_FOCUS); q_ctrl.send(c)
    print("Live chal raha hai. q = quit")

    frame_i = 0
    prob_buf = deque(maxlen=7)   # last 7 frames average -> flicker kam
    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb is None:
            continue
        frame = in_rgb.getCvFrame()
        frame_i += 1

        # light settle hone ke baad exposure + white-balance LOCK
        # (taake live images training jaisi consistent rahein)
        if frame_i == LOCK_AFTER:
            lk = dai.CameraControl()
            lk.setAutoExposureLock(True)
            lk.setAutoWhiteBalanceLock(True)
            q_ctrl.send(lk)
            print("exposure + white balance LOCKED")

        crop, box = crop_peanut(frame)

        preview = cv2.resize(frame, (960, 540))
        if crop is not None:
            arr = np.expand_dims(crop.astype("float32"), 0)   # 0-255 (preprocess model ke andar)
            prob = float(model.predict(arr, verbose=0)[0][0])  # P(good)
            prob_buf.append(prob)
            prob = sum(prob_buf) / len(prob_buf)               # smoothed (last 7 frames)
            if prob > THRESH:
                label, color, conf = "GOOD", (0, 255, 0), prob
            else:
                label, color, conf = "BAD", (0, 0, 255), 1 - prob
            cv2.putText(preview, f"{label}  {conf*100:.1f}%", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
            # box draw (belt strip ke coords ko preview scale pe laana)
            y1, bx1, by1, bx2, by2 = box
            sx, sy = 960 / frame.shape[1], 540 / frame.shape[0]
            cv2.rectangle(preview,
                          (int(bx1 * sx), int((y1 + by1) * sy)),
                          (int(bx2 * sx), int((y1 + by2) * sy)), color, 2)
        else:
            prob_buf.clear()
            cv2.putText(preview, "koi peanut nahi", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Peanut live test", preview)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

cv2.destroyAllWindows()
