"""
continuous_sort.py  -- poora system: camera classify -> Arduino serial -> servo sort
=====================================================================================
Belt lagaatar chalti (Arduino continuous_sort.ino). Yeh script har NAYE peanut ko
ek baar classify karta aur 'B'(bad)/'G'(good) Arduino ko bhejta.

Zaroori: pip install depthai opencv-python==4.10.0.84 tensorflow pyserial
Files folder mein: continuous_sort.py + peanut.weights.h5
"""

import cv2, numpy as np, time
import depthai as dai
import serial
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- SETTINGS ----------------
WEIGHTS   = "peanut.weights.h5"
COM_PORT  = "COM11"      # Arduino ka port (IDE ke neeche dikhta) - apna daalo
BAUD      = 9600
THRESH    = 0.40         # P(good) > THRESH => GOOD
IMG       = 224
TOP, BOT  = 0.24, 0.80
MARGIN    = 0.25
MIN_AREA  = 2000
START_FOCUS = 130
LOCK_AFTER  = 30
DECIDE_FRAMES = 5        # naye peanut pe itne frame average kar ke decide
# bad=0, good=1 ; sigmoid = P(good)
# ------------------------------------------

def crop_peanut(im):
    h, w = im.shape[:2]
    y1, y2 = int(h*TOP), int(h*BOT)
    belt = im[y1:y2, :]
    gray = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)
    _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((9,9),np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25,25),np.uint8))
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA: return None, None
    x,y,bw,bh = cv2.boundingRect(c)
    cx, cy = x+bw/2, y+bh/2
    side = int(max(bw,bh)*(1+MARGIN))
    bx1=max(0,int(cx-side/2)); by1=max(0,int(cy-side/2))
    bx2=min(belt.shape[1],int(cx+side/2)); by2=min(belt.shape[0],int(cy+side/2))
    crop = belt[by1:by2, bx1:bx2]
    if crop.size==0: return None, None
    return cv2.resize(crop,(IMG,IMG)), (y1,bx1,by1,bx2,by2)

def build_model():
    augment = tf.keras.Sequential([
        layers.RandomRotation(0.3), layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomZoom(0.2), layers.RandomBrightness(0.15), layers.RandomContrast(0.15)])
    base = MobileNetV2(input_shape=(IMG,IMG,3), include_top=False, weights=None)
    inp = layers.Input(shape=(IMG,IMG,3))
    x = augment(inp); x = preprocess_input(x); x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)

print("Model load ho raha...")
model = build_model(); model.load_weights(WEIGHTS); print("Model ready.")

# serial connect
try:
    ard = serial.Serial(COM_PORT, BAUD, timeout=1); time.sleep(2)
    print("Arduino connected:", COM_PORT)
except Exception as e:
    ard = None; print("WARNING: Arduino connect nahi hua:", e, "(sirf dekhne ke liye chalega)")

# OAK-D
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setVideoSize(1920,1080); cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR); cam.setFps(30)
xout = pipeline.create(dai.node.XLinkOut); xout.setStreamName("rgb"); cam.video.link(xout.input)
cin = pipeline.create(dai.node.XLinkIn); cin.setStreamName("control"); cin.out.link(cam.inputControl)

with dai.Device(pipeline) as dev:
    q = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
    qc = dev.getInputQueue("control")
    c=dai.CameraControl(); c.setManualFocus(START_FOCUS); qc.send(c)
    print("Chal raha hai. q = quit")

    fi=0; peanut_present=False; probs=[]; last_label=""
    while True:
        inr = q.tryGet()
        if inr is None: continue
        frame = inr.getCvFrame(); fi+=1
        if fi==LOCK_AFTER:
            lk=dai.CameraControl(); lk.setAutoExposureLock(True); lk.setAutoWhiteBalanceLock(True); qc.send(lk)

        crop, box = crop_peanut(frame)
        preview = cv2.resize(frame,(960,540))

        if crop is not None:
            arr = np.expand_dims(crop.astype("float32"),0)
            prob = float(model.predict(arr, verbose=0)[0][0])
            if not peanut_present:
                probs.append(prob)
                if len(probs) >= DECIDE_FRAMES:
                    avg = sum(probs)/len(probs)
                    if avg > THRESH:
                        last_label="GOOD"
                        if ard: ard.write(b'G')
                    else:
                        last_label="BAD"
                        if ard: ard.write(b'B')
                    print("Decision:", last_label, f"({avg:.2f})")
                    peanut_present=True; probs=[]
            # box
            y1,bx1,by1,bx2,by2 = box
            sx,sy = 960/frame.shape[1], 540/frame.shape[0]
            col = (0,255,0) if last_label=="GOOD" else (0,0,255)
            cv2.rectangle(preview,(int(bx1*sx),int((y1+by1)*sy)),(int(bx2*sx),int((y1+by2)*sy)),col,2)
            if last_label:
                cv2.putText(preview,last_label,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,col,2)
        else:
            peanut_present=False; probs=[]
            cv2.putText(preview,"waiting...",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        cv2.imshow("Continuous Sorter", preview)
        if (cv2.waitKey(1)&0xFF)==ord('q'): break

cv2.destroyAllWindows()
if ard: ard.close()
