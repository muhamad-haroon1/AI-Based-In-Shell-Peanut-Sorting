"""
capture_oakd.py  -- OAK-D se peanut dataset banane ki script
=============================================================
Keys (camera window pe focus hona chahiye):
    g  -> current frame ko GOOD folder mein save karo
    b  -> current frame ko BAD  folder mein save karo
    [  -> focus thoda kam (near)   ]  -> focus thoda zyada (far)
    l  -> auto-exposure + white-balance LOCK karo (light set karne ke baad dabao)
    u  -> lock hatao (unlock) agar dobara adjust karna ho
    q  -> band karo

Folder khud ban jayenge: dataset/good/  aur  dataset/bad/
"""

import os
import cv2
import depthai as dai

# ---------- SETTINGS (zaroorat ho to badlo) ----------
SAVE_DIR   = "dataset"          # yahan good/ aur bad/ banenge
START_FOCUS = 130               # 0..255  (box ki height ke hisab se [ ] se tune karo)
SAVE_WIDTH  = 1920              # save hone wali image ki resolution (full 1080p)
SAVE_HEIGHT = 1080
# -----------------------------------------------------

good_dir = os.path.join(SAVE_DIR, "good")
bad_dir  = os.path.join(SAVE_DIR, "bad")
os.makedirs(good_dir, exist_ok=True)
os.makedirs(bad_dir,  exist_ok=True)

# agar folder mein pehle se images hain to counter wahin se aage chale
good_count = len([f for f in os.listdir(good_dir) if f.endswith(".jpg")])
bad_count  = len([f for f in os.listdir(bad_dir)  if f.endswith(".jpg")])

# ---------- OAK-D pipeline ----------
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)     # RGB camera
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setVideoSize(SAVE_WIDTH, SAVE_HEIGHT)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setFps(30)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.video.link(xout.input)

# control channel (focus / exposure lock bhejne ke liye)
ctrl_in = pipeline.create(dai.node.XLinkIn)
ctrl_in.setStreamName("control")
ctrl_in.out.link(cam.inputControl)

# ---------- run ----------
focus = START_FOCUS
locked = False

with dai.Device(pipeline) as device:
    q_rgb  = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_ctrl = device.getInputQueue("control")

    # shuru mein manual focus set karo (box fixed hai, hunting nahi chahiye)
    c = dai.CameraControl()
    c.setManualFocus(focus)
    q_ctrl.send(c)

    print("OAK-D chal raha hai. Window pe: g=good  b=bad  [ ]=focus  l=lock  u=unlock  q=quit")

    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb is None:
            continue
        frame = in_rgb.getCvFrame()

        # screen pe chhota preview (asli full-res save hoti hai)
        preview = cv2.resize(frame, (960, 540))
        status = f"GOOD:{good_count}  BAD:{bad_count}  focus:{focus}  " + ("LOCKED" if locked else "auto")
        cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(preview, "g=good  b=bad  [ ]=focus  l=lock  u=unlock  q=quit",
                    (10, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("OAK-D capture", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('g'):
            path = os.path.join(good_dir, f"good_{good_count:04d}.jpg")
            cv2.imwrite(path, frame)
            good_count += 1
            print("saved", path)

        elif key == ord('b'):
            path = os.path.join(bad_dir, f"bad_{bad_count:04d}.jpg")
            cv2.imwrite(path, frame)
            bad_count += 1
            print("saved", path)

        elif key == ord('['):
            focus = max(0, focus - 5)
            c = dai.CameraControl(); c.setManualFocus(focus); q_ctrl.send(c)

        elif key == ord(']'):
            focus = min(255, focus + 5)
            c = dai.CameraControl(); c.setManualFocus(focus); q_ctrl.send(c)

        elif key == ord('l'):
            c = dai.CameraControl()
            c.setAutoExposureLock(True)
            c.setAutoWhiteBalanceLock(True)
            q_ctrl.send(c)
            locked = True
            print("exposure + white balance LOCKED")

        elif key == ord('u'):
            c = dai.CameraControl()
            c.setAutoExposureLock(False)
            c.setAutoWhiteBalanceLock(False)
            q_ctrl.send(c)
            locked = False
            print("unlocked")

cv2.destroyAllWindows()
print(f"\nDONE. good={good_count}  bad={bad_count}  (folder: {SAVE_DIR}/)")
