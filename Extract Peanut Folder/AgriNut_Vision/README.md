# AgriNut Vision — AI-Based Peanut Quality Sorting

**AI Innovation Challenge 2026 — AI Builders League**
**Team:** AgriNut Vision (Muhammad Haroon Nadeem, Muhammad Bilal, Zubia Tanoli)
**Reference:** AIC26-0BA946E7

---

## What it is

A low-cost machine that automatically grades and sorts in-shell peanuts into **good** and **bad** quality for smallholder farmers. A camera captures each peanut on a conveyor belt, a computer-vision model (MobileNetV2 CNN) classifies it as good or bad, and an Arduino-controlled conveyor + servo/air-jet physically separates the bad ones.

**Pipeline:** Camera → crop peanut → MobileNetV2 classifier (Good/Bad) → Arduino → conveyor + servo sorting.

---

## Project structure

```
AgriNut_Vision/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── capture_oakd.py         # capture dataset from OAK-D camera (keys: g=good, b=bad)
│   ├── preprocess_dataset.py   # auto-crop peanut from belt, resize to 224x224
│   ├── colab_train_peanut.py   # MobileNetV2 training (run in Google Colab, GPU)
│   ├── live_test_oakd.py       # live inference: crop -> model -> GOOD/BAD on screen
│   └── continuous_sort.py      # full system: classify -> serial -> Arduino servo sort
├── arduino/
│   ├── belt_only.ino           # belt motor test
│   ├── servo_motor_test.ino    # belt + servo sweep test
│   └── continuous_sort.ino     # belt runs + servo pushes bad peanut on serial 'B'
└── docs/
    └── AgriNut_Vision_Master_Literature.csv   # reviewed literature + patent
```

---

## Setup

1. Install Python 3.11+.
2. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set your values (e.g. Arduino COM port). No real secrets are required.

---

## How to run

**Train (Google Colab, GPU):** upload `dataset_processed.zip` to Google Drive, then run the cells in `src/colab_train_peanut.py`. It saves `peanut.weights.h5`.

**Capture a dataset (local, OAK-D):**
```
python src/capture_oakd.py
```
Keys: `g` = save good, `b` = save bad, `[` `]` = focus, `l` = lock exposure, `q` = quit.

**Preprocess (crop + resize):**
```
python src/preprocess_dataset.py
```

**Live classification (OAK-D):** place `peanut.weights.h5` next to the script, then:
```
python src/live_test_oakd.py
```

**Full sorting system (camera + Arduino):** upload `arduino/continuous_sort.ino` to the Arduino, set the COM port in `src/continuous_sort.py`, then:
```
python src/continuous_sort.py
```

---

## Model

- **Architecture:** MobileNetV2 (transfer learning) + GlobalAveragePooling + Dropout(0.4) + Dense(1, sigmoid).
- **Input:** 224×224 RGB crop of a single peanut. **Output:** P(good); threshold 0.40.
- **Labels:** bad = 0, good = 1.
- **Results:** ~90% accuracy on a held-out test set; bad-peanut recall 0.91; dataset ~670 labelled images (self-collected).

## Hardware

OAK-D camera · Arduino Uno · L298N motor driver · 12V DC motor (conveyor) · MG995 servo · IR sensor · LM2596 buck converter (5–6V for servo) · controlled-light box.

> Note: trained weights (`peanut.weights.h5`) and the image dataset are not bundled here to keep the ZIP small; the dataset link is provided in the submission form.
