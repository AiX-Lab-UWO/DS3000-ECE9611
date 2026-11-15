#!/usr/bin/env python3
"""
Live face detection demo for the classroom.

Opens your laptop camera and runs face detection in real time. You can switch
between a classic detector and a deep-learning one, compare them on the same
frame, and step through how the classic sliding-window search works.

  [1] Haar cascade  (Viola and Jones, CVPR 2001; this model by Lienhart et al.)
  [2] YuNet         (Wu, Peng and Yu, Machine Intelligence Research 2023)
  [3] Both at once

Setup (Python 3.9 or newer):
    pip install opencv-contrib-python

    Use the "contrib" package. From OpenCV 5 on, the plain opencv-python
    package no longer includes the Haar cascade detector.

Run:
    python face_detection_demo.py                     # built-in camera
    python face_detection_demo.py --source 1          # external / USB camera
    python face_detection_demo.py --source group.jpg  # a photo or video file

Model files: YuNet comes in two versions, and the script picks the one your
OpenCV can run. Keep them in the "models" folder next to this script:
    face_detection_yunet_2023mar.onnx   for OpenCV 4.8 and newer
    face_detection_yunet_2022mar.onnx   for OpenCV 4.5.4 to 4.7 (e.g. 4.6.0)
If a model file is missing, the script tries to download it into a "models"
folder next to this script.

Controls: the menu on the left of the window lists every action. Click a row
with the mouse, or press its key on the keyboard (click the camera window once
first so it receives the key presses).
    1 / 2 / 3   Haar cascade / YuNet / compare both
    w           Step-by-step walkthrough of the selected detector
                (1: Haar sliding window, 2: inside the YuNet CNN, 3: both in turn)
    + / -       Stricter / looser detection threshold
    i           Show the image the detector actually receives
    b           Pixelate detected faces (privacy)
    m           Mirror the camera image
    f           Fullscreen on / off
    space       Freeze the frame (detection keeps running on it)
    s           Save a screenshot
    h           Show / hide help
    q or Esc    Quit
"""

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------
HAAR_FILE = "haarcascade_frontalface_default.xml"
HAAR_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + HAAR_FILE,
]
YUNET_FILE = "face_detection_yunet_2023mar.onnx"        # needs OpenCV 4.8 or newer
YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/" + YUNET_FILE,
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE,
]
# The older YuNet export that OpenCV 4.5.4 to 4.7 run (the file OpenCV 4.6 uses in its own tests)
YUNET_OLD_FILE = "face_detection_yunet_2022mar.onnx"
YUNET_OLD_URLS = [
    "https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/"
    "1688402dbd9b9fc4a3a6793810f558b7407ff384/tasks/task1/onnx/yunet_120x160.onnx",
]

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:   # code pasted into a notebook cell
    SCRIPT_DIR = Path.cwd()

WINDOW = "Face detection demo"
FONT = cv2.FONT_HERSHEY_SIMPLEX
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Colours are BGR
WHITE = (255, 255, 255)
GREY = (175, 175, 175)
YELLOW = (0, 220, 255)
GREEN = (80, 230, 80)
DARK_GREEN = (40, 120, 40)
HAAR_COLOR = (255, 160, 40)   # blue
YUNET_COLOR = (80, 230, 80)   # green
# YuNet landmark order: right eye, left eye, nose tip, right mouth corner, left mouth corner
LANDMARK_COLORS = [(255, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255), (255, 0, 255)]

CAMERA_HELP = """
Could not read from the camera.
  - Close other apps that may be using it (Zoom, Teams, the Camera app).
  - macOS: System Settings > Privacy & Security > Camera, and allow your
    terminal or editor (Terminal, iTerm, VS Code, PyCharm). Then restart it.
  - Windows: Settings > Privacy & security > Camera, and allow desktop apps.
  - Try another camera index:  python face_detection_demo.py --source 1
"""


# ---------------------------------------------------------------------------
# Downloading and loading models
# ---------------------------------------------------------------------------
def download(urls, dest, min_bytes):
    """Fetch a file once and keep it. Returns the path, or None if every URL failed."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for url in urls:
        try:
            print(f"Downloading {dest.name} ...")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size >= min_bytes:
                tmp.replace(dest)
                print(f"  saved to {dest}")
                return dest
            print(f"  got an unexpectedly small file from {url}")
        except Exception as exc:  # network down, SSL certificate problems, HTTP errors
            print(f"  could not download from {url}\n  ({exc})")
    if tmp.exists():
        tmp.unlink()
    print(f"  You can download it in a browser from\n    {urls[0]}\n  and save it as\n    {dest}")
    return None


def cascade_stats(xml_path):
    """Read the number of stages and weak classifiers (features) from a cascade XML."""
    try:
        text = Path(xml_path).read_text(errors="ignore")
    except OSError:
        return None, None
    m = re.search(r"<stageNum>\s*(\d+)\s*</stageNum>", text)
    stages = int(m.group(1)) if m else None
    features = text.count("<internalNodes>") or None
    return stages, features


def find_model(filename, urls, models_dir, min_bytes=100_000):
    """Use a copy next to this script, in the current folder or in models/; else download it."""
    for folder in (SCRIPT_DIR, Path.cwd(), models_dir):
        p = folder / filename
        if p.exists() and p.stat().st_size >= min_bytes:
            return p
    return download(urls, models_dir / filename, min_bytes)


def opencv_version():
    try:
        return tuple(int(part) for part in cv2.__version__.split(".")[:2])
    except ValueError:
        return (0, 0)


UPGRADE_FIX = "Fix: run  pip install --upgrade opencv-contrib-python  and start the demo again."

# Each loader returns (detector, None) or (None, [reason, fix]); the reason is shown on screen.


def load_haar(models_dir):
    if not hasattr(cv2, "CascadeClassifier"):
        return None, ["This OpenCV package has no Haar cascade detector.",
                      "Fix: pip uninstall opencv-python, then pip install opencv-contrib-python"]
    path = None
    data = getattr(cv2, "data", None)
    if data is not None and getattr(data, "haarcascades", None):
        builtin = Path(data.haarcascades) / HAAR_FILE
        if builtin.exists():
            path = builtin
    if path is None:
        path = find_model(HAAR_FILE, HAAR_URLS, models_dir)
    if path is None:
        return None, ["The Haar model file is missing and could not be downloaded.",
                      f"Fix: put {HAAR_FILE} in the models folder next to this script."]
    try:
        return HaarDetector(path), None
    except Exception as exc:
        print(f"Haar error: {exc}")
        return None, [f"OpenCV {cv2.__version__} could not load the Haar model.", UPGRADE_FIX]


def load_yunet(models_dir):
    if not hasattr(cv2, "FaceDetectorYN"):
        return None, [f"OpenCV {cv2.__version__} is too old for YuNet (needs 4.5.4 or newer).", UPGRADE_FIX]
    # OpenCV before 4.8 can only run the 2022 model; 4.8 and newer only the 2023 one
    new, old = (YUNET_FILE, YUNET_URLS), (YUNET_OLD_FILE, YUNET_OLD_URLS)
    order = [old, new] if opencv_version() < (4, 8) else [new, old]
    missing = []
    for filename, urls in order:
        path = find_model(filename, urls, models_dir)
        if path is None:
            missing.append(filename)
            continue
        try:
            det = YuNetDetector(path)
            print(f"YuNet: using {filename}")
            return det, None
        except Exception as exc:
            print(f"YuNet could not run {filename} with OpenCV {cv2.__version__}: {exc}")
    wanted = order[0][0]
    if wanted in missing:
        return None, [f"The YuNet model file for OpenCV {cv2.__version__} is missing.",
                      f"Fix: put {wanted} in the models folder next to this script."]
    return None, [f"OpenCV {cv2.__version__} could not run the YuNet model.", UPGRADE_FIX]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
class Face:
    def __init__(self, box, score=None, landmarks=None):
        self.box = box              # (x, y, w, h) in pixels
        self.score = score          # confidence in [0, 1], or None
        self.landmarks = landmarks  # 5 x 2 array, or None

    def scaled(self, k):
        x, y, w, h = self.box
        box = (int(round(x * k)), int(round(y * k)), int(round(w * k)), int(round(h * k)))
        lm = None if self.landmarks is None else self.landmarks * k
        return Face(box, self.score, lm)


class HaarDetector:
    key = 1
    name = "Haar cascade"
    short = "Haar"
    about = "Viola-Jones 2001: hand-made light/dark features, boosted cascade"
    color = HAAR_COLOR

    def __init__(self, xml_path):
        self.cascade = cv2.CascadeClassifier(str(xml_path))
        if self.cascade.empty():
            raise RuntimeError(f"could not read {xml_path}")
        self.stages, self.features = cascade_stats(xml_path)
        self.min_neighbors = 5   # how many overlapping hits a face needs to be kept

    @staticmethod
    def prepare(bgr):
        """The cascade works on brightness only, with the contrast evened out."""
        return cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

    def model_input(self, bgr):
        return cv2.cvtColor(self.prepare(bgr), cv2.COLOR_GRAY2BGR), "grey, equalised"

    def detect(self, bgr):
        gray = self.prepare(bgr)
        side = max(24, int(0.06 * min(gray.shape[:2])))
        rects = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=self.min_neighbors, minSize=(side, side))
        return [Face(tuple(int(v) for v in r)) for r in rects]

    def threshold_text(self):
        return f"min neighbours {self.min_neighbors}"

    def adjust(self, step):
        self.min_neighbors = int(np.clip(self.min_neighbors + step, 0, 40))


class YuNetDetector:
    key = 2
    name = "YuNet CNN"
    short = "YuNet"
    about = "Wu et al. 2023: small CNN, learned features, 5 face landmarks"
    color = YUNET_COLOR

    def __init__(self, onnx_path):
        self.path = Path(onnx_path)
        self.threshold = 0.6
        self.size = (320, 320)
        self.model = cv2.FaceDetectorYN.create(str(onnx_path), "", self.size,
                                               self.threshold, 0.3, 5000)
        self.model.detect(np.zeros((320, 320, 3), np.uint8))  # fail early if unsupported

    def model_input(self, bgr):
        return bgr, "colour"

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        if (w, h) != self.size:
            self.size = (w, h)
            self.model.setInputSize(self.size)
        _, faces = self.model.detect(bgr)
        out = []
        if faces is not None:
            for f in faces:
                box = tuple(int(round(v)) for v in f[:4])
                out.append(Face(box, float(f[14]), f[4:14].reshape(5, 2).copy()))
        return out

    def threshold_text(self):
        return f"score threshold {self.threshold:.2f}"

    def adjust(self, step):
        self.threshold = float(np.clip(round(self.threshold + 0.05 * step, 2), 0.05, 0.95))
        self.model.setScoreThreshold(self.threshold)


def run_detector(det, frame, detect_width):
    """Shrink the frame, detect, and map the boxes back to full resolution."""
    h, w = frame.shape[:2]
    k = min(1.0, detect_width / w)
    small = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
    t0 = time.perf_counter()
    faces = det.detect(small)
    ms = (time.perf_counter() - t0) * 1000
    return [f.scaled(1 / k) for f in faces], ms, small


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def text_scale(img):
    h, w = img.shape[:2]
    return float(np.clip(0.65 * min(w / 1280, h / 720), 0.4, 1.3))


def thickness(scale):
    return max(1, int(round(scale * 2.4)))


def shade(img, x0, y0, x1, y1, alpha=0.6):
    """Darken a rectangle so text on top of it stays readable."""
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def _norm(lines):
    return [(l, WHITE) if isinstance(l, str) else l for l in lines]


def panel_size(lines, scale):
    lines = _norm(lines)
    th = thickness(scale)
    pad = max(6, int(14 * scale))
    line_h = int(cv2.getTextSize("Ag", FONT, scale, th)[0][1] * 2.0)
    width = max(cv2.getTextSize(t, FONT, scale, th)[0][0] for t, _ in lines) + 2 * pad
    return width, line_h * len(lines) + pad


def draw_panel(img, lines, x, y, scale, min_width=0, alpha=0.6):
    """Draw lines of text on a dark box. Returns the box's bottom-right corner."""
    lines = _norm(lines)
    th = thickness(scale)
    pad = max(6, int(14 * scale))
    line_h = int(cv2.getTextSize("Ag", FONT, scale, th)[0][1] * 2.0)
    w, h = panel_size(lines, scale)
    w = max(w, min_width)
    shade(img, x, y, x + w, y + h, alpha)
    for i, (text, color) in enumerate(lines):
        base = y + pad // 2 + line_h * i + int(line_h * 0.72)
        cv2.putText(img, text, (x + pad, base), FONT, scale, color, th, cv2.LINE_AA)
    return x + w, y + h


def put_tag(img, text, x, y, color, scale, below=False):
    """A filled label attached to the top (or bottom) edge of a box."""
    th = thickness(scale)
    (tw, tht), base = cv2.getTextSize(text, FONT, scale, th)
    h = tht + base + 6
    y0 = y if below else y - h
    y0 = int(np.clip(y0, 0, img.shape[0] - h))
    x = int(np.clip(x, 0, img.shape[1] - tw - 8))
    cv2.rectangle(img, (x, y0), (x + tw + 8, y0 + h), color, -1)
    cv2.putText(img, text, (x + 4, y0 + tht + 3), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)


def draw_face(img, face, color, lw, landmarks=True):
    x, y, w, h = face.box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, lw, cv2.LINE_AA)
    if landmarks and face.landmarks is not None:
        r = max(3, lw + 1)
        for (lx, ly), c in zip(face.landmarks, LANDMARK_COLORS):
            cv2.circle(img, (int(lx), int(ly)), r, c, -1, cv2.LINE_AA)


def expand_box(box, frac):
    x, y, w, h = box
    dx, dy = int(w * frac / 2), int(h * frac / 2)
    return x - dx, y - dy, w + 2 * dx, h + 2 * dy


def pixelate(img, box, blocks=9):
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    roi = img[y0:y1, x0:x1]
    bh = max(1, int(blocks * (y1 - y0) / (x1 - x0)))
    tiny = cv2.resize(roi, (blocks, bh), interpolation=cv2.INTER_AREA)
    img[y0:y1, x0:x1] = cv2.resize(tiny, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Sliding-window walkthrough (Haar cascade)
# ---------------------------------------------------------------------------
def walkthrough(frame, haar, is_closed, was_clicked=lambda: False):
    """Freeze the frame and show, step by step, how the cascade finds faces.
    A key press or a mouse click moves to the next step; q goes back to live view."""
    h0, w0 = frame.shape[:2]
    k = min(1.0, 960 / w0)   # work at <= 960 px wide so it stays quick
    img = cv2.resize(frame, (int(w0 * k), int(h0 * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame.copy()
    H, W = img.shape[:2]
    ts = text_scale(img)
    gray = haar.prepare(img)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    stages = f"all {haar.stages} stages" if haar.stages else "every stage"
    nav = ("Any key or click: next step     q: back to live view", GREY)

    def show_and_wait(view, lines):
        v = view.copy()
        draw_panel(v, lines, 0, 0, ts, min_width=W)
        cv2.imshow(WINDOW, v)
        was_clicked()   # ignore clicks made before this step appeared
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27) or is_closed():
                return False
            if key != 255 or was_clicked():
                return True

    # Step 1: what the cascade sees
    if not show_and_wait(gray_bgr, [
            ("Classic detector (Haar), step 1 of 5: what the detector sees", YELLOW),
            "Colour is dropped. The cascade only compares brightness between image regions,",
            "after histogram equalisation evens out the lighting.",
            nav]):
        return False

    # Step 2: animate the window sweeping over positions and sizes
    min_side = min(H, W)
    sizes, sz = [], max(24, int(0.15 * min_side))
    while sz <= int(0.9 * min_side):
        sizes.append(sz)
        sz = int(sz * 1.5)
    all_hits, skip = [], False
    for si, sz in enumerate(sizes):
        hi = int(sz * 1.5) - 1
        # The real cascade result for this band of window sizes (every position, ~1-2 px steps)
        hits = haar.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=0,
                                             minSize=(sz, sz), maxSize=(hi, hi))
        hits = [tuple(int(v) for v in r) for r in hits]
        if not skip:
            # The animation moves in big jumps; hits are revealed as the window passes them
            stride = max(sz // 2, int(np.sqrt(W * H / 70)))
            ys, xs = range(0, H - sz + 1, stride), range(0, W - sz + 1, stride)
            delay = max(1, int(1100 / max(1, len(ys) * len(xs))))
            for y in ys:
                for x in xs:
                    shown = [r for r in hits if r[1] < y or (r[1] < y + stride and r[0] < x + stride)]
                    v = gray_bgr.copy()
                    draw_panel(v, [
                        ("Classic detector (Haar), step 2 of 5: slide a window over every position, at many sizes", YELLOW),
                        f"Window {sz} x {sz} px (size {si + 1} of {len(sizes)}). "
                        "Real step is 1-2 px; shown here in big jumps.",
                        f"Green = window passed {stages} of the cascade.   "
                        f"Hits so far: {len(all_hits) + len(shown)}",
                        ("Any key or click: skip animation     q: back to live view", GREY)],
                        0, 0, ts, min_width=W, alpha=0.5)
                    # Boxes go on top of the caption so the window stays visible on the top rows
                    for r in all_hits:
                        cv2.rectangle(v, r[:2], (r[0] + r[2], r[1] + r[3]), DARK_GREEN, 1)
                    for r in shown:
                        cv2.rectangle(v, r[:2], (r[0] + r[2], r[1] + r[3]), GREEN, 2)
                    cv2.rectangle(v, (x, y), (x + sz, y + sz), YELLOW, 3)
                    _draw_pyramid_inset(v, gray_bgr, x, y, sz, ts)
                    cv2.imshow(WINDOW, v)
                    key = cv2.waitKey(delay) & 0xFF
                    if key in (ord("q"), 27) or is_closed():
                        return False
                    if key != 255 or was_clicked():
                        skip = True
                        break
                if skip:
                    break
        all_hits.extend(hits)

    # Step 3: every raw hit
    v = gray_bgr.copy()
    for r in all_hits:
        cv2.rectangle(v, r[:2], (r[0] + r[2], r[1] + r[3]), GREEN, 1)
    if not show_and_wait(v, [
            ("Classic detector (Haar), step 3 of 5: all raw hits", YELLOW),
            f"{len(all_hits)} windows passed the cascade. A real face fires many overlapping",
            "windows at nearby positions and sizes. A false alarm usually fires only a few.",
            nav]):
        return False

    # Step 4: group overlapping hits (the minNeighbors parameter)
    mn = haar.min_neighbors
    if all_hits:
        groups, weights = cv2.groupRectangles([list(r) for r in all_hits], mn, 0.2)
        groups = [tuple(int(v) for v in g) for g in groups]
        weights = [int(wt) for wt in np.ravel(weights)] if len(weights) else [1] * len(groups)
    else:
        groups, weights = [], []
    v = (img * 0.75).astype(np.uint8)
    for r in all_hits:
        cv2.rectangle(v, r[:2], (r[0] + r[2], r[1] + r[3]), DARK_GREEN, 1)
    lw = max(2, H // 200)
    for g, wt in zip(groups, weights):
        cv2.rectangle(v, g[:2], (g[0] + g[2], g[1] + g[3]), HAAR_COLOR, lw, cv2.LINE_AA)
        put_tag(v, f"face ({wt} hits)", g[0], g[1], HAAR_COLOR, ts * 0.9)
    rule = f"Groups with more than {mn} members are kept" if mn > 0 else "min neighbours is 0, so nothing is merged"
    if not show_and_wait(v, [
            ("Classic detector (Haar), step 4 of 5: merge overlapping hits", YELLOW),
            f"Overlapping windows are grouped. {rule}",
            f"(min neighbours = {mn}; change it with + / - in the live view).   "
            f"Result: {len(groups)} face{'s' if len(groups) != 1 else ''}",
            nav]):
        return False

    # Step 5: the kind of features each stage tests
    v = (img * 0.35).astype(np.uint8)
    lines = [("Classic detector (Haar), step 5 of 5: why it works", YELLOW)]
    if groups:
        _draw_feature_examples(v, img, gray, max(groups, key=lambda r: r[2] * r[3]), ts)
        n_feat = f"{haar.features:,}" if haar.features else "thousands"
        lines += [
            "Each stage adds up simple light-minus-dark rectangle tests like these two.",
            f"This model uses {n_feat} of them over {haar.stages or 'many'} stages. Most windows are "
            "rejected in the first few",
            "stages, so very little work is spent on background. That is what made it real-time in 2001."]
    else:
        lines += ["No face was found in this frame, so there is nothing to illustrate.",
                  "Face the camera and press w again."]
    lines.append(("Any key or click: back to live view", GREY))
    return show_and_wait(v, lines)


def _draw_pyramid_inset(v, gray_bgr, x, y, sz, ts):
    """Show the equivalent view: a fixed 24x24 window on a shrunken image."""
    H, W = v.shape[:2]
    r = 24.0 / sz
    pw, ph = max(1, int(W * r)), max(1, int(H * r))
    small = cv2.resize(gray_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    dw = int(np.clip(pw, 0.16 * W, 0.34 * W))   # display size; nearest-neighbour keeps pixels visible
    f = dw / pw
    dh = int(ph * f)
    inset = cv2.resize(small, (dw, dh), interpolation=cv2.INTER_NEAREST)
    wx, wy, ws = int(x * r * f), int(y * r * f), int(24 * f)
    cv2.rectangle(inset, (wx, wy), (wx + ws, wy + ws), YELLOW, 2)
    caption = ["What the cascade scans: 24x24 window,", f"image shrunk to {r * 100:.0f}%"]
    cw, chh = panel_size(caption, ts * 0.8)
    pw_total = max(dw, cw)
    x0, y0 = W - pw_total - 10, H - dh - chh - 10
    draw_panel(v, caption, x0, y0, ts * 0.8, min_width=pw_total, alpha=0.75)
    v[y0 + chh:y0 + chh + dh, x0:x0 + dw] = inset
    cv2.rectangle(v, (x0, y0 + chh), (x0 + dw - 1, y0 + chh + dh - 1), WHITE, 1)


def _draw_feature_examples(v, img, gray, box, ts):
    """Two classic Haar-like features drawn on the detected face, with real values."""
    H, W = v.shape[:2]
    x, y, w, h = box

    def region(rx0, ry0, rx1, ry1):
        return x + rx0 * w, y + ry0 * h, x + rx1 * w, y + ry1 * h

    def mean(r):
        x0, y0, x1, y1 = (int(round(c)) for c in r)
        patch = gray[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        return float(patch.mean()) if patch.size else 0.0

    # Rough positions inside a Haar face box (eyes sit about 40% of the way down)
    eyes, cheeks = region(0.12, 0.30, 0.88, 0.47), region(0.12, 0.47, 0.88, 0.64)
    eye_l, bridge, eye_r = region(0.12, 0.30, 0.40, 0.47), region(0.40, 0.30, 0.60, 0.47), region(0.60, 0.30, 0.88, 0.47)
    a_dark, a_light = mean(eyes), mean(cheeks)
    b_dark, b_light = (mean(eye_l) + mean(eye_r)) / 2, mean(bridge)
    examples = [
        ([eyes], [cheeks], "Eye band is darker than cheeks",
         f"cheeks {a_light:.0f} - eyes {a_dark:.0f} = {a_light - a_dark:+.0f}"),
        ([eye_l, eye_r], [bridge], "Nose bridge is brighter than eyes",
         f"bridge {b_light:.0f} - eyes {b_dark:.0f} = {b_light - b_dark:+.0f}"),
    ]

    # Two enlarged copies of the face, one feature on each
    side = int(min(0.36 * W, 0.50 * H))
    pad = int(0.05 * side)
    crop_box = (max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad))
    crop = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    if crop.size == 0:
        return
    sx, sy = side / crop.shape[1], side / crop.shape[0]
    top = int(0.28 * H)
    for i, (darks, lights, title, value) in enumerate(examples):
        left = int(W * (0.08 if i == 0 else 0.56))
        tile = cv2.resize(crop, (side, side), interpolation=cv2.INTER_LINEAR)
        overlay = tile.copy()
        for regs, fill in ((darks, (0, 0, 0)), (lights, (255, 255, 255))):
            for (rx0, ry0, rx1, ry1) in regs:
                p0 = (int((rx0 - crop_box[0]) * sx), int((ry0 - crop_box[1]) * sy))
                p1 = (int((rx1 - crop_box[0]) * sx), int((ry1 - crop_box[1]) * sy))
                cv2.rectangle(overlay, p0, p1, fill, -1)
                cv2.rectangle(tile, p0, p1, YELLOW, 2)
        tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)
        if top + side > H or left + side > W:
            continue
        v[top:top + side, left:left + side] = tile
        draw_panel(v, [(title, YELLOW), value], left, top + side + 6, ts * 0.9, min_width=side)


# ---------------------------------------------------------------------------
# Deep learning walkthrough (YuNet)
# ---------------------------------------------------------------------------
FIRST_CONV = "onnx_node!Conv_0"     # first convolution: 16 learned 3x3 colour filters
EARLY_LAYER = "onnx_node!Relu_1"    # its output, at 1/2 resolution
DEEP_LAYER = "onnx_node!Relu_31"    # deep in the network, at 1/16 resolution
HEAT_MAP = getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET)


def _load_dnn(path):
    """Open the same ONNX file as a plain cv2.dnn network so its inner layers can be read."""
    if hasattr(cv2.dnn, "ENGINE_CLASSIC"):   # OpenCV 5: only the classic engine exposes inner layers
        try:
            return cv2.dnn.readNetFromONNX(str(path), cv2.dnn.ENGINE_CLASSIC)
        except Exception:
            pass
    return cv2.dnn.readNetFromONNX(str(path))


def _count_params(net, names):
    total = 0
    for name in names:
        j = 0
        while True:
            try:
                p = net.getParam(name, j)
            except Exception:
                break
            if p is None or p.size == 0:
                break
            total += int(p.size)
            j += 1
    return total


def _score_maps(d, PH, PW):
    """Face score of every grid cell at each stride, decoded the way FaceDetectorYN does it."""
    maps = []
    if "conf" in d:   # 2022 model: a few anchor boxes per cell, strides 8 to 64
        conf = np.clip(d["conf"].reshape(-1, 2)[:, 1], 0, 1)
        iou = np.clip(d["iou"].reshape(-1), 0, 1)
        score = np.sqrt(conf * iou)
        i0 = 0
        for s, anchors in ((8, 3), (16, 2), (32, 2), (64, 3)):
            fh, fw = PH // s, PW // s
            n = fh * fw * anchors
            if i0 + n > score.size:
                return []
            maps.append((s, score[i0:i0 + n].reshape(fh, fw, anchors).max(axis=2)))
            i0 += n
    else:             # 2023 model: one prediction per cell, strides 8 to 32
        for s in (8, 16, 32):
            if f"cls_{s}" in d and f"obj_{s}" in d:
                c = np.clip(d[f"cls_{s}"].reshape(-1), 0, 1)
                o = np.clip(d[f"obj_{s}"].reshape(-1), 0, 1)
                if c.size == (PH // s) * (PW // s) == o.size:
                    maps.append((s, np.sqrt(c * o).reshape(PH // s, PW // s)))
    return maps


def _cells_to_image(m, stride, sw, sh):
    """A grid of cells as an (sh, sw) image where every cell covers stride x stride pixels."""
    up = np.repeat(np.repeat(m, stride, axis=0), stride, axis=1)[:sh, :sw]
    if up.shape != (sh, sw):
        up = np.pad(up, ((0, sh - up.shape[0]), (0, sw - up.shape[1])))
    return up


def _colorize(m, size, vmax=None, gray=False, interp=cv2.INTER_NEAREST):
    vmax = float(m.max()) if vmax is None else vmax
    norm = np.clip(m / vmax, 0, 1) if vmax > 0 else np.zeros(m.shape, np.float32)
    u8 = cv2.resize((norm * 255).astype(np.uint8), size, interpolation=interp)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if gray else cv2.applyColorMap(u8, HEAT_MAP)


def _paste(dst, img, x, y):
    h, w = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]


def yunet_walkthrough(frame, yunet, detect_width, is_closed, was_clicked=lambda: False):
    """Freeze the frame and show what happens inside the CNN, step by step."""
    h0, w0 = frame.shape[:2]
    k = min(1.0, detect_width / w0)   # the network gets the same input as in the live view
    small = cv2.resize(frame, (int(w0 * k), int(h0 * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame.copy()
    sh, sw = small.shape[:2]
    dk = min(1.0, 960 / w0)
    disp = cv2.resize(frame, (int(w0 * dk), int(h0 * dk)), interpolation=cv2.INTER_AREA) if dk < 1 else frame.copy()
    H, W = disp.shape[:2]
    ts = text_scale(disp)
    s2d = W / sw   # network-input pixels -> display pixels
    nav = ("Any key or click: next step     q: back to live view", GREY)

    def show(view, lines):
        v = view.copy()
        draw_panel(v, lines, 0, 0, ts, min_width=W)
        cv2.imshow(WINDOW, v)
        was_clicked()   # ignore clicks made before this step appeared
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27) or is_closed():
                return False
            if key != 255 or was_clicked():
                return True

    # The final answer and the raw candidates, from the same detector the live view uses
    t0 = time.perf_counter()
    faces = yunet.detect(small)
    ms = (time.perf_counter() - t0) * 1000
    raw = []
    try:
        raw_det = cv2.FaceDetectorYN.create(str(yunet.path), "", (sw, sh), yunet.threshold, 1.0, 5000)
        _, r = raw_det.detect(small)   # NMS threshold 1.0 = keep every overlapping box
        raw = [] if r is None else [tuple(int(round(v)) for v in row[:4]) for row in r]
    except Exception as exc:
        print(f"Could not get raw candidates: {exc}")

    # Look inside the network: run the same model file through cv2.dnn and read inner layers
    inside = None
    try:
        net = _load_dnn(yunet.path)
        names = list(net.getLayerNames())
        outs = list(net.getUnconnectedOutLayersNames())
        PH, PW = -(-sh // 64) * 64, -(-sw // 64) * 64   # pad so every stride divides evenly
        pad = cv2.copyMakeBorder(small, 0, PH - sh, 0, PW - sw, cv2.BORDER_CONSTANT, value=0)
        net.setInput(cv2.dnn.blobFromImage(pad))
        wanted = [EARLY_LAYER, DEEP_LAYER] + outs
        d = dict(zip(wanted, net.forward(wanted)))
        inside = {"early": d[EARLY_LAYER][0], "deep": d[DEEP_LAYER][0],
                  "filters": net.getParam(FIRST_CONV, 0), "params": _count_params(net, names),
                  "maps": _score_maps(d, PH, PW)}
    except Exception as exc:
        print(f"Could not look inside the network, showing the short version: {exc}")

    total = 6 if inside else 3
    step = iter(range(1, total + 1))
    biggest = max(faces, key=lambda f: f.box[2] * f.box[3]) if faces else None

    # Step: the input as numbers
    lines = [(f"Deep learning (YuNet), step {next(step)} of {total}: what the network sees", YELLOW),
             "Unlike Haar, colour is kept. The image is turned into 3 grids of numbers (blue, green, red):",
             f"{sw} x {sh} pixels x 3 channels = {sw * sh * 3:,} numbers go into the network.",
             nav]
    top = panel_size(lines, ts)[1]
    view = (disp * 0.2).astype(np.uint8)
    gap = 10
    pw = (W - 4 * gap) // 3
    ph = int(pw * sh / sw)
    lab_h = panel_size(["Blue"], ts * 0.8)[1]
    y0 = top + lab_h + 12
    chans = cv2.split(small)
    if biggest is not None and biggest.landmarks is not None:
        cx, cy = (int(v) for v in biggest.landmarks[0])   # one of the eyes
    else:
        cx, cy = sw // 2, sh // 2
    rows, cols = 5, 8
    gx0, gy0 = int(np.clip(cx - cols // 2, 0, sw - cols)), int(np.clip(cy - rows // 2, 0, sh - rows))
    for i, (label, col) in enumerate((("Blue", (255, 140, 60)), ("Green", (80, 220, 80)), ("Red", (70, 70, 255)))):
        x0 = gap + i * (pw + gap)
        _paste(view, cv2.cvtColor(cv2.resize(chans[i], (pw, ph), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_GRAY2BGR), x0, y0)
        cv2.rectangle(view, (x0, y0), (x0 + pw - 1, y0 + ph - 1), col, 2)
        draw_panel(view, [(label + " channel", col)], x0, y0 - lab_h - 2, ts * 0.8)
        if i == 1:   # mark where the numbers below come from
            sc = pw / sw
            mx, my = int(x0 + (gx0 + cols / 2) * sc), int(y0 + (gy0 + rows / 2) * sc)
            cv2.rectangle(view, (mx - 7, my - 6), (mx + 7, my + 6), YELLOW, 2)
    ns = ts * 0.9
    nth = thickness(ns)
    cw = cv2.getTextSize("255", FONT, ns, nth)[0][0] + 14
    rh = int(cv2.getTextSize("0", FONT, ns, nth)[0][1] * 2.1)
    title = [("Green values inside the yellow box (0 = dark, 255 = bright):", WHITE)]
    tw, tth = panel_size(title, ns)
    gw = max(tw, cols * cw + 20)
    gx, gy = (W - gw) // 2, y0 + ph + 16
    draw_panel(view, title, gx, gy, ns, min_width=gw, alpha=0.7)
    shade(view, gx, gy + tth, gx + gw, gy + tth + rows * rh + 8, 0.7)
    block = chans[1][gy0:gy0 + rows, gx0:gx0 + cols]
    for r_ in range(block.shape[0]):
        for c_ in range(block.shape[1]):
            cv2.putText(view, str(int(block[r_, c_])), (gx + 14 + c_ * cw, gy + tth + (r_ + 1) * rh),
                        FONT, ns, (80, 220, 80), nth, cv2.LINE_AA)
    if not show(view, lines):
        return False

    if inside:
        # Step: first layer, learned filters and what they respond to
        filt = inside["filters"]                       # (16, 3, 3, 3), channels in B, G, R order
        early = inside["early"][:, :-(-sh // 2), :-(-sw // 2)]
        lines = [(f"Deep learning (YuNet), step {next(step)} of {total}: layer 1 learns its own filters", YELLOW),
                 "Each small 3x3 colour square is a filter the network learned from training photos.",
                 "Haar's rectangles were designed by hand. Each filter slides over the whole image, and",
                 f"its feature map (right) is bright where it matches. This network has {inside['params']:,} "
                 "learned weights.",
                 nav]
        top = panel_size(lines, ts)[1]
        view = (disp * 0.15).astype(np.uint8)
        n = min(16, filt.shape[0], early.shape[0])
        ncols, nrows, cg, rg = 4, -(-n // 4), 12, 8
        mh = (H - top - 16 - (nrows - 1) * rg) // nrows
        fs = min(mh, 40)
        mw = int(mh * early.shape[2] / early.shape[1])
        if ncols * (fs + 6 + mw) + (ncols - 1) * cg > W - 20:
            mw = (W - 20 - (ncols - 1) * cg) // ncols - fs - 6
            mh = int(mw * early.shape[1] / early.shape[2])
        cell_w = fs + 6 + mw
        xs0 = (W - (ncols * cell_w + (ncols - 1) * cg)) // 2
        for i in range(n):
            r_, c_ = divmod(i, ncols)
            x, y = xs0 + c_ * (cell_w + cg), top + 10 + r_ * (mh + rg)
            kern = filt[i].transpose(1, 2, 0)
            kn = (kern - kern.min()) / (kern.max() - kern.min() + 1e-9)
            _paste(view, cv2.resize((kn * 255).astype(np.uint8), (fs, fs), interpolation=cv2.INTER_NEAREST), x, y)
            cv2.rectangle(view, (x, y), (x + fs - 1, y + fs - 1), GREY, 1)
            _paste(view, _colorize(early[i], (mw, mh), gray=True, interp=cv2.INTER_AREA), x + fs + 6, y)
        if not show(view, lines):
            return False

        # Step: deep layer, channels that respond to the face
        deep = inside["deep"]
        vh, vw = min(deep.shape[1], -(-sh // 16)), min(deep.shape[2], -(-sw // 16))
        deep = deep[:, :vh, :vw]
        flat = deep.reshape(deep.shape[0], -1)
        if faces:
            mask = np.zeros((vh, vw), bool)
            for f in faces:
                x, y, w, h = f.box
                mask[max(0, y // 16):max(0, -(-(y + h) // 16)), max(0, x // 16):max(0, -(-(x + w) // 16))] = True
            if mask.any():
                inner = deep[:, mask].mean(axis=1)
                rank = inner / (flat.mean(axis=1) + 1e-6) * (inner > 0)
            else:
                rank = flat.max(axis=1)
        else:
            rank = flat.max(axis=1)
        chosen = [int(c) for c in np.argsort(-rank)[:8]]
        lines = [(f"Deep learning (YuNet), step {next(step)} of {total}: deeper layers respond to faces", YELLOW),
                 f"Deep in the network each map is only {vw} x {vh} cells, and each cell sees a large area.",
                 f"These are the 8 of its {deep.shape[0]} channels most active on the face. Nobody programmed",
                 "them to find eyes or faces; they learned it from thousands of labelled face photos.",
                 nav]
        top = panel_size(lines, ts)[1]
        view = (disp * 0.15).astype(np.uint8)
        lab_h = panel_size(["channel 0"], ts * 0.8)[1]
        tw_ = (W - 20 - 3 * 10) // 4
        th_ = int(tw_ * H / W)
        if 2 * (th_ + lab_h + 10) > H - top - 10:
            th_ = (H - top - 10) // 2 - lab_h - 10
            tw_ = int(th_ * W / H)
        xs0 = (W - (4 * tw_ + 3 * 10)) // 2
        base = cv2.resize(disp, (tw_, th_), interpolation=cv2.INTER_AREA)
        for i, ch in enumerate(chosen):
            r_, c_ = divmod(i, 4)
            x, y = xs0 + c_ * (tw_ + 10), top + 10 + r_ * (th_ + lab_h + 10)
            heat = _colorize(_cells_to_image(deep[ch], 16, sw, sh), (tw_, th_))
            _paste(view, cv2.addWeighted(base, 0.45, heat, 0.55, 0), x, y + lab_h)
            draw_panel(view, [(f"channel {ch}", WHITE)], x, y, ts * 0.8)
        if not show(view, lines):
            return False

        # Step: every grid cell gets a face score
        maps = inside["maps"]
        if maps:
            best = int(np.argmax([float(m.max()) for _, m in maps]))   # the grid where the face scored highest
            s_b, m_b = maps[best]
            view = cv2.addWeighted(disp, 0.45, _colorize(_cells_to_image(m_b, s_b, sw, sh), (W, H), vmax=1.0), 0.55, 0)
            gh, gw_ = min(m_b.shape[0], -(-sh // s_b)), min(m_b.shape[1], -(-sw // s_b))
            lines = [(f"Deep learning (YuNet), step {next(step)} of {total}: every grid cell scores 'is a face centred here?'", YELLOW),
                     f"{len(maps)} score grids (below): fine grids catch small faces, coarse grids catch large ones.",
                     f"Shown large: the {gw_} x {gh} grid, where this face scored highest. Black = 0, yellow = 1.",
                     f"Like a sliding window, every position is checked, but in one network pass ({ms:.0f} ms here).",
                     nav]
            tw_ = min(220, (W - 20) // len(maps) - 10)
            th_ = int(tw_ * sh / sw)
            lab_h = panel_size(["stride 8"], ts * 0.75)[1]
            xs0 = (W - (len(maps) * (tw_ + 10) - 10)) // 2
            y = H - th_ - 10
            for i, (s, m) in enumerate(maps):
                vh_, vw_ = min(m.shape[0], -(-sh // s)), min(m.shape[1], -(-sw // s))
                x = xs0 + i * (tw_ + 10)
                _paste(view, _colorize(m[:vh_, :vw_], (tw_, th_), vmax=1.0), x, y)
                cv2.rectangle(view, (x, y), (x + tw_ - 1, y + th_ - 1), YELLOW if i == best else WHITE,
                              3 if i == best else 1)
                draw_panel(view, [(f"{vw_} x {vh_} cells", WHITE)], x, y - lab_h, ts * 0.75, min_width=tw_)
            if not show(view, lines):
                return False
        else:
            next(step)

    # Step: candidate boxes before merging
    lw = max(2, H // 240)
    view = (disp * 0.6).astype(np.uint8)
    for (x, y, w, h) in raw:
        p0 = (int(x * s2d), int(y * s2d))
        p1 = (int((x + w) * s2d), int((y + h) * s2d))
        cv2.rectangle(view, p0, p1, GREEN, 1, cv2.LINE_AA)
        cv2.circle(view, ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2), 3, GREEN, -1)
    if not show(view, [
            (f"Deep learning (YuNet), step {next(step)} of {total}: candidate boxes", YELLOW),
            f"Every cell scoring above {yunet.threshold:.2f} also predicts a box and 5 landmarks: "
            f"{len(raw)} candidates.",
            "Neighbouring cells see the same face, so their boxes overlap.",
            nav]):
        return False

    # Step: non-maximum suppression
    view = disp.copy()
    for (x, y, w, h) in raw:
        cv2.rectangle(view, (int(x * s2d), int(y * s2d)), (int((x + w) * s2d), int((y + h) * s2d)),
                      DARK_GREEN, 1, cv2.LINE_AA)
    for f in faces:
        fd = f.scaled(s2d)
        draw_face(view, fd, YUNET_COLOR, lw)
        put_tag(view, f"face {f.score:.2f}", fd.box[0], fd.box[1], YUNET_COLOR, ts * 0.9)
    return show(view, [
        (f"Deep learning (YuNet), step {next(step)} of {total}: non-maximum suppression (NMS)", YELLOW),
        "Keep the highest-scoring box, delete boxes that overlap it by more than 30%, repeat.",
        f"Result: {len(faces)} face{'s' if len(faces) != 1 else ''} from {len(raw)} candidates, "
        "each with a score and 5 landmarks (eyes, nose tip, mouth corners).",
        ("Any key or click: back to live view", GREY)])


# ---------------------------------------------------------------------------
# Video sources
# ---------------------------------------------------------------------------
class StaticImage:
    """Makes a still image behave like a camera."""
    def __init__(self, img):
        self.img = img

    def isOpened(self):
        return self.img is not None

    def read(self):
        return (self.img is not None), (None if self.img is None else self.img.copy())

    def set(self, *args):
        return False

    def get(self, *args):
        return 0

    def release(self):
        pass


def open_source(src, width, height):
    if src.isdigit():
        index = int(src)
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # opens much faster on Windows
            if not cap.isOpened():
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap, "camera"
    path = Path(src)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    if path.suffix.lower() in IMAGE_EXTS:
        return StaticImage(cv2.imread(str(path))), "image"
    return cv2.VideoCapture(str(path)), "video"


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
# On-screen menu: (key, label). Each row can be clicked or triggered with its key.
MENU = [
    ("1", "Classic detector (Haar)"),
    ("2", "Deep learning detector (YuNet)"),
    ("3", "Both detectors"),
    ("w", "Step-by-step walkthrough"),   # of the detector currently selected
    ("+", "Stricter threshold"),
    ("-", "Looser threshold"),
    ("i", "Show detector input"),
    ("b", "Blur faces"),
    ("m", "Mirror image"),
    ("f", "Fullscreen"),
    (" ", "Freeze frame"),
    ("s", "Save screenshot"),
    ("h", "Hide this menu"),
    ("q", "Quit"),
]
KEY_NAMES = {" ": "space"}


class App:
    def __init__(self, args):
        models_dir = Path(args.models_dir)
        self.haar, self.haar_err = load_haar(models_dir)
        self.yunet, self.yunet_err = load_yunet(models_dir)
        for name, err in (("Haar cascade", self.haar_err), ("YuNet", self.yunet_err)):
            if err:
                print(f"\n{name} is not available.\n  " + "\n  ".join(err) + "\n")
        if not self.haar and not self.yunet:
            sys.exit("No detector could be loaded, see the messages above.")

        self.cap, self.kind = open_source(args.source, args.width, args.height)
        if not self.cap.isOpened():
            sys.exit(CAMERA_HELP if self.kind == "camera" else f"Could not open {args.source}")
        if self.kind == "video":
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.delay = max(1, int(1000 / fps))
        else:
            self.delay = 30 if self.kind == "image" else 1

        self.detect_width = args.detect_width
        self.mode = 1 if self.haar else 2
        self.mirror = self.kind == "camera"
        self.show_help = True
        self.show_input = False
        self.blur = False
        self.paused = False
        self.fullscreen = False
        self.fps = None
        self.ms = {}
        self.last_t = time.perf_counter()
        self.toast_text, self.toast_until = None, 0.0
        self.can_check_close = None
        self.menu_rects = []        # clickable areas: (x0, y0, x1, y1, key)
        self.pending_key = None     # key produced by a mouse click
        self.pending_click = False  # any click, used to advance the walkthrough
        self.in_walkthrough = False
        if self.yunet_err:   # say so on screen right away, not only in the terminal
            self.toast(["The deep learning detector (YuNet) is not available."] + self.yunet_err, 12)
        elif self.haar_err:
            self.toast(["The classic detector (Haar) is not available."] + self.haar_err, 12)

    # -- state -------------------------------------------------------------
    def active(self):
        dets = {1: [self.haar], 2: [self.yunet], 3: [self.haar, self.yunet]}[self.mode]
        return [d for d in dets if d is not None]

    def toast(self, text, seconds=2.5):
        """A message at the bottom of the screen. text can be one line or a list of lines."""
        lines = [text] if isinstance(text, str) else list(text)
        self.toast_text, self.toast_until = lines, time.perf_counter() + seconds

    def set_mode(self, mode):
        if mode in (1, 3) and not self.haar:
            return self.toast(["The classic detector (Haar) is not available."] + self.haar_err, 10)
        if mode in (2, 3) and not self.yunet:
            return self.toast(["The deep learning detector (YuNet) is not available."] + self.yunet_err, 10)
        self.mode = mode
        self.ms.clear()
        self.toast({1: "Now using: Haar cascade (classic)",
                    2: "Now using: YuNet CNN (deep learning)",
                    3: "Comparing both: blue = Haar, green = YuNet"}[mode])

    # -- mouse ---------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.in_walkthrough:
            self.pending_click = True
            return
        for x0, y0, x1, y1, key in self.menu_rects:
            if x0 <= x < x1 and y0 <= y < y1:
                self.pending_key = ord(key)
                return

    def poll_click(self):
        clicked, self.pending_click = self.pending_click, False
        return clicked

    def menu_state(self, key):
        """'active' / 'on' for the current mode or a toggle that is on, 'unavailable', or None."""
        if key in ("1", "2", "3"):
            m = int(key)
            if (m == 1 and not self.haar) or (m == 2 and not self.yunet) or \
               (m == 3 and not (self.haar and self.yunet)):
                return "unavailable"
            return "active" if self.mode == m else None
        toggles = {"i": self.show_input, "b": self.blur, "m": self.mirror,
                   "f": self.fullscreen, " ": self.paused}
        if key in toggles:
            return "on" if toggles[key] else None
        return None

    def window_visible(self):
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return 0

    def window_closed(self):
        return bool(self.can_check_close) and self.window_visible() < 1

    # -- main loop ---------------------------------------------------------
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        raw, failures = None, 0
        while True:
            if raw is None or not self.paused:
                ok, new = self.cap.read()
                if not ok or new is None:
                    failures += 1
                    if self.kind == "video":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the video
                        if failures > 5:
                            print("Could not read the video file.")
                            break
                        continue
                    if failures > 40:
                        print(CAMERA_HELP)
                        break
                    time.sleep(0.05)   # cameras often need a moment to warm up
                    continue
                failures, raw = 0, new
            frame = cv2.flip(raw, 1) if self.mirror else raw
            cv2.imshow(WINDOW, self.render(frame))
            key = cv2.waitKey(self.delay) & 0xFF
            if key == 255 and self.pending_key is not None:   # a menu row was clicked
                key, self.pending_key = self.pending_key, None
            if self.can_check_close is None:
                # Some window backends do not report visibility; only check if this one does
                self.can_check_close = self.window_visible() >= 1
            elif self.window_closed():
                break
            if not self.handle_key(key, frame):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def handle_key(self, key, frame):
        if key == 255:
            return True
        if key == 27:
            return False
        c = chr(key).lower() if key < 128 else ""
        if not c:
            return True
        if c == "q":
            return False
        if c in ("1", "2", "3"):
            self.set_mode(int(c))
        elif c in ("+", "="):
            self.adjust(+1)
        elif c in ("-", "_"):
            self.adjust(-1)
        elif c == "w":
            self.run_walkthrough(frame)
        elif c == "i":
            self.show_input = not self.show_input
        elif c == "b":
            self.blur = not self.blur
            self.toast("Privacy pixelation " + ("on" if self.blur else "off"))
        elif c == "m":
            self.mirror = not self.mirror
        elif c == "f":
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif c == " ":
            self.paused = not self.paused
            self.toast("Frame frozen (detection still runs on it)" if self.paused else "Live again")
        elif c == "s":
            name = time.strftime("face_demo_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, self.last_view)
            print(f"Saved {Path(name).resolve()}")
            self.toast(f"Saved {name}")
        elif c == "h":
            self.show_help = not self.show_help
        return True

    def run_walkthrough(self, frame):
        """Walk through the detector that is selected: 1 = Haar, 2 = YuNet, 3 = Haar then YuNet."""
        self.in_walkthrough, self.pending_click = True, False
        try:
            for det in self.active():
                if det is self.haar:
                    finished = walkthrough(frame, self.haar, self.window_closed, self.poll_click)
                else:
                    finished = yunet_walkthrough(frame, self.yunet, self.detect_width,
                                                 self.window_closed, self.poll_click)
                if not finished:   # q pressed or window closed: back to the live view
                    break
        finally:
            self.in_walkthrough = False
        self.last_t = time.perf_counter()

    def adjust(self, step):
        for det in self.active():
            det.adjust(step)
        self.toast("   ".join(d.threshold_text() for d in self.active()))

    # -- rendering ---------------------------------------------------------
    def render(self, frame):
        now = time.perf_counter()
        dt, self.last_t = now - self.last_t, now
        if dt > 0:
            self.fps = 1 / dt if self.fps is None else 0.9 * self.fps + 0.1 / dt

        view = frame.copy()
        results = []
        for det in self.active():
            faces, ms, small = run_detector(det, frame, self.detect_width)
            prev = self.ms.get(det.key)
            self.ms[det.key] = ms if prev is None else 0.85 * prev + 0.15 * ms
            results.append((det, faces, small))

        if self.blur and results:
            for f in results[-1][1]:   # YuNet's boxes when comparing
                pixelate(view, expand_box(f.box, 0.25))

        compare = len(results) > 1
        lw = max(2, view.shape[0] // 240)
        tags = []
        for det, faces, _ in results:
            crowded = len(faces) > 12   # e.g. Haar with min neighbours 0: thin boxes, no labels
            for f in faces:
                draw_face(view, f, det.color, 1 if crowded else lw, landmarks=not crowded)
                if crowded:
                    continue
                parts = [det.short if compare else "face"]
                if f.score is not None:
                    parts.append(f"{f.score:.2f}")
                below = compare and det is self.yunet
                x, y, w, h = f.box
                tags.append((" ".join(parts), x, y + h if below else y, det.color, below))
        for text, x, y, color, below in tags:   # labels last so no box covers them
            put_tag(view, text, x, y, color, text_scale(view) * 0.85, below)

        self.draw_hud(view, results)
        self.last_view = view
        return view

    def draw_hud(self, view, results):
        H, W = view.shape[:2]
        ts = text_scale(view)

        # Top bar: one line per detector, then a one-line description
        lines = []
        for det, faces, _ in results:
            n = len(faces)
            what = "face" if n == 1 else "faces"
            if det is self.haar and det.min_neighbors == 0:
                what = "raw hit" if n == 1 else "raw hits (no merging)"
            lines.append((f"[{det.key}] {det.name}:  {n} {what}   "
                          f"{self.ms[det.key]:.1f} ms   {det.threshold_text()}", det.color))
        if len(results) == 1:
            lines.append((results[0][0].about, GREY))
        else:
            lines.append(("Blue = Haar, green = YuNet. Turn your head to compare.", GREY))
        _, bar_bottom = draw_panel(view, lines, 0, 0, ts, min_width=W)

        self.menu_rects = []
        if self.show_help:
            self.draw_menu(view, 10, bar_bottom + 10, ts * 0.85)

        # What each detector actually receives
        if self.show_input:
            y = bar_bottom + 10
            for det, _, small in results:
                img, desc = det.model_input(small)
                iw = int(W * 0.26)
                ih = int(img.shape[0] * iw / img.shape[1])
                cap = [(f"{det.short} sees: {small.shape[1]}x{small.shape[0]}, {desc}", det.color)]
                cw, chh = panel_size(cap, ts * 0.8)
                pw = max(iw, cw)
                if y + chh + ih > H - 10:
                    break
                x = W - pw - 10
                draw_panel(view, cap, x, y, ts * 0.8, min_width=pw, alpha=0.75)
                view[y + chh:y + chh + ih, x:x + iw] = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
                cv2.rectangle(view, (x, y + chh), (x + iw - 1, y + chh + ih - 1), det.color, 2)
                y += chh + ih + 10

        # Bottom right: FPS and status flags
        flags = [f"{self.fps:.0f} FPS" if self.fps else "-- FPS"]
        if self.paused:
            flags.append("FROZEN")
        if self.blur:
            flags.append("PRIVACY ON")
        ftxt = [("   ".join(flags), WHITE)]
        fw, fh = panel_size(ftxt, ts)
        draw_panel(view, ftxt, W - fw - 10, H - fh - 10, ts)

        if not self.show_help:   # small clickable button to bring the menu back
            label = [("Show menu (h)", WHITE)]
            bw, bh = panel_size(label, ts * 0.85)
            draw_panel(view, label, 10, H - bh - 10, ts * 0.85, alpha=0.7)
            cv2.rectangle(view, (10, H - bh - 10), (10 + bw, H - 10), GREY, 1)
            self.menu_rects.append((10, H - bh - 10, 10 + bw, H - 10, "h"))

        if self.toast_text and time.perf_counter() < self.toast_until:
            lines = [(self.toast_text[0], YELLOW)] + [(t, WHITE) for t in self.toast_text[1:]]
            tw, th = panel_size(lines, ts)
            draw_panel(view, lines, max(0, (W - tw) // 2), H - th - 70, ts, alpha=0.8)

    def draw_menu(self, view, x, y, ts):
        """The menu on the left. Every row is a button: click it or press its key."""
        th = thickness(ts)
        pad = max(6, int(14 * ts))
        line_h = int(cv2.getTextSize("Ag", FONT, ts, th)[0][1] * 2.0)
        key_w = cv2.getTextSize("space", FONT, ts, th)[0][0] + pad
        title = "Click a row, or press its key"
        suffix = {"on": "  (on)", "unavailable": "  (not available)"}
        which = {1: " (Haar)", 2: " (YuNet)", 3: " (both)"}[self.mode]   # w follows the selected detector
        rows = [(key, label + (which if key == "w" else "") + suffix.get(self.menu_state(key), ""),
                 self.menu_state(key)) for key, label in MENU]
        width = max([cv2.getTextSize(title, FONT, ts, th)[0][0]] +
                    [key_w + cv2.getTextSize(label, FONT, ts, th)[0][0] for _, label, _ in rows]) + 2 * pad
        height = line_h * (len(rows) + 1) + pad
        shade(view, x, y, x + width, y + height, 0.65)
        cv2.putText(view, title, (x + pad, y + pad // 2 + int(line_h * 0.72)), FONT, ts, YELLOW, th, cv2.LINE_AA)
        for i, (key, label, state) in enumerate(rows, start=1):
            y0 = y + pad // 2 + line_h * i
            base = y0 + int(line_h * 0.72)
            if state in ("active", "on"):
                cv2.rectangle(view, (x + 3, y0 + 1), (x + width - 3, y0 + line_h - 1), YELLOW, 1)
            color = YELLOW if state in ("active", "on") else (110, 110, 110) if state == "unavailable" else WHITE
            cv2.putText(view, KEY_NAMES.get(key, key), (x + pad, base), FONT, ts, color, th, cv2.LINE_AA)
            cv2.putText(view, label, (x + pad + key_w, base), FONT, ts, color, th, cv2.LINE_AA)
            self.menu_rects.append((x, y0, x + width, y0 + line_h, key))
        return x + width, y + height


def parse_args():
    p = argparse.ArgumentParser(description="Live face detection classroom demo")
    p.add_argument("--source", default="0",
                   help="camera index (0, 1, ...) or path to an image or video file")
    p.add_argument("--width", type=int, default=1280, help="requested camera width")
    p.add_argument("--height", type=int, default=720, help="requested camera height")
    p.add_argument("--detect-width", type=int, default=640,
                   help="frames are shrunk to this width before detection (smaller is faster)")
    p.add_argument("--models-dir", default=str(SCRIPT_DIR / "models"),
                   help="where downloaded model files are kept")
    args, _ = p.parse_known_args()   # ignore extra arguments Jupyter adds
    return args


def main():
    args = parse_args()
    try:   # hide OpenCV's informational warnings
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    print(f"OpenCV {cv2.__version__}")
    app = App(args)
    print("Window open. Press h in the window for help, q to quit.")
    app.run()


if __name__ == "__main__":
    main()
