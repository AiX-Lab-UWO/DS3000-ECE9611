# Live face detection demo

Run face detection live on your laptop camera and see how two kinds of detectors work:

* **Classic detector: Haar cascade** (Viola and Jones, 2001). Hand-designed light/dark rectangle features, checked in a window that slides over the image at many sizes.
* **Deep learning detector: YuNet** (Wu, Peng and Yu, 2023). A small convolutional neural network that also finds 5 face landmarks (eyes, nose tip, mouth corners).

You can switch between the two, run both on the same frame, and step through how each one finds a face.

## Setup

You need Python 3.9 or newer and a webcam.

```
git clone https://github.com/AiX-Lab-UWO/DS3000-ECE9611.git
cd DS3000-ECE9611/face-detection-demo
pip install -r requirements.txt
```

No git? On the course repository's main page, click the green **Code** button, choose **Download ZIP**, unzip it, and open the `face-detection-demo` folder.

Install `opencv-contrib-python`, not `opencv-python`: from OpenCV 5 on, the plain package no longer includes the Haar cascade detector. The two packages conflict, so if `opencv-python` is already installed, run `pip uninstall opencv-python` first.

## Run

In a terminal, inside the `face-detection-demo` folder:

```
python face_detection_demo.py
```

| Option | What it does |
|---|---|
| `--source 1` | use another camera (for example a USB webcam) |
| `--source photo.jpg` | run on an image or video file instead of the camera |

In Jupyter, open `run_demo.ipynb` and run its cells. The camera opens in its own window, which sometimes appears behind the browser.

## Controls

Click the camera window once so it receives key presses. Then press a key, or click a line in the menu on the left of the window.

| Key | Action |
|---|---|
| `1` | Classic detector (Haar) |
| `2` | Deep learning detector (YuNet) |
| `3` | Both detectors on the same frame (blue = Haar, green = YuNet) |
| `w` | Step-by-step walkthrough of the selected detector |
| `+` / `-` | Stricter / looser detection threshold |
| `i` | Show the image each detector actually receives |
| `b` | Pixelate detected faces (privacy) |
| `m` | Mirror the image |
| `f` | Fullscreen |
| `space` | Freeze the frame (detection keeps running on it) |
| `s` | Save a screenshot |
| `h` | Hide / show the menu |
| `q` or `Esc` | Quit |

## The walkthroughs (`w`)

**Classic detector (Haar), 5 steps:** the grayscale image the detector uses, the window sliding over positions and sizes, all raw windows that passed the cascade, how overlapping windows are merged into faces, and two of the light-minus-dark features measured on your own face.

**Deep learning (YuNet), 6 steps:** the image as numbers, the 16 filters the first layer learned, the channels of a deep layer that respond to faces, the face score of every grid cell, the candidate boxes, and non-maximum suppression down to the final face.

In mode `3`, `w` runs both walkthroughs one after the other. Press any key or click to go to the next step, and `q` to return to the live view.

## Things to try

1. In mode `3`, turn your head slowly to the side. Which detector loses your face first?
2. In mode `1`, press `-` until "min neighbours" reaches 0. How many raw windows fire for one face, and why does the detector need a merging step?
3. In mode `2`, press `-` to lower YuNet's score threshold to 0.1. Do false detections appear?
4. Cover part of your face with your hand, or dim the lights. Which detector copes better?

## Troubleshooting

* **The camera does not open.** Close other apps that use it (Zoom, Teams, the Camera app). On macOS, allow your terminal or editor under System Settings > Privacy & Security > Camera, then restart it. On Windows, allow desktop apps under Settings > Privacy & security > Camera. If you have several cameras, try `--source 1`.
* **A message at the bottom of the window says a detector is not available.** It also says why and how to fix it. Usually the fix is `pip install --upgrade opencv-contrib-python`.
* **Nothing happens when I press keys.** Click on the camera window first.

## Files

| File | Contents |
|---|---|
| `face_detection_demo.py` | the demo |
| `run_demo.ipynb` | runs the demo from Jupyter |
| `requirements.txt` | the Python packages it needs |
| `models/` | YuNet model files and their licences: `face_detection_yunet_2023mar.onnx` for OpenCV 4.8 and newer, `face_detection_yunet_2022mar.onnx` for OpenCV 4.5.4 to 4.7. The script picks the right one. |

The Haar cascade comes with OpenCV. If your OpenCV does not include it, the script downloads it into `models/` on the first run.

## Credits

* P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," CVPR 2001. Haar cascade model `haarcascade_frontalface_default.xml` by Rainer Lienhart, distributed with OpenCV.
* W. Wu, H. Peng and S. Yu, "YuNet: A Tiny Millisecond-level Face Detector," Machine Intelligence Research, 2023. Model files from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (2023 version) and [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train) (2022 version), both under the MIT licence; see `models/`.
