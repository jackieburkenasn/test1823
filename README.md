# Face Swap with MediaPipe

This project demonstrates how to perform basic face swapping between two images using Google's [MediaPipe](https://github.com/google/mediapipe) library. It is intended for educational and comedic purposes only. Do not use this project to impersonate others or create misleading or defamatory content.

## Requirements
- Python 3.8+
- mediapipe
- opencv-python

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
Prepare two images with visible faces: a source face and a target image. Run the script:
```bash
python src/face_swap.py path/to/source.jpg path/to/target.jpg --output swapped.jpg
```
The output image `swapped.jpg` will contain the source face warped onto the target image.

## Disclaimer
This code is provided for learning purposes. Always respect privacy and obtain consent from anyone whose likeness is used. Clearly disclose when content has been altered.
