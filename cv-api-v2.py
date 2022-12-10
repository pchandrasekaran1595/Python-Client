import os
import re
import io
import sys
import cv2
import base64
import platform
import requests
import numpy as np
import matplotlib.pyplot as plt

from time import time
from PIL import Image

INPUT_PATH: str = "input"


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def encode_image_to_base64(header: str = "data:image/png;base64", image: np.ndarray = None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData


def decode_image(imageData) -> np.ndarray:
    _, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    return image


def show_image(image: np.ndarray, cmap: str="gnuplot2", title: str=None) -> None:
    plt.figure()
    plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    if platform.system() == "Windows":
        figmanager = plt.get_current_fig_manager()
        figmanager.window.state("zoomed")
    plt.show()


def show_images(
    image_1: np.ndarray,
    image_2: np.ndarray, 
    cmap_1: str = "gnuplot2",
    cmap_2: str = "gnuplot2",
    title_1: str=None,
    title_2: str=None,
) -> None:

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap=cmap_1)
    plt.axis("off")
    if title_1: plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap=cmap_2)
    plt.axis("off")
    if title_2: plt.title(title_2)
    if platform.system() == "Windows":
        figmanager = plt.get_current_fig_manager()
        figmanager.window.state("zoomed")
    plt.show()


def draw_box(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_detections(image: np.ndarray, face_detections: tuple, eye_detections: tuple=None):
    if eye_detections is None:
        for (x, y, w, h) in face_detections:
            cv2.rectangle(img=image, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
    else:
        for (x1, y1, w1, h1) in face_detections:
            for (x2, y2, w2, h2) in eye_detections:
                cv2.rectangle(img=image[y1:y1+h1, x1:x1+w1], pt1=(x2, y2), pt2=(x2+w2, y2+h2), color=(255, 0, 0), thickness=2)
            cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x1+w1, y1+h1), color=(0, 255, 0), thickness=2)


def main():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--model"
    args_4: str = "--filename-1"
    args_5: str = "--filename-2"

    mode: str = "image"
    base_url: str = "http://localhost:4040"
    model: str = "classify"
    filename_1: str = "Test_1.jpg"
    filename_2: str = "Test_2.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: model: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename_1: str = sys.argv[sys.argv.index(args_4) + 1]
    if args_5 in sys.argv: filename_2: str = sys.argv[sys.argv.index(args_5) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"
    assert model == "classify" or \
           model == "detect" or \
           model == "segment" or \
           model == "remove" or \
           model == "replace" or \
           model == "depth" or \
           model == "face", f"{model.title()} is an invalid model type"

    if mode == "image":
        assert filename_1 in os.listdir(INPUT_PATH), f"{filename_1} not found in input directory"

        image = cv2.imread(os.path.join(INPUT_PATH, filename_1))

        if not re.match(r"replace", model, re.IGNORECASE):
            payload = {
                "imageData" : encode_image_to_base64(image=image)
            }

        else:
            assert filename_2 in os.listdir(INPUT_PATH), f"{filename_2} not found in input directory"

            image_2: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, filename_2))
            payload = {
                "imageData_1" : encode_image_to_base64(image=image),
                "imageData_2" : encode_image_to_base64(image=image_2)
            }

        response = requests.request(method="POST", url=f"{base_url}/{model}", json=payload)

        if response.status_code == 200 and response.json()["statusCode"] == 200:
            if model == "classify":
                print(response.json()["label"])

            if model == "detect":
                cv2.rectangle(image, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                cv2.putText(image, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                show_image(image)

            if model == "segment":
                print(f"Classes Present : {response.json()['labels']}")
                image = decode_image(response.json()["imageData"])
                show_image(image)
            
            if model == "remove":
                show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=cv2.cvtColor(src=decode_image(response.json()["bglessImageData"]), code=cv2.COLOR_BGR2RGB),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="BG Removed Image",
                )
            
            if model == "replace":
                show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=cv2.cvtColor(src=decode_image(response.json()["bgreplaceImageData"]), code=cv2.COLOR_BGR2RGB),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="BG Replaced Image",
                )
            
            if model == "depth":
                show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=decode_image(response.json()["imageData"]),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="Depth Image",
                )
            
            if model == "face":
                    face_detections = response.json()["face_detections"]
                    draw_detections(image=image, face_detections=face_detections)
        else:
            print(f"Error {response.status_code} : {response.reason}")

    else:
        pass
        
        # if platform.system() != "Windows":
        #     cap = cv2.VideoCapture(0)
        # else:
        #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        # while True:
        #     ret, frame = cap.read()
        #     if not ret: break
        #     frameData = encode_image_to_base64(image=frame)
        #     payload = {
        #         "imageData" : frameData
        #     }       

        #     response = requests.request(method="POST", url=f"{base_url}/{model}", json=payload)

        #     if response.status_code == 200 and response.json()["statusCode"] == 200:
        #         if model == "classify":
        #             cv2.putText(frame, response.json()["label"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #         if model == "detect":
        #             cv2.rectangle(frame, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
        #             cv2.putText(frame, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #         if model == "segment":
        #             disp_frameData = response.json()["imageData"]
        #             frame = decode_image(disp_frameData)
                
        #         if model == "face":
        #             face_detections = response.json()["face_detections"]
        #             draw_detections(image=frame, face_detections=face_detections)
            
        #     else:
        #         cv2.putText(frame, "Error", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
        #     cv2.imshow("Processed", frame)
            
        #     if cv2.waitKey(1) & 0xFF == ord("q"): 
        #         break
            
        # cap.release()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)