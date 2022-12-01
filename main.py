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

# ---------------------------------------------------------------------------------------------------------------------- #

def test_depth_api():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--filename"
    
    mode: str = "image"
    base_url: str = "http://localhost:9090"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: filename: str = sys.argv[sys.argv.index(args_3) + 1]

    
    url = f"{base_url}/infer"

    if mode == "image":

        assert filename in os.listdir(INPUT_PATH), "File not found"

        image = cv2.imread(os.path.join(INPUT_PATH, filename))

        payload = {
            "imageData" : encode_image_to_base64(image=image)
        }

        response = requests.post(url=url, json=payload)
        if response.status_code == 200:
            show_images(
                image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                image_2=decode_image(response.json()["imageData"]),
                cmap_1="gnuplot2",
                cmap_2="gnuplot2",
                title_1="Original",
                title_2="Depth Image",
            )
        else:
            print(f"Error {response.status_code} : {response.reason}")
    
    elif mode == "realtime":

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret: continue
            frameData = encode_image_to_base64(image=frame)
            payload = {
                "imageData" : frameData
            }       

            response = requests.post(url=url, json=payload)
            if response.status_code == 200:
                frame = cv2.cvtColor(src=decode_image(response.json()["imageData"]), code=cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(frame, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Invalid Mode")
    
# ---------------------------------------------------------------------------------------------------------------------- #

def test_cv_api():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--model"
    args_4: str = "--filename"
    
    mode: str = "image"
    base_url: str = "http://192.168.10.3:4040"
    model: str = "classify"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: model: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename: str = sys.argv[sys.argv.index(args_4) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"
    assert model == "classify" or model == "detect" or model == "segment" or model == "face", "Invalid Model Type"

    if mode == "image":
        assert filename in os.listdir(INPUT_PATH), "File not Found"

        image = cv2.imread(os.path.join(INPUT_PATH, filename))
        payload = {
            "imageData" : encode_image_to_base64(image=image)
        }

        response = requests.request(method="POST", url=f"{base_url}/{model}", json=payload)
        
        if response.status_code == 200 and response.json()["statusCode"] == 200:
            if model == "classify":
                print(response.json()["label"])

            if model == "detect":
                # print(response.json()["label"])
                cv2.rectangle(image, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                cv2.putText(image, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                show_image(image)

            if model == "segment":
                print(f"Classes Present : {response.json()['labels']}")
                image = decode_image(response.json()["imageData"])
                show_image(image)
            
            if model == "face":
                    face_detections = response.json()["face_detections"]
                    draw_detections(image=frame, face_detections=face_detections)
        else:
            print(f"Error {response.status_code} : {response.reason}")

    else:
        
        if platform.system() != "Windows":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frameData = encode_image_to_base64(image=frame)
            payload = {
                "imageData" : frameData
            }       

            response = requests.request(method="POST", url=f"{base_url}/{model}", json=payload)

            if response.status_code == 200 and response.json()["statusCode"] == 200:
                if model == "classify":
                    cv2.putText(frame, response.json()["label"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if model == "detect":
                    cv2.rectangle(frame, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                    cv2.putText(frame, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if model == "segment":
                    disp_frameData = response.json()["imageData"]
                    frame = decode_image(disp_frameData)
                
                if model == "face":
                    face_detections = response.json()["face_detections"]
                    draw_detections(image=frame, face_detections=face_detections)
            
            else:
                cv2.putText(frame, "Error", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
            cv2.imshow("Processed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------------------------- #

def test_haar_api():    

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--model"
    args_4: str = "--filename"
    
    mode: str = "image"
    base_url: str = "http://localhost:10010"
    model: str = "face"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: model: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename: str = sys.argv[sys.argv.index(args_4) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"
    assert model == "face" or model == "eye", "Invalid Model Type"

    if mode == "image":
        assert filename in os.listdir(INPUT_PATH), "File not Found"
        
        image = cv2.imread(os.path.join(INPUT_PATH, filename))
        payload = {
            "imageData" : encode_image_to_base64(image=image)
        }

        if model == "face":
            response = requests.request(method="POST", url=f"{base_url}/detect/face", json=payload)
            if response.status_code == 200:
                face_detections = response.json()["face_detections"]
                draw_detections(image=image, face_detections=face_detections)
                show_image(image=image)   
            else:
                print(f"Error {response.status_code} : {response.reason}")
        
        else:
            response = requests.request(method="POST", url=f"{base_url}/detect/eye", json=payload)
            if response.status_code == 200:
                face_detections = response.json()["face_detections"]
                eye_detections  = response.json()["eye_detections"]
                draw_detections(image=image, face_detections=face_detections, eye_detections=eye_detections)
                show_image(image=image)  
            else:
                print(f"Error {response.status_code} : {response.reason}")

    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret: continue
            frameData = encode_image_to_base64(image=frame)
            payload = {
                "imageData" : frameData
            }       

            if model == "face":
                response = requests.request(method="POST", url=f"{base_url}/detect/face", json=payload)
                if response.status_code == 200:
                    face_detections = response.json()["face_detections"]
                    draw_detections(image=frame, face_detections=face_detections)
                else:
                    cv2.putText(frame, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            else:
                response = requests.request(method="POST", url=f"{base_url}/detect/eye", json=payload)
                if response.status_code == 200:
                    face_detections = response.json()["face_detections"]
                    eye_detections  = response.json()["eye_detections"]
                    draw_detections(image=image, face_detections=face_detections, eye_detections=eye_detections)
                else:
                    cv2.putText(frame, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------------------------- #

def test_yolo_api():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--model-type"
    args_4: str = "--filename"
    args_5: str = "--version"
    
    mode: str = "image"
    base_url: str = "http://192.168.10.3:5050"
    model_type: str = "tiny"
    filename: str = "Test_1.jpg"
    version: int = 6

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: model_type: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename: str = sys.argv[sys.argv.index(args_4) + 1]
    if args_5 in sys.argv: version: int = int(sys.argv[sys.argv.index(args_5) + 1])

    assert mode == "image" or mode == "realtime", "Invalid Mode"
    assert model_type == "small" or model_type == "tiny" or model_type == "nano", "Invalid Model Type"

    if mode == "image":
        
        assert filename in os.listdir(INPUT_PATH), "File not Found"

        image: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, filename))
        payload = {
            "imageData" : encode_image_to_base64(image=image)
        }

        response = requests.request(method="POST", url=f"{base_url}/infer/v{version}/{model_type}", json=payload)
        if response.status_code == 200 and response.json()["statusCode"] == 200:
            cv2.rectangle(image, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
            show_image(image=image, title=f"{response.json()['label'].title()} : {response.json()['score']}")
        else:
            print(f"Error {response.status_code} : {response.reason}")
            print(f"Error {response.json()['statusCode']} : {response.json()['statusText']}")

    else:
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frameData = encode_image_to_base64(image=frame)
            payload = {
                "imageData" : frameData
            }       

            response = requests.request(method="POST", url=f"{base_url}/infer/v{version}/{model_type}", json=payload)
            if response.status_code == 200 and response.json()["statusCode"] == 200:
                cv2.rectangle(frame, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                cv2.putText(frame, response.json()["label"].title(), (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "ERROR", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------------------------- #

def test_facial_recognition_api():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--f1"
    args_4: str = "--f2"

    mode: str = "image"
    base_url: str = "http://localhost:3032"
    filename_1: str = "Test_3.jpg"
    filename_2: str = "Test_4.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: filename_1: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename_2: str = sys.argv[sys.argv.index(args_4) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"
     
    if mode == "image":
        assert filename_1 in os.listdir(INPUT_PATH), "File 1 not Found"
        assert filename_2 in os.listdir(INPUT_PATH), "File 2 not Found"

        image_1: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, filename_1))
        image_2: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, filename_2))
        # # image_1: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, "Face_3.jpg"))[:, ::-1, :] # Horizontal Flip
        
        payload = {
            "imageData_1" : encode_image_to_base64(image=image_1),
            "imageData_2" : encode_image_to_base64(image=image_2)
        }

        response = requests.request(method="POST", url=f"{base_url}/compare", json=payload)
        if response.status_code == 200 and response.json()["statusCode"] == 200:
            print(f"Simialrity : {float(response.json()['cosine_similarity']):.2f}")
        else:
            print(f"Error {response.status_code} : {response.reason}")
            print(f"Error {response.json()['statusCode']} : {response.json()['statusText']}")

    
    # imageData_1: str = encode_image_to_base64(image=image_1)

    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret: break
    #     payload = {
    #         "imageData_1" : imageData_1,
    #         "imageData_2" : encode_image_to_base64(image=frame)
    #     }   

    #     response = requests.request(method="POST", url="http://127.0.0.1:50000/compare", json=payload)
    #     if response.status_code == 200:
    #         if response.json()["statusCode"] == 200:
    #             # print(f"Simialrity : {float(response.json()['cosine_similarity']):.2f}")
    #             cv2.putText(frame, f"Simialrity : {float(response.json()['cosine_similarity']):.2f}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        
    #     cv2.imshow("Feed", frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"): 
    #         break
        
    # cap.release()
    # cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------------------------- #

def test_fsalary_converter_api():

    args_1: str = "--salary"
    args_2: str = "--from"
    args_3: str = "--to"
    args_4: str = "--base-url"

    salary: float = 0.0
    from_country: str = "QAT"
    to_country: str = "IND"
    base_url: str = "http://127.0.0.1:50002"

    if args_1 in sys.argv: salary = float(sys.argv[sys.argv.index(args_1) + 1]) 
    if args_2 in sys.argv: from_country = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: to_country = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: base_url = sys.argv[sys.argv.index(args_4) + 1]

    payload = {
        "salary" : salary,
        "from_country" : from_country,
        "to_country" : to_country
    }

    response = requests.request(method="POST", url=f"{base_url}/convert", json=payload)
    print(response.json())

# ---------------------------------------------------------------------------------------------------------------------- #

def test_bg_remove_api():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--li"

    mode: str = "remove"
    base_url: str = "http://192.168.10.10:3030"
    lightweight: bool = False

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: lightweight = True

    if not lightweight:
        url: str = base_url + f"/{mode}"
    else:
        url: str = base_url + f"/{mode}" + "/li"

    image: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, "Test_1.jpg"))

    payload = {
        "imageData" : encode_image_to_base64(image=image)
    }

    if mode == "remove":
        response = requests.request(method="POST", url=url, json=payload)
        if response.status_code == 200:
            show_images(
                image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                image_2=cv2.cvtColor(src=decode_image(response.json()["bglessImageData"]), code=cv2.COLOR_BGR2RGB),
                cmap_1="gnuplot2",
                cmap_2="gnuplot2",
                title_1="Original",
                title_2="BG Removed Image",
            )
        else:
            print(f"Error {response.status_code} : {response.reason}")
    

    elif mode == "replace":
        image_2: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, "Test_2.jpg"))


        payload = {
            "imageData_1" : encode_image_to_base64(image=image),
            "imageData_2" : encode_image_to_base64(image=image_2)
        }

        response = requests.request(method="POST", url=url, json=payload)
        response = requests.request(method="POST", url=url, json=payload)


        if response.status_code == 200:
            show_images(
                image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                image_2=cv2.cvtColor(src=decode_image(response.json()["bgreplaceImageData"]), code=cv2.COLOR_BGR2RGB),
                cmap_1="gnuplot2",
                cmap_2="gnuplot2",
                title_1="Original",
                title_2="BG Replaced Image",
            )
        else:
            print(f"Error {response.status_code} : {response.reason}")

# ---------------------------------------------------------------------------------------------------------------------- #

def test_bg_remove_api_render():

    args_1: str = "--mode"
    args_2: str = "--base-url"

    mode: str = "remove"
    base_url: str = "http://localhost:3030"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
        
    url: str = base_url + f"/{mode}"
    
    image: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, "Test_1.jpg"))
    payload = {
        "imageData" : encode_image_to_base64(image=image)
    }

    if mode == "remove":
        response = requests.request(method="POST", url=url, json=payload)
        if response.status_code == 200:
            show_images(
                image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                image_2=cv2.cvtColor(src=decode_image(response.json()["bglessImageData"]), code=cv2.COLOR_BGR2RGB),
                cmap_1="gnuplot2",
                cmap_2="gnuplot2",
                title_1="Original",
                title_2="BG Removed Image",
            )
        else:
            print(f"Error {response.status_code} : {response.reason}")
    

    elif mode == "replace":
        image_2: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, "Test_2.jpg"))
        payload = {
            "imageData_1" : encode_image_to_base64(image=image),
            "imageData_2" : encode_image_to_base64(image=image_2)
        }

        response = requests.request(method="POST", url=url, json=payload)
        response = requests.request(method="POST", url=url, json=payload)


        if response.status_code == 200:
            show_images(
                image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                image_2=cv2.cvtColor(src=decode_image(response.json()["bgreplaceImageData"]), code=cv2.COLOR_BGR2RGB),
                cmap_1="gnuplot2",
                cmap_2="gnuplot2",
                title_1="Original",
                title_2="BG Replaced Image",
            )
        else:
            print(f"Error {response.status_code} : {response.reason}")

# ---------------------------------------------------------------------------------------------------------------------- #

def test_fdis_api():

    args_1: str = "--base-url"
    args_2: str = "--filename"
    
    base_url: str = "http://localhost:3032"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: filename: str = sys.argv[sys.argv.index(args_2) + 1]

    assert filename in os.listdir(INPUT_PATH), "File not Found"

    image: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, filename))

    payload = {
        "imageData" : encode_image_to_base64(image=image)
    }

    response = requests.request(method="POST", url=f"{base_url}/infer/", json=payload)
    if response.status_code == 200:
        if response.json()["statusCode"] == 200:
           print(response.json()["label"])
        else:
            print(f"Error {response.json()['statusCode']} : {response.json()['statusText']}")
    else:
        print(f"Error {response.status_code} : {response.reason}")

# ---------------------------------------------------------------------------------------------------------------------- #   

def test_wic_api():

    args_1: str = "--base-url"
    args_2: str = "--filename"
    
    base_url: str = "http://localhost:3034"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: filename: str = sys.argv[sys.argv.index(args_2) + 1]

    assert filename in os.listdir(INPUT_PATH), "File not Found"

    image: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, filename))

    payload = {
        "imageData" : encode_image_to_base64(image=image)
    }

    response = requests.request(method="POST", url=f"{base_url}/infer/", json=payload)
    if response.status_code == 200:
        if response.json()["statusCode"] == 200:
           print(response.json()["label"])
        else:
            print(f"Error {response.json()['statusCode']} : {response.json()['statusText']}")
    else:
        print(f"Error {response.status_code} : {response.reason}")

# ---------------------------------------------------------------------------------------------------------------------- #   

def test_ffc_api():

    args_1: str = "--base-url"
    args_2: str = "--filename"
    
    base_url: str = "http://192.168.10.3:3036"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: filename: str = sys.argv[sys.argv.index(args_2) + 1]

    assert filename in os.listdir(INPUT_PATH), "File not Found"

    image: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, filename))

    payload = {
        "imageData" : encode_image_to_base64(image=image)
    }

    response = requests.request(method="POST", url=f"{base_url}/infer/", json=payload)
    if response.status_code == 200:
        if response.json()["statusCode"] == 200:
           print(response.json()["label"])
        else:
            print(f"Error {response.json()['statusCode']} : {response.json()['statusText']}")
    else:
        print(f"Error {response.status_code} : {response.reason}")
    
# ---------------------------------------------------------------------------------------------------------------------- #   

def test_aic_api():

    args_1: str = "--base-url"
    args_2: str = "--filename"
    
    base_url: str = "http://192.168.10.3:3038"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: filename: str = sys.argv[sys.argv.index(args_2) + 1]

    assert filename in os.listdir(INPUT_PATH), "File not Found"

    image: np.ndarray = cv2.imread(os.path.join(INPUT_PATH, filename))

    payload = {
        "imageData" : encode_image_to_base64(image=image)
    }

    response = requests.request(method="POST", url=f"{base_url}/infer/", json=payload)
    if response.status_code == 200:
        if response.json()["statusCode"] == 200:
           print(response.json()["label"])
        else:
            print(f"Error {response.json()['statusCode']} : {response.json()['statusText']}")
    else:
        print(f"Error {response.status_code} : {response.reason}")

# ---------------------------------------------------------------------------------------------------------------------- #   


def main():
    test_bg_remove_api_render()

# ---------------------------------------------------------------------------------------------------------------------- #   

if __name__ == '__main__':
    sys.exit(main() or 0)

# ---------------------------------------------------------------------------------------------------------------------- #   

