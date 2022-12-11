import os
import sys
import cv2
import platform
import requests
import utils as u


def main():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--model-type"
    args_4: str = "--version"
    args_5: str = "--filename"
    
    mode: str = "image"
    base_url: str = "http://192.168.10.3:5050"
    model_type: str = "tiny"
    version: int = 6
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: model_type: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: version: int = int(sys.argv[sys.argv.index(args_4) + 1])
    if args_5 in sys.argv: filename: str = sys.argv[sys.argv.index(args_5) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"
    assert model_type == "small" or model_type == "tiny" or model_type == "nano", "Invalid Model Type"

    if mode == "image":
        
        assert filename in os.listdir(u.INPUT_PATH), f"{filename} not found in input directory"

        image = cv2.imread(os.path.join(u.INPUT_PATH, filename))
        payload = {
            "imageData" : u.encode_image_to_base64(image=image)
        }

        response = requests.request(method="POST", url=f"{base_url}/infer/v{version}/{model_type}", json=payload)
        if response.status_code == 200 and response.json()["statusCode"] == 200:
            cv2.rectangle(image, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
            u.show_image(image=image, title=f"{response.json()['label'].title()} : {response.json()['score']}")
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
            frameData = u.encode_image_to_base64(image=frame)
            payload = {
                "imageData" : frameData
            }       

            response = requests.request(method="POST", url=f"{base_url}/infer/v{version}/{model_type}", json=payload)
            if response.status_code == 200 and response.json()["statusCode"] == 200:
                cv2.rectangle(frame, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                cv2.putText(frame, response.json()["label"].title(), (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "ERROR", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()