import os
import sys
import cv2
import platform
import requests
import utils as u


def main():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--model"
    args_4: str = "--filename"
    
    mode: str = "image"
    base_url: str = "http://localhost:3060"
    model: str = "face"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: model: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename: str = sys.argv[sys.argv.index(args_4) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"
    assert model == "face" or model == "eye", "Invalid Model Type"

    if mode == "image":
        assert filename in os.listdir(u.INPUT_PATH), f"{filename} not found in input directory"
        
        image = cv2.imread(os.path.join(u.INPUT_PATH, filename))
        payload = {
            "imageData" : u.encode_image_to_base64(image=image)
        }

        if model == "face":
            response = requests.request(method="POST", url=f"{base_url}/detect/face", json=payload)
            if response.status_code == 200 and response.json()["statusText"] == 200:
                face_detections = response.json()["face_detections"]
                u.draw_detections(image=image, face_detections=face_detections)
                u.show_image(image=image)   
            else:
                print(f"Error {response.status_code} : {response.reason}")
        
        else:
            response = requests.request(method="POST", url=f"{base_url}/detect/eye", json=payload)
            if response.status_code == 200:
                face_detections = response.json()["face_detections"]
                eye_detections  = response.json()["eye_detections"]
                u.draw_detections(image=image, face_detections=face_detections, eye_detections=eye_detections)
                u.show_image(image=image)  
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
            if not ret: continue
            frameData = u.encode_image_to_base64(image=frame)
            payload = {
                "imageData" : frameData
            }       

            if model == "face":
                response = requests.request(method="POST", url=f"{base_url}/detect/face", json=payload)
                if response.status_code == 200 and response.json()["statusText"] == 200:
                    face_detections = response.json()["face_detections"]
                    u.draw_detections(image=frame, face_detections=face_detections)
                else:
                    cv2.putText(frame, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            else:
                response = requests.request(method="POST", url=f"{base_url}/detect/eye", json=payload)
                if response.status_code == 200:
                    face_detections = response.json()["face_detections"]
                    eye_detections  = response.json()["eye_detections"]
                    u.draw_detections(image=frame, face_detections=face_detections, eye_detections=eye_detections)
                else:
                    cv2.putText(frame, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main() or 0)