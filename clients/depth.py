import os
import sys
import cv2
import platform
import requests
import utils as u


def main():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--filename"
    
    mode: str = "image"
    base_url: str = "http://localhost:9090"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: filename: str = sys.argv[sys.argv.index(args_3) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"
    
    url = f"{base_url}/infer"

    if mode == "image":

        assert filename in os.listdir(u.INPUT_PATH), f"{filename} not found in input directory"

        image = cv2.imread(os.path.join(u.INPUT_PATH, filename))

        payload = {
            "imageData" : u.encode_image_to_base64(image=image)
        }

        response = requests.post(url=url, json=payload)
        if response.status_code == 200 and response.json()["statusCode"] == 200:
            u.show_images(
                image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                image_2=u.decode_image(response.json()["imageData"]),
                cmap_1="gnuplot2",
                cmap_2="gnuplot2",
                title_1="Original",
                title_2="Depth Image",
            )
        else:
            print(f"Error {response.status_code} : {response.reason}")
    
    elif mode == "realtime":

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

            response = requests.post(url=url, json=payload)
            if response.status_code == 200 and response.json()["statusCode"] == 200:
                frame = cv2.cvtColor(src=u.decode_image(response.json()["imageData"]), code=cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(frame, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Depth Inference", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Invalid Mode")


if __name__ == "__main__":
    sys.exit(main() or 0)
