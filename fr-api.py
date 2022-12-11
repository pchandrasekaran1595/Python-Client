import os
import sys
import cv2
import platform
import requests
import utils as u

def main():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--filename-1"
    args_4: str = "--filename-2"

    mode: str = "image"
    base_url: str = "http://localhost:3032"
    filename_1: str = "Test_1.jpg"
    filename_2: str = "Test_2.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: filename_1: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename_2: str = sys.argv[sys.argv.index(args_4) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"

    if mode == "image":
        assert filename_1 in os.listdir(u.INPUT_PATH), f"{filename_1} not found in input directory"
        assert filename_2 in os.listdir(u.INPUT_PATH), f"{filename_2} not found in input directory"

        image_1 = cv2.imread(os.path.join(u.INPUT_PATH, filename_1))
        image_2 = cv2.imread(os.path.join(u.INPUT_PATH, filename_2))

        payload = {
            "imageData_1" : u.encode_image_to_base64(image=image_1),
            "imageData_2" : u.encode_image_to_base64(image=image_2)
        }

        response = requests.request(method="POST", url=f"{base_url}/compare", json=payload)

        if response.status_code == 200 and response.json()["statusCode"] == 200:
            print(f"Simialrity : {float(response.json()['cosine_similarity']):.2f}")
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

        assert filename_1 in os.listdir(u.INPUT_PATH), f"{filename_1} not found in input directory"
        image = cv2.imread(os.path.join(u.INPUT_PATH, filename_1))
        imageData = u.encode_image_to_base64(image=image)

        while True:
            ret, frame = cap.read()
            if not ret: break
            payload = {
                "imageData_1" : imageData,
                "imageData_2" : u.encode_image_to_base64(image=frame)
            }  

            response = requests.request(method="POST", url=f"{base_url}/compare", json=payload)

            if response.status_code == 200 and response.json()["statusCode"] == 200:
                cv2.putText(
                    frame,
                    f"{response.json()['cosine_similarity']}",
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    1
                )
                
            
            else:
                cv2.putText(frame, "Error", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
            cv2.imshow("Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)