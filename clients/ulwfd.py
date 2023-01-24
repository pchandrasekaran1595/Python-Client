import os
import sys
import cv2
import requests
import utils as u


def main():

    args_1: str = "--base-url"
    args_2: str = "--filename"
    args_3: str = "--size"
    
    base_url: str = "http://localhost:7070"
    filename: str = "Test_1.jpg"
    size: int = 320

    if args_1 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: filename: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: size: int = int(sys.argv[sys.argv.index(args_3) + 1])

    assert filename in os.listdir(u.INPUT_PATH), f"{filename} not found in input directory"

    image = cv2.imread(os.path.join(u.INPUT_PATH, filename))

    payload = {
        "imageData" : u.encode_image_to_base64(image=image)
    }

    response = requests.request(method="POST", url=f"{base_url}/infer/{size}", json=payload)
    if response.status_code == 200 and response.json()["statusCode"] == 200:
        x1 = response.json()["x1"]
        y1 = response.json()["y1"]
        x2 = response.json()["x2"]
        y2 = response.json()["y2"]
        # print(x1, y1, x2, y2)
        cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)
        u.show_image(image)
    else:
        print(f"Error {response.status_code} : {response.reason}")
    

if __name__ == "__main__":
    sys.exit(main() or 0)