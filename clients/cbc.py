import os
import sys
import cv2
import requests
import utils as u


def main():

    args_1: str = "--base-url"
    args_2: str = "--filename"
    
    base_url: str = "http://localhost:3052"
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: filename: str = sys.argv[sys.argv.index(args_2) + 1]

    assert filename in os.listdir(u.INPUT_PATH), f"{filename} not found in input directory"

    image = cv2.imread(os.path.join(u.INPUT_PATH, filename))

    payload = {
        "imageData" : u.encode_image_to_base64(image=image)
    }

    response = requests.request(method="POST", url=f"{base_url}/infer", json=payload)
    if response.status_code == 200 and response.json()["statusCode"] == 200:
        print(response.json()["label"])
    else:
        print(f"Error {response.status_code} : {response.reason}")
    

if __name__ == "__main__":
    sys.exit(main() or 0)