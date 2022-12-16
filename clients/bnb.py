import os
import sys
import cv2
import requests
import utils as u


def main():

    args_1: str = "--base-url"
    args_2: str = "--filename"
    args_3: str = "--size"
    
    base_url: str = "http://localhost:3046"
    filename: str = "Test_1.jpg"
    size: int = 224

    if args_1 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: filename: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: size: int = int(sys.argv[sys.argv.index(args_3) + 1])

    assert filename in os.listdir(u.INPUT_PATH), f"{filename} not found in input directory"
    assert size == 224 or size == 384, f"{size} is invalid. Supported sizes are 224 and 384"

    image = cv2.imread(os.path.join(u.INPUT_PATH, filename))
    # image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=5)

    payload = {
        "imageData" : u.encode_image_to_base64(image=image)
    }

    response = requests.request(method="POST", url=f"{base_url}/infer/{size}", json=payload)
    if response.status_code == 200 and response.json()["statusCode"] == 200:
        print(response.json()["probability"])
    else:
        print(f"Error {response.status_code} : {response.reason}")
    

if __name__ == "__main__":
    sys.exit(main() or 0)