import os
import sys
import cv2
import requests
import utils as u


def main():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--li"
    args_4: str = "--filename-1"
    args_5: str = "--filename-2"

    mode: str = "remove"
    base_url: str = "http://localhost:3030"
    lightweight: bool = False
    filename_1: str = "Test_1.jpg"
    filename_2: str = "Test_2.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: lightweight = True
    if args_4 in sys.argv: filename_1: str = sys.argv[sys.argv.index(args_4) + 1]
    if args_5 in sys.argv: filename_2: str = sys.argv[sys.argv.index(args_5) + 1]

    if not lightweight:
        url: str = base_url + f"/{mode}"
    else:
        url: str = base_url + f"/{mode}" + "/li"

    assert filename_1 in os.listdir(u.INPUT_PATH), f"{filename_1} not found in input directory"
            
    image = cv2.imread(os.path.join(u.INPUT_PATH, filename_1))

    payload = {
        "imageData_1" : u.encode_image_to_base64(image=image)
    }

    if mode == "remove":
        response = requests.request(method="POST", url=url, json=payload)
        if response.status_code == 200 and response.json()["statusCode"] == 200:
            u.show_images(
                image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                image_2=cv2.cvtColor(src=u.decode_image(response.json()["bglessImageData"]), code=cv2.COLOR_BGR2RGB),
                cmap_1="gnuplot2",
                cmap_2="gnuplot2",
                title_1="Original",
                title_2="BG Removed Image",
            )
        else:
            print(f"Error {response.status_code} : {response.reason}")
    
    elif mode == "replace":
        assert filename_2 in os.listdir(u.INPUT_PATH), f"{filename_2} not found in input directory"

        image_2 = cv2.imread(os.path.join(u.INPUT_PATH, "Test_2.jpg"))

        payload = {
            "imageData_1" : u.encode_image_to_base64(image=image),
            "imageData_2" : u.encode_image_to_base64(image=image_2)
        }

        response = requests.request(method="POST", url=url, json=payload)

        if response.status_code == 200 and response.json()["statusCode"] == 200:
            u.show_images(
                image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                image_2=cv2.cvtColor(src=u.decode_image(response.json()["bgreplaceImageData"]), code=cv2.COLOR_BGR2RGB),
                cmap_1="gnuplot2",
                cmap_2="gnuplot2",
                title_1="Original",
                title_2="BG Replaced Image",
            )
        else:
            print(f"Error {response.status_code} : {response.reason}")


if __name__ == "__main__":
    sys.exit(main() or 0)
