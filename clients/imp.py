import os
import sys
import cv2
import requests
import utils as u

from typing import Union


def main():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--process"
    args_4: str = "--filename"

    mode: str = "image"
    base_url: str = "http://localhost:6060"
    process: Union[str, None] = None
    filename: str = "Test_1.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: process: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename: str = sys.argv[sys.argv.index(args_4) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"

# ---

    if process == "blur":
        blur_type = sys.argv[sys.argv.index(args_3) + 2]
        if blur_type == "gauss":
            setup = sys.argv[sys.argv.index(args_3) + 3] + ","
        
            gaussian_blur_kernel_size: str = setup.split(",")[0]
            gaussian_blur_sigmaX: str = setup.split(",")[1]

            if int(gaussian_blur_kernel_size) == 1: gaussian_blur_kernel_size = None
            elif int(gaussian_blur_kernel_size) % 2 == 0: int(gaussian_blur_kernel_size) + 1
            else: gaussian_blur_kernel_size = int(gaussian_blur_kernel_size)

            if gaussian_blur_sigmaX != "": gaussian_blur_sigmaX = float(gaussian_blur_sigmaX)
            else: gaussian_blur_sigmaX = None

            payload: dict = {
                "gaussian_blur_kernel_size" : gaussian_blur_kernel_size,
                "gaussian_blur_sigmaX" : gaussian_blur_sigmaX,
            }
        
        elif blur_type == "average":
            payload: dict = {
                "average_blur_kernel_size" : int(sys.argv[sys.argv.index(args_3) + 3])
            }

        elif blur_type == "median":
            payload: dict = {
                "median_blur_kernel_size" : int(sys.argv[sys.argv.index(args_3) + 3]),
            }
    
        url: str = f"{base_url}/{process}/{blur_type}"

# ---

    elif process == "contrast":
        contrast_type = sys.argv[sys.argv.index(args_3) + 2]
        if contrast_type == "gamma":
            payload: dict = {
                "gamma" : float(sys.argv[sys.argv.index(args_3) + 3])
            }

        elif contrast_type == "linear":
            payload: dict = {
                "linear" : int(sys.argv[sys.argv.index(args_3) + 3]),
            }
        
        url: str = f"{base_url}/{process}/{contrast_type}"

# ---

    elif process == "equalize":
        equalize_type = sys.argv[sys.argv.index(args_3) + 2]
        if equalize_type == "clahe":
            setup = sys.argv[sys.argv.index(args_3) + 3] + ","

            clipLimit = float(setup.split(",")[0])
            tileGridSize = setup.split(",")[1]

            if tileGridSize != "": tileGridSize = int(tileGridSize)
            else: tileGridSize = None
        
            payload: dict = {
                "clipLimit" : clipLimit,
                "tileGridSize" : tileGridSize,
            }
        
        elif equalize_type == "histogram":
            payload: dict = {
                
            }
        
        url: str = f"{base_url}/{process}/{equalize_type}"

# ---

    if mode == "image":
        assert filename in os.listdir(u.INPUT_PATH), f"{filename} not found in input directory"

        image = cv2.imread(os.path.join(u.INPUT_PATH, filename))

        payload["imageData"] = u.encode_image_to_base64(image=image)

        response = requests.request(method="POST", url=url, json=payload)
        if response.status_code == 200 and response.json()["statusCode"] == 200:
            u.show_image(image=u.decode_image(response.json()["imageData"]), title=f"{process.title()} Processed")

    else:
        pass


if __name__ == "__main__":
    sys.exit(main() or 0)