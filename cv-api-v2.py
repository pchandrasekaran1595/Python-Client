import os
import sys
import cv2
import requests
import utils as u


def main():

    args_1: str = "--mode"
    args_2: str = "--base-url"
    args_3: str = "--model"
    args_4: str = "--filename-1"
    args_5: str = "--filename-2"

    mode: str = "image"
    base_url: str = "http://localhost:4040"
    model: str = "classify"
    filename_1: str = "Test_1.jpg"
    filename_2: str = "Test_2.jpg"

    if args_1 in sys.argv: mode: str = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: base_url: str = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: model: str = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: filename_1: str = sys.argv[sys.argv.index(args_4) + 1]
    if args_5 in sys.argv: filename_2: str = sys.argv[sys.argv.index(args_5) + 1]

    assert mode == "image" or mode == "realtime", "Invalid Mode"
    assert model == "classify" or \
           model == "detect" or \
           model == "segment" or \
           model == "remove" or \
           model == "replace" or \
           model == "depth" or \
           model == "face", f"{model.title()} is an invalid model type"

    if mode == "image":
        assert filename_1 in os.listdir(u.INPUT_PATH), f"{filename_1} not found in input directory"

        image = cv2.imread(os.path.join(u.INPUT_PATH, filename_1))

        if model != "replace":
            payload = {
                "imageData" : u.encode_image_to_base64(image=image)
            }

        else:
            assert filename_2 in os.listdir(u.INPUT_PATH), f"{filename_2} not found in input directory"

            image_2 = cv2.imread(os.path.join(u.INPUT_PATH, filename_2))
            payload = {
                "imageData_1" : u.encode_image_to_base64(image=image),
                "imageData_2" : u.encode_image_to_base64(image=image_2)
            }

        response = requests.request(method="POST", url=f"{base_url}/{model}", json=payload)

        if response.status_code == 200 and response.json()["statusCode"] == 200:
            if model == "classify":
                print(response.json()["label"])

            if model == "detect":
                cv2.rectangle(image, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                cv2.putText(image, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                u.show_image(image)

            if model == "segment":
                print(f"Classes Present : {response.json()['labels']}")
                image = u.decode_image(response.json()["imageData"])
                u.show_image(image)
            
            if model == "remove":
                u.show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=cv2.cvtColor(src=u.decode_image(response.json()["bglessImageData"]), code=cv2.COLOR_BGR2RGB),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="BG Removed Image",
                )
            
            if model == "replace":
                u.show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=cv2.cvtColor(src=u.decode_image(response.json()["bgreplaceImageData"]), code=cv2.COLOR_BGR2RGB),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="BG Replaced Image",
                )
            
            if model == "depth":
                u.show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=u.decode_image(response.json()["imageData"]),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="Depth Image",
                )
            
            if model == "face":
                    face_detections = response.json()["face_detections"]
                    u.draw_detections(image=image, face_detections=face_detections)
        else:
            print(f"Error {response.status_code} : {response.reason}")

    else:
        pass
        
        # if platform.system() != "Windows":
        #     cap = cv2.VideoCapture(0)
        # else:
        #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        # while True:
        #     ret, frame = cap.read()
        #     if not ret: break
        #     frameData = encode_image_to_base64(image=frame)
        #     payload = {
        #         "imageData" : frameData
        #     }       

        #     response = requests.request(method="POST", url=f"{base_url}/{model}", json=payload)

        #     if response.status_code == 200 and response.json()["statusCode"] == 200:
        #         if model == "classify":
        #             cv2.putText(frame, response.json()["label"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #         if model == "detect":
        #             cv2.rectangle(frame, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
        #             cv2.putText(frame, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #         if model == "segment":
        #             disp_frameData = response.json()["imageData"]
        #             frame = decode_image(disp_frameData)
                
        #         if model == "face":
        #             face_detections = response.json()["face_detections"]
        #             draw_detections(image=frame, face_detections=face_detections)
            
        #     else:
        #         cv2.putText(frame, "Error", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
        #     cv2.imshow("Processed", frame)
            
        #     if cv2.waitKey(1) & 0xFF == ord("q"): 
        #         break
            
        # cap.release()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)