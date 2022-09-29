"""
Object Measurement Project: Main file

Project idea from:
https://pysource.com/2021/05/28/measure-size-of-an-object-with-opencv-aruco-marker-and-python/
"""


import time
from object_detector_module import *


# --- User defined parameters ----------------------------------------------------
# Set Capture path
CAPTURE_PATH = "test_videos/20220906_010113.mp4"  # video file
# CAPTURE_PATH = 0  # webcam

# Set capture size/resolution
CAP_SIZE = (1280, 720)

# Set denoise type: 'GaussianBlur', 'medianBlur', 'bilateralFilter', 'fastNlMeans'
DENOISE_TYPE = 'bilateralFilter'

# Set edge mask type: 'adaptiveThreshold', 'Canny'
MASK_TYPE = 'Canny'

# --- Show test info ---
TEST_INFO = False
# --------------------------------------------------------------------------------


def main():
    # --- Main parameters ---

    # Video
    cap = cv2.VideoCapture(CAPTURE_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])

    # Load Aruco detector (5x5 cm)
    aruco_parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

    # Video start time
    ptime = 0

    while True:
        # Reading capture frame
        ret, img = cap.read()
        # If frame is read correctly ret is True
        if not ret:
            print("Stream end. Exiting ...")
            break

        # --- Aruco marker ---
        corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_parameters)
        # If aruco marker detected
        if corners:
            # Draw polygon around the marker
            int_corners = np.int0(corners)

            # Aruco Perimeter
            aruco_perimeter = cv2.arcLength(int_corners[0], True)

            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 20

            # --- Pre-process image ---
            gray_img = pre_process_image(img, denoise_type=DENOISE_TYPE)

            # --- Detect contours in image ---
            contours = detect_contours(gray_img, mask_type=MASK_TYPE, test=TEST_INFO)

            # Draw polygon around the marker
            cv2.polylines(img, int_corners, True, (100, 255, 0), 5)

            # Number of objects
            cv2.putText(img,
                        f"Objects: {len(contours)}",
                        (int(10), int(25)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            # --- Draw object boundaries/info ---
            for cnt in contours:
                # Get rect
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), _ = rect

                # Get Width and Height of the Objects by applying the Ratio pixel to cm
                object_width = w / pixel_cm_ratio
                object_height = h / pixel_cm_ratio

                # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Center of object
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                # Width Info
                cv2.polylines(img, [box], True, (255, 0, 0), 2)
                cv2.putText(img,
                            f"Width {round(object_width, 1)} cm",
                            (int(x - 100), int(y)),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 127, 0), 2)
                # Heigth Info
                cv2.putText(img,
                            f"Height {round(object_height, 1)} cm",
                            (int(x - 100), int(y + 30)),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 127, 0), 2)

            # Draw Marker name
            cv2.putText(img,
                        "5x5 Marker",
                        (int_corners[0][0][0] + [10, 10]),
                        cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 0), 2)

            # Frame end time
            ctime = time.time()
            # Fps
            fps = 1 / (ctime - ptime)
            # New frame start time
            ptime = ctime

            cv2.putText(img,
                        f"FPS: {int(fps)}", (10, 60),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # If aruco marker NOT detected
        else:
            cv2.putText(img,
                        "Aruco Marker not found", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        # Press "esc" to quit
        if key == 27:
            filename = f'test image {DENOISE_TYPE} {MASK_TYPE}.jpg'
            cv2.imwrite(filename, img)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
