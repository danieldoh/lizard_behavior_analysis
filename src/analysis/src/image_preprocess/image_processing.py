import cv2
from tqdm import tqdm
import numpy as np

def undistorted_video(video_path):
    #image_path = "/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection/image/img051.png"
    #img = cv2.imread(image_path)
    print(video_path)
    output_video_path = video_path.split(".mp4")[0] + "_undistorted.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video file.2")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("frame number: ", frame_count)
    print("fps: ", fps)

    # Create VideoWriter object for the undistorted video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


    mtx = np.array([[4.68391167e+03, 0.00000000e+00, 1.11786072e+03],
                    [0.00000000e+00, 4.99791593e+03, 3.80250196e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)

    dist = np.array([[-1.01626151e+01, 1.43823283e+02, 9.73250120e-02, 6.71661673e-03, -7.78147789e+02]], dtype=np.float64)

    #mtx[0,2] -= 50
    #mtx[1,2] -= 10

    with tqdm(total=frame_count, desc='Generating video') as pbar:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: cap.read() problem")
                break

            undistorted_frame = cv2.undistort(frame, mtx, dist)

            out.write(undistorted_frame)

            pbar.update(1)

    cap.release()

    return output_video_path, frame_count, fps

if __name__ == "__main__":
    video_path = "../../image/2.mp4"
    new_video = undistorted_video(video_path)

    #cap = cv2.VideoCapture(video_path)
    #print(cap.get(cv2.CAP_PROP_FPS))
    #print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
