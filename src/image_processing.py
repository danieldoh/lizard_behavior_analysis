import cv2
from tqdm import tqdm
import numpy as np

def undistorted_video(video_name):
    #image_path = "/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection/image/img051.png"
    #img = cv2.imread(image_path)
    video_path = f"./videos/{video_name}.mp4"
    output_video_path = f"./videos/{video_name}_ud.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
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


    mtx = np.array([[4.12738636e+03, 0.00000000e+00, 1.00139447e+03],
                    [0.00000000e+00, 4.01714435e+03, 4.20501594e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)

    dist = np.array([[-6.09079560e+00, 6.54263523e+01, 6.04755147e-02, -5.59706492e-02, -3.75578050e+02]], dtype=np.float64)

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
