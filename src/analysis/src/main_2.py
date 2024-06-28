# This is the main, which reads the data and export to other function.
# This src is mainly to analyze the behavior of lizards, such as speed, movement, and direction.

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from image_preprocess.image_processing import undistorted_video
from get_data_csv import read_joint_csv

pix_value = []

def mouse_callback(event, x, y, flags, params):
    if event == 2:
        global pix_value
        pix_value.append([x,y])
        print([x,y])

def calculate_pixel(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', mouse_callback)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pix_length_x = pix_value[1][0] - pix_value[0][0] # x
    pix_length_y = pix_value[1][1] - pix_value[0][1] # y
    print(pix_value)
    print("X pixel length: ", pix_length_x)
    print("Y pixel length: ", pix_length_y)
    #real_length = 0.0486
    real_length_x = 0.4131
    real_length_y = 0.4131
    ratio_x = real_length_x / pix_length_x
    ratio_y = real_length_y / pix_length_y
    return ratio_x, ratio_y

def undistort_video(input_video_path):
    #input_video_path = input("Type video name (without .mp4): ")
    input_video_path = "/Users/doh/HJ/Research/lizard_joints/lizard-doh-2024-01-22/videos/" + input_video_path + ".mp4"
    output_video_path, frame_number, fps = undistorted_video(input_video_path)
    print(f"Generated undistorted video. Saved at {output_video_path}")

    return frame_number, fps

def joint_csv():
    csv_path = input("Type csv path (without .csv extension): ")
    # [M3 path]
    #csv_path = "5_32DLC_resnet50_lizardJan22shuffle4_24000_el"
    #csv_path = "/Users/hdoh/HJ/Research/prof_byeon/lizard/videos/" + csv_path + ".csv"
    # [M1 path]
    #csv_path = "10_52DLC_resnet50_lizardJan22shuffle4_24000_el"
    #csv_path = "/Users/doh/HJ/Research/lizard_joints/lizard-doh-2024-01-22/videos/" + csv_path + ".csv"

    # {name of joints: [[x coordinate list], [y coordinate list]]}
    csv_path = "../../data/" + csv_path + ".csv"
    lizard_b_pos, lizard_y_pos, frame_num = read_joint_csv(csv_path)
    return lizard_b_pos, lizard_y_pos, frame_num

def process_data(joints_pos, joint_name, fps):
    x_array = np.array(joints_pos[joint_name][0])
    y_array = np.array(joints_pos[joint_name][1])

    # remove "NaN" from the array
    x_array_fps = []
    y_array_fps = []
    for idx, (x, y) in enumerate(zip(x_array, y_array)):
        if x.strip().lower() == 'nan' or x == 0:
            x_array[idx] = float(x_array[idx-1])
        else:
            x_array[idx] = float(x_array[idx])

        if y.strip().lower() == 'nan' or x == 0:
            y_array[idx] = float(y_array[idx-1])
        else:
            y_array[idx] = float(y_array[idx])

        if idx % fps == 0:
            x_array_fps.append(x_array[idx])
            y_array_fps.append(y_array[idx])

    x_array = x_array.astype(float)
    y_array = y_array.astype(float)

    x_array_fps = np.array(x_array_fps).astype(float)
    y_array_fps = np.array(y_array_fps).astype(float)

    return x_array, y_array, x_array_fps, y_array_fps

def calculate_speed(pix_ratio_m, x_array, y_array, x_array_fps, y_array_fps, frame_number, fps=14.0):

    distances = np.sqrt(np.diff(x_array_fps * pix_ratio_m)**2 + np.diff(y_array_fps * pix_ratio_m)**2)
    #distances =  pix_ratio_m * distances

    #frame_rate = fps
    #frame_rate = 15
    #time_interval = 1 / frame_rate

    speeds = distances
    video_length = int(frame_number / fps)

    time_interval_list = [t for t in range(1, video_length+1)]

    #total_distance = calculate_total_distance(x_array, y_array, pix_ratio_m)
    average_speed = sum(speeds) / video_length

    return speeds, distances, time_interval_list, average_speed

def draw_speed_plot(speeds, time_interval):
    plt.plot(time_interval, speeds, marker='o', linestyle='-', markersize=1)

    plt.xlabel('Time Interval (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed')

    plt.show()

def draw_heatmap(x_array, y_array, pix_ratio_m):
    heatmap, x_edges, y_edges = np.histogram2d(pix_ratio_m * x_array, pix_ratio_m * y_array, bins=(40, 30))

    plt.imshow(heatmap, extent=[258 * pix_ratio_m, 1654 * pix_ratio_m, 108 * pix_ratio_m, 1079 * pix_ratio_m], cmap='viridis')
    plt.colorbar(label='count')

    plt.xlabel('width (m)')
    plt.ylabel('length (m)')
    plt.title('lizard positions (heatmap)')

    plt.show()

def draw_scatter_plot(x_array, y_array, pix_ratio_m, title):
    # movement scatter plot
    #draw_scatter_check = input("Draw scatter plot [y/n]: ")
    x_array_m = x_array * pix_ratio_m
    y_array_m = y_array * pix_ratio_m
    print(x_array_m, y_array_m)

    plt.scatter(x_array_m, y_array_m, c='red', marker='o', label='Lizard Positions', s=0.01)
    # should be fixed when the undistorted video is evaluated.
    plt.xlim(0, 0.695)
    plt.ylim(0.470, 0)
    plt.xlabel('Width (m)')
    plt.ylabel('Length (m)')
    plt.title(f'{title} Positions (Direction)')

    for i in range(len(x_array_m) - 1):
        dx = x_array_m[i+1] - x_array_m[i]
        dy = y_array_m[i+1] - y_array_m[i]

        plt.arrow(x_array_m[i], y_array_m[i], dx, dy, head_width=0.005, head_length=0.005, fc='black', ec='black')

    plt.tight_layout()

    plt.show()

def draw_direction(x_array, y_array, pix_ratio_m):
    x_array_m = x_array * pix_ratio_m
    y_array_m = y_array * pix_ratio_m
    plt.scatter(x_array_m, y_array_m, c='red', marker='o', label='Lizard Positions', s=0.01)

    # should be fixed when the undistorted video is evaluated.
    plt.xlim(258 * pix_ratio_m, 1654 * pix_ratio_m)
    plt.ylim(108 * pix_ratio_m, 1079 * pix_ratio_m)
    plt.xlabel('Width (m)')
    plt.ylabel('Length (m)')
    plt.title('Lizard Positions')

    plt.gca().invert_yaxis()

    for i in range(len(x_array_m) - 1):
        dx = x_array_m[i+1] - x_array_m[i]
        dy = y_array_m[i+1] - y_array_m[i]

        plt.arrow(x_array_m[i], y_array_m[i], dx, dy, head_width=0.005, head_length=0.005, fc='black', ec='black')

    plt.show()

def merge_heatmap_direction(x_array, y_array, pix_ratio_m):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    x_array_m = x_array * pix_ratio_m
    y_array_m = y_array * pix_ratio_m
    heatmap, x_edges, y_edges = np.histogram2d(x_array_m, y_array_m, bins=(40, 30))

    im = ax[0].imshow(heatmap, extent=[0.0, 0.695, 0, 0.470], cmap='viridis')
    fig.colorbar(im, ax=ax[0], label='count')
    ax[0].set_xlabel('Width (m)')
    ax[0].set_ylabel('Length (m)')
    ax[0].invert_yaxis()
    ax[0].set_title('Lizard Positions (Heatmap)')

    ax[1].scatter(x_array_m, y_array_m, c='red', marker='o', label='Lizard Positions', s=0.01)
    # should be fixed when the undistorted video is evaluated.
    ax[1].set_xlim(0, 0.695)
    ax[1].set_ylim(0.470, 0)
    ax[1].set_xlabel('Width (m)')
    ax[1].set_ylabel('Length (m)')
    ax[1].set_title('Lizard Positions (Direction)')

    for i in range(len(x_array_m) - 1):
        dx = x_array_m[i+1] - x_array_m[i]
        dy = y_array_m[i+1] - y_array_m[i]

        ax[1].arrow(x_array_m[i], y_array_m[i], dx, dy, head_width=0.005, head_length=0.005, fc='black', ec='black')

    plt.tight_layout()

    plt.show()

def calculate_total_distance(x_array, y_array, pix_ratio_m):
    #distances = np.sqrt(np.diff(pix_ratio_m * x_array)**2 + np.diff(pix_ratio_m * y_array)**2)
    distances = np.sqrt(np.diff(x_array)**2 + np.diff(y_array)**2)
    #print(f"total distances: {sum(distances)}m")
    #print(f"total distances: {sum(pix_ratio_m * distances)}m")
    return sum(pix_ratio_m * distances)

# vector1 : foot to leg joint
# vector2: leg joint to foot
def calculate_degree(vector1, vector2):
    degree_list = []
    for vec1, vec2 in zip(vector1, vector2):
        dot_product = np.dot(vec1, vec2)
        magnitude_1 = np.linalg.norm(vec1)
        magnitude_2 = np.linalg.norm(vec2)

        cosine_theta = dot_product / (magnitude_1 * magnitude_2)

        angle_radians = np.arccos(cosine_theta)

        angle_degrees = np.degrees(angle_radians)

        degree_list.append(angle_degrees)

    return degree_list

def joint_degree(joints_pos, frame_number, fps):
    legs_list = ['lf_leg1', 'lf_leg2', 'body1', 'rf_leg1', 'rf_leg2', 'lb_leg1', 'lb_leg2', 'body3', 'rb_leg1', 'rb_leg2']
    joint_dict = {}
    vector_list = []

    for i, joint in enumerate(legs_list):
        x_array, y_array, _, _ = process_data(joints_pos, joint, fps)
        joint_dict[joint] = np.column_stack((x_array, y_array))

    vector_list.append(joint_dict['lf_leg2'] - joint_dict['lf_leg1'])
    vector_list.append(joint_dict['body1'] - joint_dict['lf_leg2'])

    vector_list.append(joint_dict['rf_leg2'] - joint_dict['rf_leg1'])
    vector_list.append(joint_dict['body1'] - joint_dict['rf_leg2'])

    vector_list.append(joint_dict['lb_leg2'] - joint_dict['lb_leg1'])
    vector_list.append(joint_dict['body3'] - joint_dict['lb_leg2'])

    vector_list.append(joint_dict['rb_leg2'] - joint_dict['rb_leg1'])
    vector_list.append(joint_dict['body3'] - joint_dict['rb_leg2'])

    left_front_degrees = calculate_degree(vector_list[0], vector_list[1])
    right_front_degrees = calculate_degree(vector_list[2], vector_list[3])
    left_back_degrees = calculate_degree(vector_list[4], vector_list[5])
    right_back_degrees = calculate_degree(vector_list[6], vector_list[7])

    video_length = frame_number / fps

    time_interval = np.linspace(0, video_length, frame_number)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0, 0].plot(time_interval, left_front_degrees)
    ax[0, 0].set_title('Left Front Leg Angle')
    ax[0, 0].set_xlabel('Time (s)')
    ax[0, 0].set_ylabel('Angle (deg)')

    ax[0, 1].plot(time_interval, right_front_degrees)
    ax[0, 1].set_title('Right Front Leg Angle')
    ax[0, 1].set_xlabel('Time (s)')
    ax[0, 1].set_ylabel('Angle (deg)')

    ax[1, 0].plot(time_interval, left_back_degrees)
    ax[1, 0].set_title('Left Back Leg Angle')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 0].set_ylabel('Angle (deg)')

    ax[1, 1].plot(time_interval, right_back_degrees)
    ax[1, 1].set_title('Right Back Leg Angle')
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_ylabel('Angle (deg)')

    plt.tight_layout()

    plt.show()

    '''sa_lf_rb = ((45 - np.arctan2(left_front_degrees, right_back_degrees)) / 90) * 100

    sa_rf_lb = ((45 - np.arctan2(right_front_degrees, left_back_degrees)) / 90) * 100

    print("Left Front - Right Back ASA:", np.mean(sa_lf_rb))
    print("Right Front - Left Back ASA:", np.mean(sa_rf_lb))'''


def joint_speed(pix_ratio_m, joints_pos, frame_number, fps, title):
    legs_list = ['lf_leg1', 'lf_leg2', 'rf_leg1', 'rf_leg2', 'lb_leg1', 'lb_leg2', 'rb_leg1', 'rb_leg2']
    joint_speed = {}
    joint_average_speed = {}
    time_interval = None

    for i, joint in enumerate(legs_list):
        x_array, y_array, x_array_fps, y_array_fps = process_data(joints_pos, joint, fps)
        speeds, _, time_interval, average_speed = calculate_speed(pix_ratio_m, x_array, y_array, x_array_fps, y_array_fps, frame_number, fps)
        joint_speed[joint] = speeds
        joint_average_speed[joint] = average_speed

    #frame_number = 204
    #fps = 14.881458966565349
    #video_length = frame_number / fps

    #time_interval = np.linspace(0, video_length, frame_number)

    fig, ax = plt.subplots(2, 4, figsize=(10, 8))

    fig.suptitle(title, fontsize = 16)

    title_list = ['Left Front Foot','Left Front Knee',
                  'Right Front Foot','Right Front Knee',
                  'Left Back Foot','Left Back Knee',
                  'Right Back Foot','Right Back Knee']

    for i in range(4):
        ax[0, i].plot(time_interval, joint_speed[legs_list[i]], marker='o', linestyle='-', markersize=1)
        ax[0, i].set_title(title_list[i])
        ax[0, i].set_xlabel('Time (s)')
        ax[0, i].set_ylabel('Speed (m/s)')

        ax[1, i].plot(time_interval, joint_speed[legs_list[i+4]], marker='o', linestyle='-', markersize=1)
        ax[1, i].set_title(title_list[i+4])
        ax[1, i].set_xlabel('Time (s)')
        ax[1, i].set_ylabel('Speed (m/s)')

    plt.tight_layout()

    plt.show()

    for i, joint in enumerate(legs_list):
        print(f"{title_list[i]}: {joint_average_speed[joint]} m/s")

def stride_distance(pix_ratio_m, joints_pos, frame_number, fps, title):
    legs_list = ['lf_leg1', 'rf_leg1', 'lb_leg1', 'rb_leg1']
    joint_stride = {}
    time_interval = None

    for i, joint in enumerate(legs_list):
        x_array, y_array, x_array_fps, y_array_fps = process_data(joints_pos, joint, fps)
        distances = pix_ratio_m * (np.sqrt(np.diff(x_array)**2 + np.diff(y_array)**2))
        joint_stride[joint] = distances

    video_length = frame_number / fps

    time_interval = np.linspace(0, video_length, frame_number-1)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    fig.suptitle(title, fontsize=16)

    title_list = ['Left Front Foot','Right Front Foot',
                  'Left Back Foot','Right Back Foot']

    for i in range(2):
        ax[0, i].plot(time_interval, joint_stride[legs_list[i]], marker='o', linestyle='-', markersize=2)
        ax[0, i].set_title(title_list[i])
        ax[0, i].set_xlabel('Time (s)')
        ax[0, i].set_ylabel('Stride (m)')

        ax[1, i].plot(time_interval, joint_stride[legs_list[i+2]], marker='o', linestyle='-', markersize=2)
        ax[1, i].set_title(title_list[i+2])
        ax[1, i].set_xlabel('Time (s)')
        ax[1, i].set_ylabel('Stride (m)')

    plt.tight_layout()

    plt.show()

    for i, foot in enumerate(title_list):
        stride_mean = np.mean(joint_stride[legs_list[i]])
        print(f'Mean of {foot}: {stride_mean}m')

    asi_value_lf_rb = np.abs(joint_stride['lf_leg1'] - joint_stride['rb_leg1']) / np.maximum(joint_stride['lf_leg1'], joint_stride['rb_leg1'])
    asi_value_rf_lb = np.abs(joint_stride['rf_leg1'] - joint_stride['lb_leg1']) / np.maximum(joint_stride['rf_leg1'], joint_stride['lb_leg1'])

    average_asi_lf_rb = np.mean(asi_value_lf_rb)
    average_asi_rf_lb = np.mean(asi_value_rf_lb)

    print(f"Left front - right back feet ASI: {average_asi_lf_rb * 100}")
    print(f"Right front - left back feet ASI: {average_asi_rf_lb * 100}")

if __name__ == "__main__":

    # **********************************
    #pix_ratio_cm = 0.08237
    pix_ratio_m = 0.000427
    pix_ratio_cm = pix_ratio_m * 100
    joints_pos = {}
    fps = 14.0
    x_array = np.empty((0,))
    y_array = np.empty((0,))
    distances = np.empty((0,))
    speed = np.empty((0,))
    time_interval_list = []
    frame_number = 774
    # **********************************

    parser = argparse.ArgumentParser(description='get input')

    parser.add_argument("--read_csv_check", "-rc", action="store_true", help="want to read csv file?")
    parser.add_argument("--ratio_check", "-rat", action="store_true", help="want to check ratio?")
    parser.add_argument("--undistort_check", "-undt", action="store_true", help="want to undistort the video?")
    parser.add_argument("--video_path", help="Type video name (without .mp4)")
    parser.add_argument("--speed_check", "-spd", action="store_true", help="want to calculate the speed?")
    parser.add_argument("--heatmap_check", "-heat", action="store_true", help="want to plot heatmap?")
    parser.add_argument("--scatter_check", "-scat", action="store_true", help="want to plot scatter?")
    parser.add_argument("--direction_check", "-dir", action="store_true", help="want to plot direction?")
    parser.add_argument("--distance_check", "-dis", action="store_true", help="want to calculate distance?")
    parser.add_argument("--degree_check", "-deg", action="store_true", help="want to calculate degree?")
    parser.add_argument("--joint_speed_check", "-jspd", action="store_true", help="want to calculate joint speed?")
    parser.add_argument("--movement_check", "-mv", action="store_true", help="want to calculate movement?")
    parser.add_argument("--stride_check", "-str", action="store_true", help="want to calculate stride?")

    args = parser.parse_args()

    # Read csv data
    if args.read_csv_check:
        lizard_b_pos, lizard_y_pos, frame_number = joint_csv()
        joints_name = ['head', 'neck', 'body1', 'lf_leg1', 'lf_leg2', 'rf_leg1', 'rf_leg2',
                    'body2', 'body3', 'lb_leg1', 'lb_leg2', 'rb_leg1', 'rb_leg2', 'mid_tail', 'tail']
        print("These are joints name: ", joints_name)

    # Ratio Check
    if args.ratio_check:
        # [M1 path]
        #chessboard_img = "/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection/image/undistored_chessboard.png"
        #chessboard_img = "/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection/image/chessboard.jpg"
        #cap = cv2.VideoCapture('/Users/doh/HJ/Research/lizard_joints/lizard-doh-2024-01-22/videos/10_52DLC_resnet50_lizardJan22shuffle4_24000_el_id_labeled.mp4')
        # [M3 path]
        #ret, img = cap.read()

        #chessboard_img = "/Users/hdoh/HJ/Research/prof_byeon/lizard_detection/image/undistored_chessboard.png"

        img = "../../videos/tapo_undistorted.png"
        img = cv2.imread(img)
        pix_ratio_m_x, pix_ratio_m_y = calculate_pixel(img)
        pix_ratio_cm_x = pix_ratio_m_x * 100
        pix_ratio_cm_y = pix_ratio_m_y * 100
        print(f"X Ratio is {pix_ratio_m_x}m = {pix_ratio_cm_x}cm")
        print(f"Y Ratio is {pix_ratio_m_y}m = {pix_ratio_cm_y}cm")
        #cap.release()
    else:
        print(f"Default Ratio is {pix_ratio_m}m = {pix_ratio_cm}cm")

    # Create undistored video
    if args.undistort_check:
        if not args.video_path:
            print("Error: no video path, include --video_path when you run the code.")
        frame_number, fps = undistort_video(args.video_path)

    '''if args.movement_check:
        b_x_array, b_y_array, _, _ = process_data(lizard_b_pos, "body2", fps)
        merge_heatmap_direction(b_x_array, b_y_array, pix_ratio_m)

        y_x_array, y_y_array, _, _ = process_data(lizard_y_pos, "body2", fps)
        merge_heatmap_direction(y_x_array, y_y_array, pix_ratio_m)'''

    if args.speed_check:
        if lizard_b_pos != {}:
            b_x_array, b_y_array, b_x_array_fps, b_y_array_fps = process_data(lizard_b_pos, "body2", fps)
            speeds, distances, time_interval, average_speed = calculate_speed(pix_ratio_m, b_x_array, b_y_array, b_x_array_fps, b_y_array_fps, frame_number, fps)
            print(f"Black Lizard Average Speed: {average_speed} m/s")
            draw_speed_plot(speeds, time_interval)

        if lizard_y_pos != {}:
            y_x_array, y_y_array, y_x_array_fps, y_y_array_fps = process_data(lizard_y_pos, "body2", fps)
            speeds, distances, time_interval, average_speed = calculate_speed(pix_ratio_m, y_x_array, y_y_array, y_x_array_fps, y_y_array_fps, frame_number, fps)
            print(f"Yellow Lizard Average Speed: {average_speed} m/s")
            draw_speed_plot(speeds, time_interval)

    # draw heatmap of movement
    if args.heatmap_check:
        x_array, y_array, x_array_fps, y_array_fps = process_data(joints_pos, "body2", fps)
        draw_heatmap(x_array, y_array, pix_ratio_m)

    # draw scatter plot of movement
    if args.movement_check:
        if lizard_b_pos != {}:
            b_x_array, b_y_array, _, _ = process_data(lizard_b_pos, "body2", fps)
            draw_scatter_plot(b_x_array, b_y_array, pix_ratio_m, "Black-Dot Lizard")

        if lizard_y_pos != {}:
            y_x_array, y_y_array, _, _ = process_data(lizard_y_pos, "body2", fps)
            draw_scatter_plot(y_x_array, y_y_array, pix_ratio_m, "Yellow-Dot Lizard")

    # draw direction plot of movement
    if args.direction_check:
        draw_direction(x_array, y_array, pix_ratio_m)

    # calculate the total distance of movement
    if args.distance_check:
        if lizard_b_pos != {}:
            b_x_array, b_y_array, _, _ = process_data(lizard_b_pos, "body2", fps)
            total_distance = calculate_total_distance(b_x_array, b_y_array, pix_ratio_m)
            print(f"Black-Dot Lizard total distances: {total_distance} m")

        if lizard_y_pos != {}:
            y_x_array, y_y_array, _, _ = process_data(lizard_y_pos, "body2", fps)
            total_distance = calculate_total_distance(b_x_array, b_y_array, pix_ratio_m)
            print(f"Yellow-Dot Lizard total distances: {total_distance} m")

    # calculate the joint degree
    if args.degree_check:
        if lizard_b_pos != {}:
            joint_degree(lizard_b_pos, frame_number, fps)

        if lizard_y_pos != {}:
            joint_degree(lizard_y_pos, frame_number, fps)

    # calculate the joint speed
    if args.joint_speed_check:
        if lizard_b_pos != {}:
            joint_speed(pix_ratio_m, lizard_b_pos, frame_number, fps, "Black-Dot Lizard Joint Speed")

        if lizard_y_pos != {}:
            joint_speed(pix_ratio_m, lizard_y_pos, frame_number, fps, "Yellow-Dot Lizard Joint Speed")

    # calculate the stride
    if args.stride_check:

        if lizard_b_pos != {}:
            stride_distance(pix_ratio_m, lizard_b_pos, frame_number, fps, "Black-Dot Lizard Stride Distance")

        print(type(lizard_y_pos))
        print(lizard_y_pos)
        if lizard_y_pos != {}:
            stride_distance(pix_ratio_m, lizard_y_pos, frame_number, fps, "Yellow-Dot Lizard Stride Distance")
