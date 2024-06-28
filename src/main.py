import os
import sys
import time
import json
import argparse
import requests
from zipfile import ZipFile
import tarfile
from shutil import copyfile
from dataclasses import dataclass, field
from datetime import datetime

import yaml
import glob

import random
import numpy as np
import pandas as pd
import cv2

from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

#sys.path.append("./analysis/src/image_preprocess/")

@dataclass(frozen=True)
class TrainingConfig:
    DATASET_YAML:   str = "lizard-keypoints.yaml"
    MODEL:          str = "yolov8m-pose.pt"
    EPOCHS:         int = 1000
    KPT_SHAPE:    tuple = (15,3)
    PROJECT:        str = "../Lizard-Keypoints"
    NAME:           str = f"{MODEL.split('.')[0]}_{EPOCHS}_epochs"
    CLASSES_DICT:  dict = field(default_factory = lambda:{0 : "lizard_b", 1: "lizard_y"})

@dataclass(frozen=True)
class DatasetConfig:
    IMAGE_SIZE:    int   = 640
    BATCH_SIZE:    int   = 32
    CLOSE_MOSAIC:  int   = 10
    MOSAIC:        float = 0.4
    FLIP_LR:       float = 0.0

def draw_landmarks(image, landmarks, COLORS_RGB_MAP):

    radius = 5
    # Check if image width is greater than 1000 px.
    # To improve visualization.
    if (image.shape[1] > 1000):
        radius = 8

    for idx, kpt_data in enumerate(landmarks):

        loc_x, loc_y = kpt_data[:2].astype("int").tolist()
        color_id = list(COLORS_RGB_MAP[int(kpt_data[-1])].values())[0]

        cv2.circle(image,
                   (loc_x, loc_y),
                   radius,
                   color=color_id[::-1],
                   thickness=-1,
                   lineType=cv2.LINE_AA)

    return image

def draw_boxes(image, detections, class_name, score=None, color=(0,255,0)):

    font_size = 0.25 + 0.07 * min(image.shape[:2]) / 100
    font_size = max(font_size, 0.5)
    font_size = min(font_size, 0.8)
    text_offset = 3

    thickness = 2
    # Check if image width is greater than 1000 px.
    # To improve visualization.
    if (image.shape[1] > 1000):
        thickness = 10

    xmin, ymin, xmax, ymax = detections[:4].astype("int").tolist()
    conf = round(float(detections[-1]),2)
    cv2.rectangle(image,
                  (xmin, ymin),
                  (xmax, ymax),
                  color=(0,255,0),
                  thickness=thickness,
                  lineType=cv2.LINE_AA)

    if class_name == 1.0 or class_name == "lizard_y":
        class_name = "lizard_y"
    else:
        class_name = "lizard_b"

    display_text = f"{class_name}"

    if score is not None:
        display_text+=f": {score:.2f}"

    (text_width, text_height), _ = cv2.getTextSize(display_text,
                                                   cv2.FONT_HERSHEY_SIMPLEX,
                                                   font_size, 2)

    cv2.rectangle(image,
                      (xmin, ymin),
                      (xmin + text_width + text_offset, ymin - text_height - int(15 * font_size)),
                      color=color, thickness=-1)

    image = cv2.putText(
                    image,
                    display_text,
                    (xmin + text_offset, ymin - int(10 * font_size)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (0, 0, 0),
                    2, lineType=cv2.LINE_AA,
                )

    return image

def visualize_annotations(image, box_data, keypoints_data, class_name, COLORS_RGB_MAP):

    image = image.copy()

    shape_multiplier = np.array(image.shape[:2][::-1]) # (W, H).
    # Final absolute coordinates (xmin, ymin, xmax, ymax).
    box_data = box_data.reshape(1,-1)
    denorm_boxes = np.zeros_like(box_data)

    # De-normalize center coordinates from YOLO to (xmin, ymin).
    denorm_boxes[:, :2] = (shape_multiplier/2.) * (2*box_data[:,:2] - box_data[:,2:])

    # De-normalize width and height from YOLO to (xmax, ymax).
    denorm_boxes[:, 2:] = denorm_boxes[:,:2] + box_data[:,2:]*shape_multiplier

    for boxes, kpts, in zip(denorm_boxes, keypoints_data):
        # De-normalize landmark coordinates.
        kpts[:, :2]*= shape_multiplier
        image = draw_boxes(image, boxes, class_name)
        image = draw_landmarks(image, kpts, COLORS_RGB_MAP)
    return image

def data_visualization(IMG_PATH, LABEL_PATH, COLORS_RGB_MAP):
    IMAGE_FILES = os.listdir(IMG_PATH)
    NUM_LANDMARKS = 15

    num_samples = 8
    num_rows = 2
    num_cols = num_samples//num_rows

    fig, ax = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(25, 15),
        )

    random.seed(45)
    random.shuffle(IMAGE_FILES)

    for idx, (file, axis) in enumerate(zip(IMAGE_FILES[:num_samples], ax.flat)):

        image = cv2.imread(os.path.join(IMG_PATH, file))

        # Obtain the txt file for the corresponding image file.
        filename = "".join(file.split(".")[:-1])
        # Split each object instance in separate lists.
        with open(os.path.join(LABEL_PATH, filename+".txt"), "r") as file:
            label_data = [x.split() for x in file.read().strip().splitlines() if len(x)]

        label_data = np.array(label_data, dtype=np.float32)

        # YOLO BBox instances in [x-center, y-center, width, height] in normalized form.
        box_instances_0 = label_data[0,1:5]
        instance_kpts_0 = []
        kpts_data_0 = label_data[0,5:].reshape(-1, NUM_LANDMARKS, 3)
        class_name = label_data[0,0]

        for inst_kpt_0 in kpts_data_0:
            vis_ids_0 = np.where(inst_kpt_0[:, -1]>0.)[0]
            vis_kpts_0 = inst_kpt_0[vis_ids_0][:,:2]
            vis_kpts_0 = np.concatenate([vis_kpts_0, np.expand_dims(vis_ids_0, axis=-1)], axis=-1)
            instance_kpts_0.append(vis_kpts_0)

        image_ann = visualize_annotations(image, box_instances_0, instance_kpts_0, class_name, COLORS_RGB_MAP)

        if label_data.shape[0] == 2:
            instance_kpts_1 = []
            box_instances_1 = label_data[1,1:5]
            kpts_data_1 = label_data[1,5:].reshape(-1, NUM_LANDMARKS, 3)
            class_name = label_data[1,0]

            for inst_kpt_1 in kpts_data_1:
                vis_ids_1 = np.where(inst_kpt_1[:, -1]>0.)[0]
                vis_kpts_1 = inst_kpt_1[vis_ids_1][:,:2]
                vis_kpts_1 = np.concatenate([vis_kpts_1, np.expand_dims(vis_ids_1, axis=-1)], axis=-1)
                instance_kpts_1.append(vis_kpts_1)

            image_ann = visualize_annotations(image_ann, box_instances_1, instance_kpts_1, class_name, COLORS_RGB_MAP)

        axis.imshow(image_ann[...,::-1])
        axis.axis("off")


    plt.tight_layout(h_pad=4., w_pad=4.)
    plt.show();

def create_yolo_boxes_kpts(img_size, boxes, lm_kpts):

    IMG_W, IMG_H = img_size
    # Modify kpts with visibilities as 1s to 2s.
    #vis_ones = np.where(lm_kpts[:, -1] == 1.)
    #lm_kpts[vis_ones, -1] = 2.

    # Normalizing factor for bboxes and kpts.
    res_box_array = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
    res_lm_array = np.array([IMG_W, IMG_H]).reshape((-1,2))

    # Normalize landmarks in the range [0,1].
    norm_kps_per_img = lm_kpts.copy()
    #norm_kps_per_img[:, :-1] = norm_kps_per_img[:, :-1] / res_lm_array
    norm_kps_per_img_dummy = norm_kps_per_img[:-1].reshape((-1,2)) / res_lm_array
    norm_kps_per_img[:-1] = norm_kps_per_img_dummy.reshape(-1)

    # Normalize bboxes in the range [0,1].
    norm_bbox_per_img = boxes / res_box_array

    # Create bboxes coordinates to YOLO.
    # x_c, y_c = x_min + bbox_w/2. , y_min + bbox_h/2.
    yolo_boxes = norm_bbox_per_img.copy()
    yolo_boxes[:2] = norm_bbox_per_img[:2] + norm_bbox_per_img[2:]/2.

    return yolo_boxes, norm_kps_per_img

def create_yolo_txt_files(json_data, LABEL_PATH):

    images_data = json_data["images"]
    annotate_data = json_data["annotations"]
    annotate_len = len(annotate_data)

    #for i in range(0, annotate_len, 2):
    kps_dict = {}
    image_ids = []
    IMG_WIDTH, IMG_HEIGHT = images_data[0]["width"], images_data[0]["height"]
    for annot in annotate_data:
        IMAGE_ID = annot["image_id"]
        CLASS_ID = annot["category_id"]
        image_ids.append(IMAGE_ID)

        landmark_kpts  = np.nan_to_num(np.array(annot["keypoints"], dtype=np.float32))
        landmarks_bboxes = np.array(annot["bbox"], dtype=np.float32)

        bboxes_yolo, kpts_yolo = create_yolo_boxes_kpts(
                                            (IMG_WIDTH, IMG_HEIGHT),
                                            landmarks_bboxes,
                                            landmark_kpts)

        x_c_norm, y_c_norm, box_width_norm, box_height_norm = round(bboxes_yolo[0],5),\
                                                              round(bboxes_yolo[1],5),\
                                                              round(bboxes_yolo[2],5),\
                                                              round(bboxes_yolo[3],5),\

        kps_flattend = [round(ele,5) for ele in kpts_yolo.flatten().tolist()]
        if IMAGE_ID in kps_dict:
            kps_dict[IMAGE_ID] += "\n"
            kps_dict[IMAGE_ID] += f"{CLASS_ID-1} {x_c_norm} {y_c_norm} {box_width_norm} {box_height_norm} "
            kps_dict[IMAGE_ID] += " ".join(map(str, kps_flattend))
        else:
            kps_dict[IMAGE_ID] = f"{CLASS_ID-1} {x_c_norm} {y_c_norm} {box_width_norm} {box_height_norm} "
            kps_dict[IMAGE_ID] += " ".join(map(str, kps_flattend))

    for id in image_ids:
        TXT_FILE = str(id) +".txt"
        with open(os.path.join(LABEL_PATH, TXT_FILE), "w") as f:
            line = kps_dict[id]
            f.write(line)

def get_ids(json_data):
    ids = np.array([], dtype=int)

    for annot in json_data["images"]:
        ids = np.append(ids, annot["id"])

    return ids

def change_image_name(image_path, json_path, json_data):
    for i in range(len(json_data["images"])):
        original_name = json_data["images"][i]["file_name"]
        new_name = str(json_data["images"][i]["id"]) + ".jpg"
        old_path = os.path.join(image_path, original_name)
        new_path = os.path.join(image_path, new_name)
        os.rename(old_path, new_path)
        json_data["images"][i]["file_name"] = new_name

    with open(json_path, 'w') as file:
        json.dump(json_data, file, indent=4)

def prepare_predictions(
    image_dir_path,
    video_name,
    model,
    COLORS_RGB_MAP,
    BOX_IOU_THRESH = 0.55,
    BOX_CONF_THRESH=0.30,
    KPT_CONF_THRESH=0.68):

    #image_path = os.path.join(image_dir_path, image_filename)
    #image = cv2.imread(image_path).copy()
    image_shape = (1080, 1920, 3)
    shape_multiplier = np.array(image_shape[:2][::-1]) # (W, H).

    csv_dict = {}
    image_list = os.listdir(image_dir_path)
    frame_list = sorted([int(frame.split(".")[0]) for frame in image_list])

    prev_image_dir_path = os.path.join("./prediction", video_name)
    labelled_dir = os.path.join(prev_image_dir_path, "labelled")
    print(labelled_dir)
    os.makedirs(labelled_dir, exist_ok=True)

    with tqdm(total=len(frame_list), desc="Predicting Frames") as pbar:
        for frame_num in frame_list:
            image_path = os.path.join(image_dir_path, str(frame_num) + ".png")
            print(image_path)
            image = cv2.imread(image_path).copy()
            results = model.predict(image_path, conf=BOX_CONF_THRESH, iou=BOX_IOU_THRESH, save_txt=True, save_frames=True)[0].cpu()

            '''if not len(results.boxes.xyxy):
                return image'''
            # Get the predicted boxes, conf scores and keypoints.
            pred_labels = results.boxes.cls.tolist()

            lizard_dict = {}
            if not len(pred_labels):
                cv2.imwrite(os.path.join(labelled_dir, f"{frame_num}.png"), image)
                lizard_dict = {}
            else:
                pred_boxes = results.boxes.xyxy.numpy()
                pred_box_conf = results.boxes.conf.numpy()
                pred_kpts_xy = results.keypoints.xy.numpy()
                pred_kpts_xyn = results.keypoints.xyn.numpy()
                pred_kpts_conf = results.keypoints.conf.numpy()

                # Draw predicted bounding boxes, conf scores and keypoints on image.
                for label, boxes, score, draw_kpts, kpts, confs in zip(pred_labels, pred_boxes, pred_box_conf, pred_kpts_xy, pred_kpts_xyn, pred_kpts_conf):
                    kpts[:, :2] *= shape_multiplier
                    kpts_ids = np.where(confs < KPT_CONF_THRESH)[0]
                    kpts_ids_draw = np.where(confs > KPT_CONF_THRESH)[0]
                    draw_kpts = draw_kpts[kpts_ids_draw]
                    kpts[kpts_ids] = np.nan
                    filter_kpts = kpts
                    filter_kpts = np.concatenate([filter_kpts, np.expand_dims(np.arange(0,15,1), axis=-1)], axis=-1)
                    draw_kpts = np.concatenate([draw_kpts, np.expand_dims(kpts_ids_draw, axis=-1)], axis=-1)
                    image = draw_boxes(image, boxes, label, score=score)
                    image = draw_landmarks(image, draw_kpts, COLORS_RGB_MAP)
                    cv2.imwrite(os.path.join(labelled_dir, f"{frame_num}.png"), image)
                    print(label)
                    if label == 0.0:
                        lizard_dict["lizard_b"] = [boxes, filter_kpts[:,:2]]
                    elif label == 1.0:
                        lizard_dict["lizard_y"] = [boxes, filter_kpts[:,:2]]

            csv_dict[frame_num] = lizard_dict
            pbar.update(1)

    print(csv_dict)
    return csv_dict

def prepare_predictions_wrong(
    video_path,
    model,
    BOX_IOU_THRESH = 0.70,
    BOX_CONF_THRESH=0.50):

    video_path = "../../../lizard_videos/lizard-hdoh-2024-02-20/videos/10_52.mp4"
    results = model(video_path, save=True, conf=BOX_CONF_THRESH, iou=BOX_IOU_THRESH, save_txt=True, save_frames=True)

def denorm(box_data, keypoints_data):

    image_shape = (1080, 1920, 3)
    shape_multiplier = np.array(image_shape[:2][::-1]) # (W, H).
    # Final absolute coordinates (xmin, ymin, xmax, ymax).
    box_data = box_data.reshape(1,-1)
    denorm_boxes = np.zeros_like(box_data)

    # De-normalize center coordinates from YOLO to (xmin, ymin).
    denorm_boxes[:, :2] = (shape_multiplier/2.) * (2*box_data[:,:2] - box_data[:,2:])

    # De-normalize width and height from YOLO to (xmax, ymax).
    denorm_boxes[:, 2:] = denorm_boxes[:,:2] + box_data[:,2:]*shape_multiplier


    for boxes, kpts, in zip(denorm_boxes, keypoints_data):
        # De-normalize landmark coordinates.
        kpts[:, :2]*= shape_multiplier

    return denorm_boxes, keypoints_data

def data_denorm(LABEL_PATH):
    LABEL_FILES = os.listdir(LABEL_PATH)

    file_list = []
    frame_num = []
    csv_dict = {}

    for file in LABEL_FILES:
        filename = "".join(file.split(".")[:-1])
        file_list.append(filename)

        frame_name = str(filename).split("_")[-1]
        frame_num.append(int(frame_name))

    NUM_LANDMARKS = 15

    frame_num = sorted(frame_num)
    print(len(frame_num))

    for idx, file in enumerate(file_list):
        with open(os.path.join(LABEL_PATH, file+".txt"), "r") as file:
            label_data = [x.split() for x in file.read().strip().splitlines() if len(x)]

        label_data = np.array(label_data, dtype=np.float32)

        # YOLO BBox instances in [x-center, y-center, width, height] in normalized form.
        lizard_dict = {}
        for data in label_data:
            box_instances = data[1:5]
            instance_kpts = []
            kpts_data = data[5:].reshape(-1, NUM_LANDMARKS, 3)
            class_name = data[0]

            for inst_kpt in kpts_data:
                vis_ids = np.where(inst_kpt[:, -1]>0.)[0]
                vis_kpts = inst_kpt[vis_ids][:,:2]
                vis_kpts = np.concatenate([vis_kpts, np.expand_dims(vis_ids, axis=-1)], axis=-1)
                instance_kpts.append(vis_kpts)

            if class_name == 1.0:
                lizard_class = "lizard_y"
            else:
                lizard_class = "lizard_b"

            denorm_boxes, keypoints_data = denorm(box_instances, instance_kpts)
            lizard_dict[lizard_class] = [denorm_boxes, keypoints_data]
        csv_dict[frame_num[idx]] = lizard_dict

    return csv_dict

def data_to_csv(csv_dict, data_folder, video_name):
    joint_names = ["head", "neck", "body1", "lf_leg1", "lf_leg2", "rf_leg1",
                   "rf_leg2", "body2", "body3", "lb_leg1", "lb_leg2", "rb_leg1",
                   "rb_leg2", "mid_tail", "tail"]
    joint_names.extend(joint_names)
    csv_joints = []
    csv_lizards = []
    csv_coords = []

    print(csv_dict)
    for i, joint in enumerate(joint_names):
        csv_joints.append(joint)
        csv_joints.append(joint)
        csv_coords.append("x")
        csv_coords.append("y")
        if i <= 14:
            csv_lizards.append("lizard_b")
            csv_lizards.append("lizard_b")
        else:
            csv_lizards.append("lizard_y")
            csv_lizards.append("lizard_y")

    df = pd.DataFrame({"individuals": csv_lizards,
                       "joints": csv_joints,
                       "coords": csv_coords})
    data_dict = {}
    data_len = []
    for frame in csv_dict:
        lizard_b_coords = []
        lizard_y_coords = []
        for lizard in csv_dict[frame]:
            if lizard == "lizard_b":
                for coords in csv_dict[frame][lizard][1]:
                    lizard_b_coords.append(coords[0])
                    lizard_b_coords.append(coords[1])
            elif lizard == "lizard_y":
                for coords in csv_dict[frame][lizard][1]:
                    lizard_y_coords.append(coords[0])
                    lizard_y_coords.append(coords[1])
        if lizard_b_coords == []:
            lizard_b_coords = [np.nan for _ in range(30)]
        if lizard_y_coords == []:
            lizard_y_coords = [np.nan for _ in range(30)]

        lizard_b_coords.extend(lizard_y_coords)
        lizard_coords = lizard_b_coords
        data_dict[int(frame)] = lizard_coords

    df_data = pd.DataFrame(data_dict)

    df = pd.concat([df, df_data], axis=1)

    #df = df.set_index('individuals')
    df = df.T
    df.to_csv(f"./data/{data_folder}/{video_name}.csv")

def extract_frame(video_name):
    #video_path = f"./videos/{video_name}_ud.mp4"
    video_path = f"./videos/{video_name}.mp4"
    #video_path = f"/mnt/e/lizard_backup/lizard_1/{video_name}.mp4"
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = video_name.split(".")[0]
    output_directory = f'./prediction/{video_name}/frames/'
    os.makedirs(output_directory, exist_ok=True)

    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        frame_number = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            resized_frame = cv2.resize(frame, (640, 640))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            image.save(os.path.join(output_directory, f'{frame_number}.png'))
            frame_number += 1
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    return output_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train?')
    parser.add_argument("--visual", "-v", action="store_true", help="visualization?")
    parser.add_argument("--train", "-t", action="store_true", help="train?")
    parser.add_argument("--eval", "-e", action="store_true", help="evalutate?")
    parser.add_argument("--rename", "-r", action="store_true", help="rename?")
    parser.add_argument("--label", "-l", action="store_true", help="label?")
    parser.add_argument("--pred", "-p", action="store_true", help="prediction?")
    parser.add_argument("--video_name", "-vn", help="What is video name")
    parser.add_argument("--video_date", "-vd", help="What is video date")
    parser.add_argument("--pred_num", "-pn", help="What is the number of prediction?")

    args = parser.parse_args()

    TRAIN_JSON_PATH = os.path.join("../train", "_annotations.coco.json")
    VALID_JSON_PATH = os.path.join("../valid", "_annotations.coco.json")
    TEST_JSON_PATH = os.path.join("../test", "_annotations.coco.json")

    with open(TRAIN_JSON_PATH) as file:
        train_json_data = json.load(file)

    with open(VALID_JSON_PATH) as file:
        valid_json_data = json.load(file)

    with open(TEST_JSON_PATH) as file:
        test_json_data = json.load(file)

    train_ids = get_ids(train_json_data)
    valid_ids = get_ids(valid_json_data)
    test_ids = get_ids(test_json_data)

    print(f"Train IDs number: {len(train_ids)}, Valid IDs number: {len(valid_ids)}, Test IDs number: {len(test_ids)}")

    TRAIN_DIR = f"train"
    TRAIN_FOLDER_IMG = f"images"
    TRAIN_FOLDER_LABELS = f"labels"

    VALID_DIR = f"valid"
    VALID_FOLDER_IMG = f"images"
    VALID_FOLDER_LABELS = f"labels"

    TEST_DIR = f"test"
    TEST_FOLDER_IMG = f"images"
    TEST_FOLDER_LABELS = f"labels"

    TRAIN_IMG_PATH = os.path.join("../", TRAIN_DIR, TRAIN_FOLDER_IMG)
    TRAIN_LABEL_PATH = os.path.join("../", TRAIN_DIR, TRAIN_FOLDER_LABELS)

    VALID_IMG_PATH = os.path.join("../", VALID_DIR, VALID_FOLDER_IMG)
    VALID_LABEL_PATH = os.path.join("../", VALID_DIR, VALID_FOLDER_LABELS)

    TEST_IMG_PATH = os.path.join("../", TEST_DIR, TEST_FOLDER_IMG)
    TEST_LABEL_PATH = os.path.join("../", TEST_DIR, TEST_FOLDER_LABELS)

    os.makedirs(TRAIN_IMG_PATH, exist_ok=True)
    os.makedirs(TRAIN_LABEL_PATH, exist_ok=True)
    os.makedirs(VALID_IMG_PATH, exist_ok=True)
    os.makedirs(VALID_LABEL_PATH, exist_ok=True)
    os.makedirs(TEST_IMG_PATH, exist_ok=True)
    os.makedirs(TEST_LABEL_PATH, exist_ok=True)

    if args.rename:
        change_image_name(TRAIN_IMG_PATH, TRAIN_JSON_PATH, train_json_data)
        change_image_name(VALID_IMG_PATH, VALID_JSON_PATH, valid_json_data)
        change_image_name(TEST_IMG_PATH, TEST_JSON_PATH, test_json_data)

    if args.label:
        create_yolo_txt_files(train_json_data, TRAIN_LABEL_PATH)
        create_yolo_txt_files(valid_json_data, VALID_LABEL_PATH)
        create_yolo_txt_files(test_json_data, TEST_LABEL_PATH)

    ann_meta_data = pd.read_csv("../keypoint_definitions.csv")
    COLORS = ann_meta_data["Hex colour"].values.tolist()

    COLORS_RGB_MAP = []
    for color in COLORS:
        R, G, B = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        COLORS_RGB_MAP.append({color: (R, G, B)})

    train_images = os.listdir(TRAIN_IMG_PATH)
    valid_images = os.listdir(VALID_IMG_PATH)
    test_images = os.listdir(TEST_IMG_PATH)

    print(f"Training images: {len(train_images)}, Validation Images: {len(valid_images)}")

    if (args.visual):
        data_visualization(TRAIN_IMG_PATH, TRAIN_LABEL_PATH, COLORS_RGB_MAP)
        #data_visualization(VALID_IMG_PATH, VALID_LABEL_PATH, COLORS_RGB_MAP)
        #data_visualization(TEST_IMG_PATH, TEST_LABEL_PATH, COLORS_RGB_MAP)

    train_config = TrainingConfig()
    data_config = DatasetConfig()
    if (args.train):

        #current_dir = os.getcwd()
        current_dir = "../"
        data_dict = dict(
                        path      = current_dir,
                        train     = os.path.join(current_dir, TRAIN_DIR, TRAIN_FOLDER_IMG),
                        val       = os.path.join(current_dir, VALID_DIR, VALID_FOLDER_IMG),
                        names     = train_config.CLASSES_DICT,
                        kpt_shape = list(train_config.KPT_SHAPE),
                    )
        print(list(train_config.KPT_SHAPE))

       #with open(train_config.DATASET_YAML, "w") as config_file:
       #     yaml.dump(data_dict, config_file)
        train_start = input("Start Training(y/n): ")
        if train_start == "y":
            pose_model = model = YOLO(train_config.MODEL)

            pose_model.train(data    = train_config.DATASET_YAML,
                        epochs       = train_config.EPOCHS,
                        imgsz        = data_config.IMAGE_SIZE,
                        batch        = data_config.BATCH_SIZE,
                        project      = train_config.PROJECT,
                        name         = train_config.NAME,
                        close_mosaic = data_config.CLOSE_MOSAIC,
                        mosaic       = data_config.MOSAIC,
                        fliplr       = data_config.FLIP_LR
                    )

    if (args.eval):
        ckpt_path  = os.path.join(train_config.PROJECT, train_config.NAME, "weights", "best.pt")
        model_pose = YOLO(ckpt_path)

        metrics = model_pose.val()

    if (args.pred):
        ckpt_path  = os.path.join(train_config.PROJECT, train_config.NAME, "weights", "best.pt")
        print(ckpt_path)
        model_pose = YOLO(ckpt_path)

        video_date_data_path = os.path.join("./data", args.video_date)

        if not os.path.exists(video_date_data_path):
            os.makedirs(video_date_data_path)

        IMG_DIR = extract_frame(args.video_name)
        #IMG_DIR = "./prediction/3_8/frames/"
        csv_dict = prepare_predictions(IMG_DIR, args.video_name, model_pose, COLORS_RGB_MAP)
        data_to_csv(csv_dict, args.video_date, args.video_name)

