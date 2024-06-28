import os
import cv2
import shutil
import argparse

from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip

def moveVideo(video_dir, video_path, video_list, date_time):
    date_list = [video.split(".")[0].split("_")[0] for video in video_list]
    date_list = set(date_list)

    for date_folder in date_list:
        new_date_folder = os.path.join(video_dir, date_folder)

        if not os.path.exists(new_date_folder):
            os.makedirs(new_date_folder)
            print(f"{new_date_folder} is created")
        else:
            print(f"{new_date_folder} already exists")

        videos = date_time[date_folder]
        for video in videos:
            video  = f"{date_folder}_{video}.mp4"
            current_video_path = os.path.join(video_path, video)
            shutil.move(current_video_path, new_date_folder)

def getTime(video_list):
    time_list = [video.split(".")[0].split("_") for video in video_list]
    date_time = {}
    for t in time_list:
        time_videoNum = t[1]+"_"+t[2]
        if t[0] in date_time:
            date_time[t[0]].append(time_videoNum)
        else:
            date_time[t[0]] = [time_videoNum]
    return date_time

def getRemainderTime(date_time):

    remainder_time = {}

    for k in date_time:
        given_time = date_time[k][-1].split("_")[0]
        given_datetime = datetime.strptime(given_time, '%H%M%S')

        midnight = given_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        time_until_midnight = midnight - given_datetime

        hours, remainder = divmod(time_until_midnight.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        formatted_time_until_midnight = '{:02d}{:02d}{:02d}'.format(hours, minutes, seconds)

        remainder_time[k] = formatted_time_until_midnight

    return remainder_time

def crop_video(video_dir, video_path, date_time, remainder_time):
    for date in date_time:
        next_day = int(date[6:8]) + 1
        if len(str(next_day)) == 2:
            next_day = date[:6] + str(next_day)
        else:
            next_day = date[:6] + "0" + str(next_day)

        input_video_path = os.path.join(video_path, f'{date}_{date_time[date][-1]}.mp4')

        prev_output_dir = os.path.join(video_dir, date)
        next_output_dir = os.path.join(video_dir, next_day)
        if not os.path.exists(prev_output_dir):
            os.makedirs(prev_output_dir)
            print(f"{prev_output_dir} is created")

        if not os.path.exists(next_output_dir):
            os.makedirs(next_output_dir)
            print(f"{next_output_dir} is created")

        output_video_path_prev = os.path.join(prev_output_dir, f'{date}_{date_time[date][-1]}prev.mp4')
        output_video_path_next = os.path.join(next_output_dir, f'{next_day}_000000_1next.mp4')

        print(output_video_path_prev)
        print(output_video_path_next)

        start_time = remainder_time[date]

        start_time = int(start_time[:2]) * 3600 + int(start_time[2:4]) * 60 + int(start_time[4:])

        video_clip = VideoFileClip(input_video_path, audio=False)

        print(output_video_path_prev)
        cropped_video_clip = video_clip.subclip(0, start_time-1)
        cropped_video_clip.write_videofile(output_video_path_prev, codec='libx264')

        print(output_video_path_next)
        cropped_video_clip_next = video_clip.subclip(start_time)
        cropped_video_clip_next.write_videofile(output_video_path_next, codec='libx264')

    video_clip.close()
    cropped_video_clip.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Path")
    parser.add_argument("--video_dir", "-vd", help="Video Directory")
    parser.add_argument("--video_folder", "-vf", help="Video Folder")

    args = parser.parse_args()

    VIDEO_DIR = args.video_dir
    VIDEO_FOLDER = args.video_folder

    VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_FOLDER)

    VIDEO_LIST = os.listdir(VIDEO_PATH)


    date_time = getTime(VIDEO_LIST)

    # move videos to the "date-based" folder
    moveVideo(VIDEO_DIR, VIDEO_PATH, VIDEO_LIST, date_time)

    remainder_time = getRemainderTime(date_time)
    #calculate the time before 12am, and cut it using moviepy, and create a new video file

    crop_video(VIDEO_DIR, VIDEO_PATH, date_time, remainder_time)


