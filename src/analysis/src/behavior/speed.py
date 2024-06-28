# This calculates the speed of each lizards.
# First, we follow the equation, distance / time
# Second, generate the line graph for each hour to easily compare speed hourly

import cv2

import numpy as np
import matplotlib as mat
from main import get_data
from movement import calculate_movement

def calculate_speed():
    print("Calculating the speed")

    get_data()
    calculate_movement()

def visualize_speed():
    print("Visualizing calculated speed")

if __name__ == "__main__":
    calculate_speed()
    visualize_speed()
