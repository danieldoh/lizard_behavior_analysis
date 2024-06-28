# This calculates the direction of each lizard moved.
# First, find the trajectory using head and tail.
# Second, visualize the trajectory using matplotlib.

import cv2

import numpy as np 
import matplotlib as mat

from main import get_data

def calculate_direction():
    print("Calculating direction")
    get_data()

def visualize_direction():
    print("Visualize direction")

if __name__ == "__main__":

    calculate_direction()
    visualize_direction()