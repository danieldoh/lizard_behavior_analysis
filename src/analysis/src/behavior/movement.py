# This calculates the movement of two lizards.
# First, calculate the ratio of real:image length
# Second, plot the movement different between day and night using line graph
# Third, calculate the total distances they moved.

import cv2

import numpy as np
import matplotlib as mat

from main import get_data

def calculate_movement():
    print("Calculating movement")

    get_data()

def visualize_movement():
    print("Visualize movement")

if __name__ == "__main__":

    calculate_movement()
    visualize_movement()
