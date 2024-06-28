# Lizard Behavior Analysis
Lizard Behavior Analysis Using an Aerosol Exposure Chamber Containing Cameras

## Installation

1. Create a virtual environment:
   - If you do have "Anaconda" on your local computer, please install it
   - Create a new virtual environment:
     ```sh
     conda create -n [environment name] python=3.9
     ```
2. Install the required modules using **requirements.txt**:
   ```sh
   pip install -r requirements.txt
   ```

## Model Download

1. Download the trained model. And place it in the project directory.
   - Link: https://drive.google.com/drive/folders/12U1MClQj8xj2cc6RqjN_ld0HKLZL31Ux?usp=sharing


## File Structure

1. If you need to train the network again with a new dataset
   
2. In **src** folder, create the following folders:
   - data
   - prediction
   - runs
   - videos

## Video Preparation
Place the video in the **videos** folder in the **src** folder.

## Prediction

Go to **src** folder:
```sh
cd src
```

Run:
```sh
python main.py -p -vn [video_name] -vd [video_date]
```

## Check Result

1. Go to **prediction** folder:

```sh
cd src/prediction/[video_name]
```

2. Check **Labelled** folder
