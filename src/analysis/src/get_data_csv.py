import pandas as pd
import numpy as np

def read_joint_csv(csv_path):
    df = pd.read_csv(csv_path)

    lizards_row = df.iloc[0].tolist()[1:]
    joints_row = df.iloc[1].tolist()[1:]
    coords_row = df.iloc[2].tolist()[1:]

    lizard_b_pos = {}
    lizard_y_pos = {}

    data = df.iloc[3::,1::].values
    for idx, (lizard, joint) in enumerate(zip(lizards_row, joints_row)):
        if idx % 2 == 1:
            x = data[:,idx-1]
            y = data[:,idx]
            if lizard == "lizard_b":
                if not pd.isnull(x).all() and not pd.isnull(x).all():
                    lizard_b_pos[joint] = [x.tolist(), y.tolist()]
            elif lizard == "lizard_y":
                if not pd.isnull(x).all() and not pd.isnull(x).all():
                    lizard_y_pos[joint] = [x.tolist(), y.tolist()]
    
    return lizard_b_pos, lizard_y_pos, (df.shape[0]-3)

if __name__ == "__main__":

    #csv_path = '/Users/doh/HJ/Research/lizard_joints/lizard-doh-2024-01-22/videos/5_32DLC_resnet50_lizardJan22shuffle4_24000_el.csv'
    csv_path = "../../data/20240318/3_8_ud.csv"
    lizard_b_pos, lizard_y_pos, frame_number = read_joint_csv(csv_path)
    print(frame_number)
