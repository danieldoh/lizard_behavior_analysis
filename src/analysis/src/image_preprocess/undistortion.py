import cv2
import numpy as np

imgpoints = np.zeros((50,2), dtype=np.float32)
count = -1

def mouse_callback(event, x, y, flags, params):
    if event == 2:
        global imgpoints
        global count
        count += 1
        imgpoints[count] = [x,y]
        print(count)

if __name__ == "__main__":

    # setting
    #image_dir = "/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection/image/"
    image_dir = "../../image/"
    image_paths = ["1.png", "2.png"]#"3.png"]
    image_path_list = []
    for path in image_paths:
        concat_path = image_dir + path
        image_path_list.append(concat_path)

    objpoints = np.array([
        [0, 4.86, 0],
        [19.44, 4.86, 0],
        [34.02, 4.86, 0],
        [48.6, 4.86, 0],
        [68, 4.86, 0],
        [0, 17.01, 0],
        [19.44, 17.01, 0],
        [34.02, 17.01, 0],
        [48.6, 17.01, 0],
        [68, 17.01, 0],
        [0, 26.73, 0],
        [19.44, 26.73, 0],
        [34.02, 26.73, 0],
        [48.6, 26.73, 0],
        [68, 26.73, 0],
        [0, 36.45, 0],
        [19.44, 36.45, 0],
        [34.02, 36.45, 0],
        [48.6, 36.45, 0],
        [68, 36.45, 0],
        [0, 46.17, 0],
        [19.44, 46.17, 0],
        [34.02, 46.17, 0],
        [48.6, 46.17, 0],
        [68, 46.17, 0]
    ], dtype=np.float32)

    objpoints = np.tile(objpoints, (2,1))

    # read image
    for path in image_path_list:
        img = cv2.imread(path)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', mouse_callback)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(imgpoints)

    print(objpoints)
    print(imgpoints)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints], (img.shape[1], img.shape[0]), None, None)

    print(mtx)
    print("mtx type: ", mtx.dtype)
    print("mtx size: ", mtx.size)
    print("mtx shape: ", mtx.shape)

    print(dist)
    print("dist type: ", dist.dtype)
    print("dist size: ", dist.size)
    print("dist shape: ", dist.shape)

    #img2 = cv2.imread("/Users/doh/HJ/Research/prof_byeon/lizard/lizard_detection/image/img051.png")
    img = cv2.imread("../../image/1.png")
    img2 = cv2.imread("../../image/test.png")
    #h,  w = img.shape[:2]
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistored_img = cv2.undistort(img, mtx, dist, None, None)
    undistored_img_2 = cv2.undistort(img2, mtx, dist, None, None)

    top_left = (imgpoints[0,0], imgpoints[0,1])
    top_right = (imgpoints[4,0], imgpoints[4,1])
    bottom_left = (imgpoints[20,0], imgpoints[20,1])
    bottom_right = (imgpoints[24,0], imgpoints[24,1])

    #undistored_cropped_img = undistored_img[int(top_left[1]):int(bottom_left[1]), int(top_left[0]):int(top_right[0])]

    cv2.imshow("Original Image", img)
    #cv2.imshow("Undistored Image", undistored_cropped_img)
    cv2.imshow("Undistored Image", undistored_img)
    cv2.imshow("Original Image 2", img2)
    cv2.imshow("Undistored Image 2", undistored_img_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #imgpoints = np.array([[136, 123], [538, 54], [907, 38], [1283, 38], [1701, 72], [148, 400], [552, 368], [920, 357], [1286, 349], [1698, 357], [178, 619], [574, 610], [929, 601], [1279, 589], [1676, 575], [224, 820], [604, 829], [935, 823], [1266, 809], [1646, 777], [273, 995], [623, 1018], [942, 1013], [1251, 998], [1607, 961], [42, 144], [436, 82], [795, 65], [1150, 73], [1539, 106], [35, 408], [442, 380], [801, 368], [1157, 358], [1557, 369], [43, 627], [454, 617], [810, 608], [1161, 595], [1554, 582], [76, 839], [474, 843], [816, 838], [1155, 819], [1536, 785], [113, 1028], [489, 1054], [824, 1042], [1148, 1024], [1514, 976], [151, 147], [519, 76], [871, 50], [1244, 41], [1676, 63], [148, 402], [528, 371], [883, 348], [1255, 342], [1689, 348], [166, 612], [542, 601], [892, 592], [1252, 583], [1680, 577], [202, 805], [565, 820], [896, 822], [1245, 811], [1655, 796], [239, 983], [576, 1009], [900, 1005], [1228, 1007], [1582, 989], [97, 156], [469, 70], [846, 27], [1240, 4], [1715, 19], [114, 425], [491, 382], [850, 360], [1239, 340], [1706, 335], [146, 632], [516, 619], [865, 605], [1231, 594], [1673, 574], [195, 822], [546, 834], [871, 834], [1215, 822], [1630, 796], [237, 987], [563, 1015], [869, 1017], [1192, 1009], [1588, 984], [188, 166], [577, 99], [937, 84], [1297, 85], [1703, 119], [184, 433], [590, 400], [948, 388], [1310, 382], [1713, 386], [208, 646], [605, 637], [958, 629], [1309, 618], [1705, 603], [243, 854], [628, 861], [962, 856], [1299, 842], [1683, 811], [282, 1032], [649, 1064], [970, 1061], [1289, 1039], [1652, 998]], dtype=np.float32)

