
#~~CLAHE Algorithm (Contrast Limiting Adaptive Histogram Enhancement)
import sys
import cv2
import os

def ImgEnhancer(root_image):
    #~~Step 1: Apply median blur filter
    img = cv2.imread(root_image)

    #~~Step 2: Convert Color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #~~Step 3: Split color channels
    l, a, b = cv2.split(lab_img)

    #~~Step 4: Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    clahe_lab_img = clahe.apply(l)

    #~~ Step 5: Combine the CLAHE enhanced L-channel back to A and B channels
    updated_clahe_lab_img = cv2.merge((clahe_lab_img, a, b))

    #~~ Step 6: Convert color space
    final_img = cv2.cvtColor(updated_clahe_lab_img, cv2.COLOR_LAB2BGR)

    #~~ Step 7: Scale and save image to uploads folder
    img_scaled = cv2.resize(final_img, (400, 400), interpolation=cv2.INTER_AREA)
    pre_process = "CLACHE_" + os.path.basename(root_image)
    cv2.imwrite(pre_process, img_scaled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#root_image = sys.argv[1]
root_image = "Soil.png"
ImgEnhancer(root_image)