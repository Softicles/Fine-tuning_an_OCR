import os
from pdf2image import convert_from_path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_pdf_to_jpg(pdf_path, output_folder = "output_images"):
    """
    Converts each page of a pdf to image
    
    Args:
        pdf_path: 
        output_folder:
    """

    try:
        os.makedirs(output_folder, exist_ok = True)

        images = convert_from_path(pdf_path, fmt = "jpeg")

        for i, image in enumerate(images):
            image_filename = os.path.join(output_folder, f"image_{i + 1}.jpg")
            image.save(image_filename, "JPEG")
            print(f"Saved {image_filename}")

        print(f"Conversion completed for {pdf_path}, saving output to {output_folder}")

    except Exception as e:
        print("Exception occured", e)

def perform_segmentation(image_path, output_folder = "segmented_words"):
    # reading and resizing

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape

    if w > 1000:

        new_w = 1000
        ar = w / h
        new_h = int(new_w/ar)

        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)

    thresh_img = thresholding(img)

    #dilation
    kernel = np.ones((3,85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)

    (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)

    img2 = img.copy()
    words_list = []

    for line in sorted_contours_lines:

        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated[y:y+h, x:x+w]

        # draw contours on each word
        (cnt, hierarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])

        for word in sorted_contour_words:
            if cv2.contourArea(word) < 400:
                continue
            
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
            cv2.rectangle(img2, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (255,255,100),2)



def thresholding(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,115,255,cv2.THRESH_BINARY_INV)
    plt.imshow(thresh, cmap='gray')
    