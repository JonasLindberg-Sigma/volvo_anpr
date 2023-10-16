import cv2
import numpy as np
import keras_ocr
import pytesseract
import PIL

def ocr_image(img, coordinates):
    pipeline = keras_ocr.pipeline.Pipeline()
    x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
    img = img[y:h, x:w]
    img = cv2.resize(img, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
    #print(img.shape)
    #cv2.imshow('License Plate', img)
    #cv2.waitKey(0)
    kernel3 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel3)
    text = ""
    pred = pipeline.recognize([img])
    # pred contains predicted text in the input image. Some number plates have text on the black border.
    for p in pred[0]:
        text += str(p[0])
    if text[0].isdigit():
        text = text[3:] + text[0:3]
    return text.upper()
