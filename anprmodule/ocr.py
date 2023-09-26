import cv2
import keras_ocr

def ocr_image(img, coordinates):
    pipeline = keras_ocr.pipeline.Pipeline()
    x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
    img = img[y:h, x:w]

    img = cv2.resize(img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)

    text = ""
    pred = pipeline.recognize([img])
    # pred contains predicted text in the input image. Some number plates have text on the black border.
    for p in pred[0]:
        text += str(p[0])
    if text[0].isdigit():
        text = text[3:] + text[0:3]
    return text.upper()