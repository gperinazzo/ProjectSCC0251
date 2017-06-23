# import the necessary packages
import numpy as np
import cv2
import math


def verticalProjection(img):
    "Return a list containing the sum of the pixels in each column"
    (h, w) = img.shape[:2]
    return [np.sum(img[0:h, j:j+1]) / 255 for j in range(w)]


def horizontalProjection(img):
    "Return a list containing the sum of the pixels in each row"
    (h, w) = img.shape[:2]
    return [np.sum(img[i:i+1, 0:w]) / 255 for i in range(h)]


def skew_angle(edges):
    (h, w) = edges.shape[:2]
    threshold = max(h, w)
    while True:
        print('Testing with treshold', threshold)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

        if lines is None or len(lines) == 0:
            threshold -=  threshold // 4
            continue

        angles = []
        for line in lines:
            for rho, theta in line:
                angle = (theta - np.pi / 2.0)
                angle = math.degrees(angle) % 360
                if angle > 180:
                    angle = angle - 360
                if angle > 60 or angle < -60:
                    continue
                angles .append(angle)
        if not len(angles):
            threshold -=  threshold // 4
            continue
        break
    

    angle = np.mean(angles)
    return angle


def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255)


def find_breaks(array, *, ignore_single=True):
    start = -1
    state = False
    breaks = []
    first = 0
    last = len(array)
    for i in range(len(array)):
        if array[i]:
            first = max(i - 1, 0)
            break

    for i in range(len(array) - 1, first, -1):
        if array[i]:
            last = min(i + 1, len(array))
            break

    for i in range(first, last):
        if not state and not array[i]:
            state = True
            start = i
        elif state and array[i]:
            if not ignore_single or i - start > 1:
                breaks.append(start + (i - start) // 2)
            state = False
    return breaks, first, last


def read_image(name):
    image = cv2.imread(name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    mean = np.average(gray)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 1001, mean//4)
    return gray


def process_image(name):
    # load the image from disk
    gray = read_image(name)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    angle = skew_angle(edges)

    rotated = rotate(gray, angle)

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    inverse = cv2.bitwise_not(rotated)
    morph = cv2.erode(inverse, kernel)


    hp = horizontalProjection(morph)
    vp = verticalProjection(morph)


    ahp = hp - min(*hp) - np.average(hp) // 2
    ahp = np.clip(ahp, 0, None)

    breaks, first, last = find_breaks(ahp)

    (h, w) = morph.shape[:2]
    lines = [first, *breaks, last]
    for point in lines:
        cv2.line(rotated, (0, point), (w, point), (0, 0, 255), 2)

    for i in range(1, len(lines)):
        vp = verticalProjection(morph[lines[i-1]:lines[i], :])
        breaks, first, last = find_breaks(vp, ignore_single=False)
        vert_lines = [first, *breaks, last]
        for point in vert_lines:
            cv2.line(rotated, (point, lines[i-1]), (point, lines[i]), (0, 0, 255), 2)



    morph = cv2.bitwise_not(morph)

    cv2.imwrite('edges.png', edges)
    cv2.imwrite('gray.png', gray)
    cv2.imwrite('morph.png', morph)
    cv2.imwrite('rotated.png', rotated)

