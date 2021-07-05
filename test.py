import cv2
import numpy as np

'''
def nothing(x):
    pass


# create Trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)
'''


def mask(img_hsv, HSV_min, HSV_max):
    thresh = cv2.inRange(img_hsv, HSV_min, HSV_max)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


while True:
    img = cv2.imread("example.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    '''
    # get value Trackbars
    H_min = cv2.getTrackbarPos("H_min", "Trackbars")
    S_min = cv2.getTrackbarPos("S_min", "Trackbars")
    V_min = cv2.getTrackbarPos("V_min", "Trackbars")
    H_max = cv2.getTrackbarPos("H_max", "Trackbars")
    S_max = cv2.getTrackbarPos("S_max", "Trackbars")
    V_max = cv2.getTrackbarPos("V_max", "Trackbars")
    

    HSV_min = np.array([H_min, S_min, V_min])
    HSV_max = np.array([H_max, S_max, V_max])
    '''

    HSV_min = np.array([23, 50, 19])
    HSV_max = np.array([46, 229, 255])

    result = cv2.bitwise_and(img, img, mask=mask(hsv, HSV_min, HSV_max))
    cv2.imshow("Original", img)
    # cv2.imshow("mask", mask(hsv,HSV_min, HSV_max))
    cv2.imshow("result", result)
    cv2.imwrite("result.jpg", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyAllWindows()