import platform
import cv2, numpy as np

api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
cap = cv2.VideoCapture(0, api)

rgb_1_check = [(np.array([0, 90, 60]), np.array([10, 255, 255]), np.array([170, 90, 60]), np.array([179, 255, 255]), "r", (0, 0, 255)), (np.array([40, 90, 50]), np.array([85, 255, 255]), "g", (0, 255, 0)), 
               (np.array([95, 90, 50]), np.array([130, 255, 255]), "b", (255, 0, 0)), (np.array([20, 90, 80]), np.array([35, 255, 255]), "y", (0, 255, 255)), (np.array([140, 90, 60]), np.array([169, 255, 255]), "p", (255, 0, 255)),
               (np.array([0, 60, 40]), np.array([179, 255, 255]))] # красный, зелёный, синий, жёлтый, пурпурный, цветной

kernel = np.ones((1, 1), np.uint8)

def center(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return (0, 0)
    return (round(M["m10"]/M["m00"]), round(M["m01"]/M["m00"]))


while True:
    ok, frame = cap.read()
    if not ok or frame is None: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for d_hsv in rgb_1_check[:-1]:
        if (len(d_hsv) >= 6):
            mask_cvet = cv2.inRange(hsv, d_hsv[0], d_hsv[1]) | cv2.inRange(hsv, d_hsv[2], d_hsv[3])
        else:
            mask_cvet = cv2.inRange(hsv, d_hsv[0], d_hsv[1])
        mask_cvet = cv2.morphologyEx(mask_cvet, cv2.MORPH_OPEN, kernel)
        crds, _ = cv2.findContours(mask_cvet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coords_cv = [c for c in crds if cv2.contourArea(c) >= 50]
        for c in coords_cv:
            coords_c = center(c)
            cv2.drawMarker(frame, coords_c, d_hsv[-1], cv2.MARKER_CROSS, 24, 2)
        mask_cvet_bgr = cv2.cvtColor(mask_cvet, cv2.COLOR_GRAY2BGR)


    cv2.imshow("etz_tester", (frame))
    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows() 