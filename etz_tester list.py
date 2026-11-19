import platform
import cv2, numpy as np

api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
cap = cv2.VideoCapture(0, api)

lower_black = np.array([0, 0, 0]) # чёрный
upper_black = np.array([255, 100, 110])

kernel = np.ones((1, 1), np.uint8)

k_per = 0.6


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()

aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 21
aruco_params.adaptiveThreshWinSizeStep = 4
aruco_params.minMarkerPerimeterRate = 0.005

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def center(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return (0, 0)
    return (round(M["m10"]/M["m00"]), round(M["m01"]/M["m00"]))

def px_to_cell(px, py):
    if const_px_to_cell is None:
        return None
    v = const_px_to_cell @ np.array([px, py, 1.0])
    if abs(v[2]) < 1e-9:
        return None
    return float(round(v[0] / v[2], 1)), float(round(v[1] / v[2], 1))

def to_img(pt):
    if grid is None:
        return None
    v = grid @ np.array([pt[0], pt[1], 1.0])
    if v[2] == 0:
        return (0, 0)
    return (int(round(v[0] / v[2])), int(round(v[1] / v[2])))


while True:
    ok, frame = cap.read()
    if not ok or frame is None: break
    corners, ids, _ = detector.detectMarkers(frame)
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    crds, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords = [c for c in crds if cv2.contourArea(c) >= 50]
    grid = None
    const_px_to_cell = None
    if coords:
        pts = max(coords, key=cv2.contourArea).reshape(-1, 2)
        cx = pts[:, 0]
        cy = pts[:, 1]
        l_verh = list(pts[np.argmin(cx + cy)])
        l_niz = list(pts[np.argmin(cx - cy)])
        d_r_niz = list(pts[np.argmax(cx + cy)])
        r_niz = list(pts[np.argmax(cx - cy)])

        r_verh = [w, 0]
        if ids is not None and len(corners) > 0:
            r_verh = min(corners, key=lambda m: np.hypot(np.mean(m[0, :, 0]) - 0, np.mean(m[0, :, 1])))
            r_verh = [w - 1 - int(r_verh[0, 1, 0]), int(r_verh[0, 1, 1])]
            cv2.drawMarker(frame, tuple(r_verh), (155, 155, 155), cv2.MARKER_CROSS, 20, 2)
            

        d = np.hypot(abs(r_niz[0] - d_r_niz[0]), abs(r_niz[1] - d_r_niz[1]))
        r_verh = (round(r_verh[0] + d), round(r_verh[1] + d * k_per))
        l_verh = (round(l_verh[0] + d * k_per), round(l_verh[1]))
        l_niz = (round(l_niz[0] + d), round(l_niz[1] - d))
        cv2.drawMarker(frame, l_verh, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.drawMarker(frame, r_niz, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.drawMarker(frame, l_niz, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.drawMarker(frame, r_verh, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cell_w, cell_h = 9, 6
        cell_size = np.float32([[0, 0], [cell_w, 0], [cell_w, cell_h], [0, cell_h]])
        grid = cv2.getPerspectiveTransform(cell_size, np.float32([l_verh, r_verh, r_niz, l_niz]))
        nx, ny = 9, 6
        cell_color, cell_thick = (255, 255, 255), 2
        if np.all(np.isfinite(grid)) and abs(np.linalg.det(grid)) > 1e-6:
            for i in range(nx + 1):
                cell_x = cell_w * i / nx
                cv2.line(frame, to_img((cell_x, 0)), to_img((cell_x, cell_h)), cell_color, cell_thick, cv2.LINE_AA)
            for j in range(ny + 1):
                cell_y = cell_h * j / ny
                cv2.line(frame, to_img((0, cell_y)), to_img((cell_w, cell_y)), cell_color, cell_thick, cv2.LINE_AA)
            const_px_to_cell = np.linalg.inv(grid)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


    cv2.imshow("etz_tester", np.hstack((frame, mask_bgr)))
    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows() 