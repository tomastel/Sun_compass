import cv2
import csv
import glob
import numpy as np
import pandas as pd
from ahrs import Quaternion
from ahrs.filters import Madgwick
from datetime import datetime, timedelta
from pvlib.solarposition import get_solarposition

# --------------------------------------------------------------
# ------------ DATA IMPORT AND FILTERING FUNCTIONS -------------
# --------------------------------------------------------------

def gps_epoch_time_to_unix(gps_week: int, 
                           gps_sec: float) -> timedelta:
    gps_epoch = datetime(1980, 1, 6) # GPS epoch Jan 6th 1980
    unix_epoch = datetime(1970, 1, 1)
    diff = timedelta(weeks=gps_week, seconds=gps_sec)
    leap_sec = timedelta(seconds=18)
    gps_time = gps_epoch + diff - leap_sec

    return (gps_time - unix_epoch).total_seconds()


def read_gt_data(path: str) -> np.ndarray:
    data_list = []
    keys = ['GPS_time_week', 'GPS_time_sec', 
            'az[rad]', 'age[s]']

    try:
        with open(path, newline='') as file:
            reader = csv.reader(file)
            next(reader) # Skip header

            for row in reader:
                sample = {keys[i]: float(value) for i, 
                          value in enumerate(row)}
                sample['unix_time'] = gps_epoch_time_to_unix(sample['GPS_time_week'], sample['GPS_time_sec'])
                data_list.append(sample)

    except FileNotFoundError:
        print('No GPS data found.')
        return None

    return np.array(data_list)


def sort_data(imu_data: np.ndarray, 
              metadata:np.ndarray, 
              gps_data: np.ndarray) -> np.ndarray:

    def find_closest(target, array):
        return np.abs(array - target).argmin()

    #remove data from right camera
    metadata_leftcam = np.array([row[0] for row in metadata])

    # Match IMU readings with frame and GPS data by timestamp
    imu_ts = np.array([int(item[0][0]*1e9) for item in imu_data])
    frame_ts = np.array([frame['timestamp_ns'] for frame in metadata_leftcam])

    closest_indices_imu = np.array([find_closest(ts, imu_ts) for ts in frame_ts])
    matched_imu_readings = np.array([imu_data[idx] for idx in closest_indices_imu])

    comp_rp_est = comp_filter(imu_data, 0.1, 1/2000)
    matched_comp_est = np.array([comp_rp_est[idx] for idx in closest_indices_imu])

    madgwick_rp_est = madgwick_filter(imu_data, 1/2000)
    matched_madgwick_est = np.array([madgwick_rp_est[idx] for idx in closest_indices_imu])

    if gps_data is not None:
        gps_ts = np.array([gps['unix_time']*1e9 for gps in gps_data])
        closest_indices_gps = np.array([find_closest(ts, gps_ts) for ts in frame_ts])
        matched_gps = ([gps_data[idx] for idx in closest_indices_gps])

    sorted_data = []
    for i, frame in enumerate(metadata_leftcam):
        s = {}
        imu_sample = matched_imu_readings[i]
        madgwick_sample = matched_madgwick_est[i]
        complementary_sample = matched_comp_est[i]

        s['frame_id'] = frame['frame_id']
        s['ts_sec'] = imu_sample[0][0]

        s['gyro'] = np.array([imu_sample['gyro_data_x'][0], imu_sample['gyro_data_y'][0], imu_sample['gyro_data_z'][0]])
        s['inc'] = np.array([imu_sample['incl_data_x'][0], imu_sample['incl_data_y'][0], imu_sample['incl_data_z'][0]])

        s['roll_b'] = -np.arctan2(s['inc'][1], s['inc'][2])
        s['pitch_b'] = -np.arcsin(-s['inc'][0]/np.linalg.norm(s['inc']))

        s['roll_comp_b'] = -complementary_sample[0]
        s['pitch_comp_b'] = -complementary_sample[1]

        s['roll_madgwick_b'] = -madgwick_sample[0]
        s['pitch_madgwick_b'] = -madgwick_sample[1]
        s['yaw_madgwick_b'] = madgwick_sample[2]

        s['g_b'] = rpy2vec(madgwick_sample[0], madgwick_sample[1], madgwick_sample[2])

        if gps_data is not None:
            s['ref_heading_rad'] = wrap_angle_2pi(matched_gps[i]['az[rad]'] - np.pi/2)

        sorted_data.append(s)

    return np.array(sorted_data)


def comp_filter(raw_imu_data: np.ndarray, alpha: float, T: float) -> np.ndarray:
    raw_inc_data = np.array([np.array([i['incl_data_x'][0], i['incl_data_y'][0], i['incl_data_z'][0]]) for i in raw_imu_data])
    
    raw_gyro_data = np.array([np.array([i['gyro_data_x'][0], i['gyro_data_y'][0], i['gyro_data_z'][0]]) for i in raw_imu_data])

    phi_inc_list = np.array([np.arctan2(i[1], i[2]) for i in raw_inc_data])

    theta_inc_list = np.array([np.arcsin(-i[0]/np.linalg.norm(i)) for i in raw_inc_data])

    phi_hat = 0
    theta_hat = 0

    estimates = []

    for i, gyro_sample in enumerate(raw_gyro_data):   
        phi_dot = gyro_sample[0] + gyro_sample[1]*np.sin(phi_hat)*np.tan(theta_hat) + gyro_sample[2]*np.cos(phi_hat)*np.tan(theta_hat)
        theta_dot = gyro_sample[1]*np.cos(phi_hat) - gyro_sample[2]*np.sin(phi_hat)

        phi_hat = alpha*phi_inc_list[i] + (1-alpha)*(phi_hat + T*phi_dot)
        theta_hat = alpha*theta_inc_list[i] + (1-alpha)*(theta_hat + T*theta_dot)

        estimates.append([phi_hat, theta_hat])

    return np.array(estimates)


def madgwick_filter(raw_imu_data: np.ndarray, T: float) -> np.ndarray:
    raw_inc_data = np.array([np.array([i['incl_data_x'][0], i['incl_data_y'][0], i['incl_data_z'][0]]) for i in raw_imu_data])
    raw_gyro_data = np.array([np.array([i['gyro_data_x'][0], i['gyro_data_y'][0], i['gyro_data_z'][0]]) for i in raw_imu_data])

    madgwick_filter = Madgwick(gyr=np.deg2rad(raw_gyro_data), acc=9.81*raw_inc_data, Dt=T)

    rp_list = []
    
    for q in madgwick_filter.Q:
        rpy = Quaternion(q).to_angles()
        rp_list.append(rpy)

    return np.array(rp_list)


# --------------------------------------------------------------
# ---------------- DATA MANIPULATION FUNCTIONS -----------------
# --------------------------------------------------------------


def create_circle_array(n: int) -> np.ndarray:
    array = np.zeros((n, n), np.uint8)
    center = (n//2, n//2)
    for y in range(n):
        for x in range(n):
            distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if distance < n / 2:
                array[y, x] = 1
    return array


def rpy2vec(phi: float, theta: float, psi: float) -> np.ndarray:
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]])

    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]])
    
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]])
    
    return R_z @ R_y @ R_x @ np.array([0, 0, -1])


def wrap_angle_2pi(angle: float) -> float:
    return (angle + 2*np.pi) % (2*np.pi)


def wrap_angle_180(angle: float) -> float:
    angle = angle % 360
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    return angle


def wrap_angle_90(angle: float) -> float:
    angle = wrap_angle_180(angle)
    if angle > 90:
        angle = 180 - angle
    elif angle < -90:
        angle = -180 - angle
    return angle


def azze2vec(az: float, ze: float) -> np.ndarray:
    x = np.sin(ze)*np.cos(az)
    y = np.sin(ze)*np.sin(az)
    z = np.cos(ze)

    return np.array([x,y,z])


# --------------------------------------------------------------
# ---------------- SUN AND HEADING CALCULATIONS ----------------
# --------------------------------------------------------------


def find_sun_img(img: np.ndarray) -> np.ndarray:
    eps = 1e-5
    gray_smooth = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    _, thresh = cv2.threshold(gray_smooth, 220, 255, cv2.THRESH_BINARY)

    kernel_erode = create_circle_array(5)
    eroded = cv2.erode(thresh, kernel_erode, iterations=1)

    kernel_dilate = create_circle_array(5)
    dilated = cv2.dilate(eroded, kernel_dilate, iterations=1)

    m = cv2.moments(dilated, True)
    center = np.array([int(m['m10'] / (m['m00']+eps)), int(m['m01'] / (m['m00']+eps))])
    area = m['m00']
    
    if center.all() == 0 or area > 10000:
        return None

    return center


def find_sun_b(sun_coords_px: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    sun_c = np.linalg.inv(intrinsics) @ np.append(sun_coords_px, 1)
    sun_c = sun_c / np.linalg.norm(sun_c)
    
    R_b_c = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]])

    sun_b = R_b_c @ sun_c

    return sun_b


def find_sun_n(sun_b: np.ndarray, imu_data: dict, time: pd.DatetimeIndex, coordinates: np.ndarray = None, solarpos: pd.DataFrame = None, imu_filter: str = None) -> dict:
    
    valid_imu_filters = [None, 'complementary', 'madgwick']

    if imu_filter not in valid_imu_filters:
        raise ValueError(f'Invalid imu_filter: {imu_filter}. Valid options are: {valid_imu_filters}')

    if imu_filter is None:
        phi = imu_data['roll_b']
        theta = imu_data['pitch_b']
    elif imu_filter == 'madgwick':
        phi = imu_data['roll_madgwick_b']
        theta = imu_data['pitch_madgwick_b']
    elif imu_filter == 'complementary':
        phi = imu_data['roll_comp_b']
        theta = imu_data['pitch_comp_b']

    if solarpos is None:
        # Function from pvlib
        lat, lon = coordinates
        solarpos = get_solarposition(time, lat, lon, altitude=0)
    
    sun_az_ned = np.deg2rad(solarpos.azimuth.values[0])
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]])

    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]])

    R_l_b = R_y @ R_x    
    sun_l = R_l_b @ sun_b

    psi = sun_az_ned + np.arctan2(sun_l[1], sun_l[0])

    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]])
    

    R_flip = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]]) 
    
    R_n_b = R_flip @ R_z @ R_y @ R_x

    sun_n = R_n_b @ sun_b

    x_b = R_n_b @ np.array([1, 0, 0])
    heading_rad = np.arctan2(x_b[1], x_b[0])
    heading_rad = (2*np.pi - heading_rad) % (2*np.pi) #adjust to range [0, 2pi) rad

    calculated_zenith_rad = np.arccos(sun_l[2])
    calculated_elevation_rad = np.arcsin(sun_l[2])

    correct_elevation_deg = solarpos.apparent_elevation.values[0]
    correct_zenith_deg = 90 - correct_elevation_deg

    return {'sun_azimuth_rad': sun_az_ned,
            'sun_zenith_rad': np.deg2rad(correct_zenith_deg),
            'sun_b': sun_b,
            'sun_l': sun_l,
            'sun_n': sun_n,
            'g_b': imu_data['g_b'],
            'heading_deg': np.rad2deg(heading_rad),
            'ref_heading_deg': np.rad2deg(imu_data['ref_heading_rad']),
            'calculated_elevation_deg': np.rad2deg(calculated_elevation_rad), 
            'calculated_zenith_deg': np.rad2deg(calculated_zenith_rad), 
            'correct_elevation_deg': correct_elevation_deg,
            'correct_zenith_deg': correct_zenith_deg, 
            'R_n_b': R_n_b}


def haversine_earth_km(coords1: list, coords2: list) -> float:
    r_earth = 6371 # average radius of Earth
    phi1, lambda1 = np.deg2rad(coords1)
    phi2, lambda2 = np.deg2rad(coords2)
    return 2*r_earth*np.arcsin(np.sqrt((1-np.cos(phi2-phi1) + np.cos(phi1)*np.cos(phi2)*(1-np.cos(lambda2-lambda1)))/2))


# --------------------------------------------------------------
# ----------------- CAMERA AND VIDEO FUNCTIONS -----------------
# --------------------------------------------------------------


def split_and_undistort(folder_path: str, intrinsics: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    video_in = ['video', 'video_high']  

    for video in video_in:
        input_path = folder_path + video
        cap = cv2.VideoCapture(input_path+'.mp4')

        if not cap.isOpened():
            print("Error: Could not open video.")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2

        new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist_coeffs, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, new_intrinsics, (w,h), 5)

        output_path = input_path+'_left.avi'
        
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out_left = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame is None:
                cap.release()
                print("Released Video Resource")
                break

            left = frame[:h,:]

            left_undistorted = cv2.remap(left, mapx, mapy, cv2.INTER_LINEAR)

            x, y, w_new, h_new = roi
            left_cropped = left_undistorted[y:y+h_new, x:x+w_new]
            left_resized = cv2.resize(left_cropped, (w,h))

            out_left.write(left_resized)

        cap.release()
        out_left.release()
        cv2.destroyAllWindows()

    return new_intrinsics


def loadImgs(path: str, pattern_size: tuple, pattern_pts: np.ndarray):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print('Failed to load', path)
        return None

    found, corners = cv2.findChessboardCorners(img, pattern_size)

    if found:
        term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1)
        cv2.cornerSubPix(img, corners, (5,5), (-1,-1), term_crit)
    else:
        print('Chessboard not found')
        return None
    
    print(path, 'OK')
    return (corners.reshape(-1,2), pattern_pts)


def calibrate_camera(path: str, square_size: float, pattern_size: tuple) -> np.array:

    indices = np.indices(pattern_size, dtype=np.float32)*square_size
    coords_3D = np.transpose(indices, [2,1,0]).reshape(-1,2)
    pattern_pts = np.concatenate([coords_3D, np.zeros([coords_3D.shape[0], 1], dtype=np.float32)], axis=1)

    img_names = glob.glob(path + '*.jpg')

    chessboards = [loadImgs(i, pattern_size, pattern_pts) for i in img_names]
    chessboards = [x for x in chessboards if x is not None]

    obj_pts = [] #3D points
    img_pts = [] #2D points

    for(corners, pattern_points) in chessboards:
        img_pts.append(corners)
        obj_pts.append(pattern_points)

    h, w = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[:2]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w,h), None, None)

    return camera_matrix, dist_coeffs


def get_chessboard_imgs(video_path: str) -> str:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    img_path_left = video_path[:video_path.rfind('/')+1]+'chess_imgs_left/'
    img_path_right = video_path[:video_path.rfind('/')+1]+'chess_imgs_right/'

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            cap.release()
            print("Released Video Resource")
            break
        
        if 128 < i < 390 and i % 2 == 0:
            left = frame[:height//2,:]
            cv2.imwrite(img_path_left+str(i)+'.jpg', left)
        
        if 410 < i < 545 and i % 2 == 0:
            right = frame[height//2:,:]
            cv2.imwrite(img_path_right+str(i)+'.jpg', right)

        i += 1

    cap.release()
    cv2.destroyAllWindows()

    return img_path_left, img_path_right


def calibrate_and_save_params() -> None:
    square_size = 100.7 #mm
    pattern_size = (6,4)
    path_imgs_left = 'datasets/calib/chess_imgs_left/'

    intrinsics_left, dist_coeffs_left = calibrate_camera(path_imgs_left, square_size, pattern_size)

    left = 'datasets/calib/parameters_left.npy'
    with open(left, 'wb') as f:
        np.save(f, intrinsics_left)
        np.save(f, dist_coeffs_left)
    return