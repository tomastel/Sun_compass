import cv2
import json
import pytz
import myfuncs
import numpy as np
import scipy.optimize as scipy
from datetime import datetime
from pandas import DatetimeIndex
from pvlib.solarposition import get_solarposition

import symforce
symforce.set_log_level('ERROR')
import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer


def sun_residual(R_n_b:sf.Rot3, sun_b:sf.V3, sun_n_opt:sf.V3, sun_cov:sf.M33) -> sf.V2:
    sun_n_pred = R_n_b * sun_b
    sun_n_pred_unit = sf.Unit3.from_vector(sun_n_pred)
    sun_n_opt_unit = sf.Unit3.from_vector(sun_n_opt)

    diff = sf.V2(sun_n_opt_unit.local_coordinates(sun_n_pred_unit))
    diff_D = diff.jacobian(sun_n_opt)

    diff_cov = diff_D * sun_cov * diff_D.T
    diff_sqrt_info = sf.M22(diff_cov.inv().mat.cholesky())
    
    return diff_sqrt_info * diff


def imu_residual(R_n_b:sf.Rot3, g_b:sf.V3, g_n:sf.V3, g_cov:sf.M33) -> sf.V2:
    g_n_pred = R_n_b.inverse()*g_b
    g_n_pred_unit = sf.Unit3.from_vector(g_n_pred)
    g_n_unit = sf.Unit3.from_vector(g_n) # [0,0,1]

    diff = sf.V2(g_n_unit.local_coordinates(g_n_pred_unit))
    diff_D = diff.jacobian(g_n)

    diff_cov = diff_D * g_cov * diff_D.T
    diff_sqrt_info = sf.M22(diff_cov.inv().mat.cholesky())

    return diff_sqrt_info * diff


def estimate_rotation(sun: dict, optimizer: sf.Optimizer) -> np.ndarray:
    initial_values = Values(
        R_n_b = sf.Rot3.from_rotation_matrix(np.eye(3)),
        sun_b = sf.V3(sun['sun_b']),
        sun_n_opt = sf.V3(myfuncs.azze2vec(sun['sun_azimuth_rad'], sun['sun_zenith_rad'])),
        g_b = sf.V3(sun['g_b']),
        g_n = sf.V3(0, 0, 1),
        sun_b_cov = sf.M33.eye(3)*.3,
        g_b_cov = sf.M33.eye(3)*.7,
        )

    result = optimizer.optimize(initial_values)

    if result.status != Optimizer.Status.SUCCESS:
        print(result.status)

    return result.optimized_values['R_n_b'].to_rotation_matrix()


def find_coordinates(actual_heading_deg, est_elev_deg, sun_body, imu_data, time, x0):
    def obj(x):
        lat, lon = x
        solarpos = get_solarposition(time, lat, lon, altitude=0)

        est_sun_w = myfuncs.find_sun_n(sun_body, imu_data, time, solarpos = solarpos)
        est_heading_deg = est_sun_w['heading_deg']
        actual_elev_deg = est_sun_w['correct_elevation_deg']
        
        res1 = (est_heading_deg - actual_heading_deg)**2
        res2 = (est_elev_deg - actual_elev_deg)**2

        return res1 + res2

    perturbation = 0.1
    initial_simplex = np.array([x0,
                                x0 + np.array([perturbation,0]),
                                x0 + np.array([0,perturbation])])

    result = scipy.minimize(obj, x0=x0, method='Nelder-Mead', tol=1e-2,
                            options={'initial_simplex':initial_simplex})

    assert result.success
    optimized_lat, optimized_lon = result.x
    return {'lat': optimized_lat, 'lon': optimized_lon}, result.nit


def main():
    folder_path = 'datasets/2024-05-15T12-33-07/'

    # --------------- Read and sort/clean raw data ----------------

    with open((folder_path + 'img_data.json'), 'r') as file:
        raw_metadata = np.array(json.load(file))

    gps_data = myfuncs.read_gt_data(folder_path + 'reference_headings.csv')
    raw_imu_data = np.load(folder_path+'nav_data.npz')['stim_data']
    sorted_imu_data = myfuncs.sort_data(raw_imu_data, raw_metadata, gps_data)

    # ------------------- Get camera parameters --------------------

    with open('datasets/calib/parameters_left.npy', 'rb') as f:
        intrinsics = np.load(f)
        dist_coeffs = np.load(f)

    # ----------------- Split and undistort video ------------------

    new_intrinsics = myfuncs.split_and_undistort(folder_path, intrinsics, dist_coeffs)

    # ----------------- Loose-coupling estimation ------------------

    dt_raw = datetime.fromtimestamp(sorted_imu_data[0]['ts_sec'])
    dt_local = pytz.timezone('Europe/Oslo').localize(dt_raw)
    dt = DatetimeIndex([dt_local])

    coords = np.array([63.440575, 0.413613]) # [lat, lon] at Skippergata 14, 7042 Trondheim

    video_path = folder_path + 'video_left.avi'
    cap = cv2.VideoCapture(video_path)
    max_steps = len(sorted_imu_data)

    sun_px_list, sun_list = [], []

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            print("Released Video Resource")
            break
            
        sun_px = myfuncs.find_sun_img(frame)
        if sun_px is None:
            sun_px_list.append(None)
            sun_list.append({'ref_heading_deg': np.rad2deg(sorted_imu_data[i]['ref_heading_rad']), 'heading_deg': None})
        else:
            sun_b = myfuncs.find_sun_b(sun_px, new_intrinsics)
            sun_n = myfuncs.find_sun_n(sun_b, sorted_imu_data[i], dt, coords, imu_filter='madgwick')

            sun_px_list.append(sun_px)
            sun_list.append(sun_n)

        i += 1
        if i >= max_steps-1:
            break

    cap.release()

    # ----------- Tight-coupling estimation (Symforce) ------------

    factors = []

    factors.append(Factor(
        residual = sun_residual,
        keys = ['R_n_b', 'sun_b', 'sun_n_opt', 'sun_b_cov']))

    factors.append(Factor(
        residual = imu_residual,
        keys = ['R_n_b', 'g_b', 'g_n', 'g_b_cov']))
        
    optimizer = Optimizer(
        factors = factors,
        optimized_keys = ['R_n_b'],
        debug_stats = True,
        params = Optimizer.Params(iterations = 500))
            
    R_n_b_est_list = []

    for i, sun in enumerate(sun_list):
        if sun['heading_deg'] is None:
            R_n_b_est_list.append(None)
            sun['estimated_heading_deg'] = None
        else:
            sun_b = sun['sun_b']
            g_b = sun['g_b']
            R_n_b_est = estimate_rotation(optimizer, sun)
            x_b = R_n_b_est @ np.array([1,0,0]) # x-axis of body expressed in NED
            heading_est = np.arctan2(x_b[1], x_b[0])
            heading_est = np.rad2deg(myfuncs.wrap_angle_2pi(heading_est))
            sun['estimated_heading_deg'] = heading_est
            sun_n_est = R_n_b_est @ sun['sun_b']
            elevation_est = np.arcsin(sun_n_est[2])
            
            R_n_b_est_list.append(R_n_b_est)

    # ----------------- Position estimation ------------------

    ref_coords = [63.440575, 10.413613]

    dt_raw = datetime.fromtimestamp(sorted_imu_data[0]['ts_sec'])
    dt_local = pytz.timezone('Europe/Oslo').localize(dt_raw)
    dt = DatetimeIndex([dt_local])

    x0 = [90, 0]

    d_list = []
    est_coords_list = []
    for i, sun in enumerate(sun_list):
        if sun['heading_deg'] is not None:
            heading = sun['ref_heading_deg']
            elev = sun['calculated_elevation_deg']
            sun_body = sun['sun_b']
            imu = sorted_imu_data[i]

            est_coords, nit = find_coordinates(heading, elev, sun_body, imu, dt, x0)
            est_coords = [myfuncs.wrap_angle_90(est_coords['lat']), myfuncs.wrap_angle_180(est_coords['lon'])]
            est_coords_list.append(est_coords)
            d = myfuncs.haversine_earth_km(ref_coords, est_coords)
            d_list.append(d)
            x0 = est_coords
        else:
            d_list.append(np.nan)
            
    d_list = np.array(d_list)


if __name__ == "__main__":
    main()