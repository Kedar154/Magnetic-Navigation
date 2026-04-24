"""
MagNav Flight Dashboard — Flask backend
Implements the FULL pipeline from magnav_final.ipynb:
  1. Load real HDF5 flight data (Flt1002–Flt1007)
  2. Tolles-Lawson (TL) compensation
  3. MLP Neural-Network (NN) compensation  ← trains on flights != test flight
  4. Real EKF map-matching with EMAG2 (emag_up.tif) for THREE tracks:
       EKF+IGRF · EKF+TL · EKF+NN
  5. Serve everything to the dashboard as JSON

Install:
  pip install flask h5py numpy pandas scipy scikit-learn rasterio

Place emag_up.tif on your Desktop (same folder as the data/ directory).
Run:
  python app.py
Open: http://localhost:5050
"""

import os, glob, json, traceback
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# ── Config ──────────────────────────────────────────────────────────────────
DESKTOP   = os.path.expanduser("~/Desktop/")
DATA_DIR  = os.path.join(DESKTOP, "data/")
MAP_PATH  = os.path.join(DESKTOP, "emag_up.tif")   # EMAG2 anomaly raster

KEEP_COLS = [
    'flight', 'line', 'tt',
    'flux_b_x', 'flux_b_y', 'flux_b_z', 'flux_b_t',
    'mag_1_uc', 'mag_1_lag', 'mag_1_c', 'mag_1_igrf',
    'ins_lat', 'ins_lon', 'ins_alt',
    'ins_pitch', 'ins_roll', 'ins_yaw',
    'ins_vn', 'ins_vw', 'ins_vu',
    'ins_acc_x', 'ins_acc_y', 'ins_acc_z',
    'lat', 'lon', 'utm_x', 'utm_y', 'diurnal',
]

R_EARTH  = 6_378_137.0
DEG2RAD  = np.pi / 180.0
RAD2DEG  = 180.0 / np.pi
AC_WIN   = 200          # uniform-filter window for AC extraction

# ── Global cache ─────────────────────────────────────────────────────────────
_mag_map   = None          # rasterio interpolator (lazy-loaded)
_nn_cache  = {}            # {train_flight_key: (scaler_X, scaler_y, nn_model)}
_df_cache  = {}            # {flight_id: DataFrame}

# ── HDF5 loader ──────────────────────────────────────────────────────────────
def load_h5(path):
    with h5py.File(path, 'r') as f:
        available = set(f.keys())
        cols = [c for c in KEEP_COLS if c in available]
        return pd.DataFrame({c: f[c][:] for c in cols})

def get_flight_df(flight_id):
    if flight_id not in _df_cache:
        path = os.path.join(DATA_DIR, f"Flt{flight_id}_train.h5")
        df   = load_h5(path)
        flt  = float(flight_id)
        _df_cache[flight_id] = df[df['flight'] == flt].copy().reset_index(drop=True)
    return _df_cache[flight_id]

# ── EMAG2 map ─────────────────────────────────────────────────────────────────
def get_mag_map():
    global _mag_map
    if _mag_map is not None:
        return _mag_map
    if not os.path.exists(MAP_PATH):
        return None
    try:
        import rasterio
        from scipy.interpolate import RegularGridInterpolator
        with rasterio.open(MAP_PATH) as src:
            data = src.read(1).astype(np.float64)
            rows, cols = data.shape
            lons = src.transform[2] + np.arange(cols) * src.transform[0]
            lats = src.transform[5] + np.arange(rows) * src.transform[4]
            if lats[0] > lats[-1]:
                lats = lats[::-1]
                data = np.flipud(data)
            _mag_map = RegularGridInterpolator(
                (lats, lons), data, bounds_error=False, fill_value=0.0)
        print(f"   EMAG2 map loaded: lat {_mag_map.grid[0].min():.1f}–{_mag_map.grid[0].max():.1f}")
        return _mag_map
    except Exception as e:
        print(f"   Warning: could not load EMAG2 map: {e}")
        return None

def map_query(mag_map, lat_deg, lon_deg):
    lon_w = float(lon_deg) % 360.0
    return float(mag_map([[float(lat_deg), lon_w]])[0])

def map_gradient(mag_map, lat_deg, lon_deg, step_m=2000.0):
    dlat_deg = step_m / (R_EARTH * DEG2RAD)
    dlon_deg = step_m / (R_EARTH * np.cos(lat_deg * DEG2RAD) * DEG2RAD)
    val_c = map_query(mag_map, lat_deg, lon_deg)
    val_n = map_query(mag_map, lat_deg + dlat_deg, lon_deg)
    val_e = map_query(mag_map, lat_deg, lon_deg + dlon_deg)
    return (val_n - val_c) / step_m, (val_e - val_c) / step_m, val_c

# ── Tolles-Lawson matrix ─────────────────────────────────────────────────────
def build_tl_matrix(df):
    bx = df['flux_b_x'].values.astype(np.float64)
    by = df['flux_b_y'].values.astype(np.float64)
    bz = df['flux_b_z'].values.astype(np.float64)
    bt = np.sqrt(bx**2 + by**2 + bz**2)
    bt = np.where(bt == 0, 1e-10, bt)
    cx, cy, cz = bx/bt, by/bt, bz/bt

    def deriv(x): return np.gradient(x)
    dcx, dcy, dcz = deriv(cx), deriv(cy), deriv(cz)

    cols = [
        cx, cy, cz,
        bt*cx**2, bt*cx*cy, bt*cx*cz,
        bt*cy**2, bt*cy*cz, bt*cz**2,
        bt*cx*dcx, bt*cx*dcy, bt*cx*dcz,
        bt*cy*dcy, bt*cy*dcz, bt*cz*dcz,
        bt*cy*dcx, bt*cz*dcx, bt*cz*dcy,
    ]
    return np.column_stack(cols)

def extract_ac(signal, window=AC_WIN):
    trend = uniform_filter1d(signal.astype(float), size=window)
    return signal - trend

def tl_compensate(df):
    """Ridge TL compensation → returns mag_1_tl series."""
    from sklearn.linear_model import Ridge
    A = build_tl_matrix(df)
    y = df['mag_1_lag'].values.astype(np.float64)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(A, y)
    pred = model.predict(A)
    tl_ac = pred - pred.mean()
    return y - tl_ac

# ── NN model (train on all flights except the test flight) ────────────────────
def get_nn_model(test_flight_id):
    """Train/cache MLP on all flights except test_flight_id. Returns (scaler_X, scaler_y, model)."""
    key = str(test_flight_id)
    if key in _nn_cache:
        return _nn_cache[key]

    print(f"   Training NN (leaving out flight {test_flight_id})…")
    try:
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        train_dfs = []
        for path in sorted(glob.glob(os.path.join(DATA_DIR, "Flt*_train.h5"))):
            fid = os.path.basename(path).replace("Flt","").replace("_train.h5","")
            if fid == str(test_flight_id):
                continue
            try:
                df_f = load_h5(path)
                flt  = float(fid)
                df_f = df_f[df_f['flight'] == flt].copy().reset_index(drop=True)
                # need TL columns
                df_f['mag_1_tl'] = tl_compensate(df_f)
                train_dfs.append(df_f)
            except Exception as e:
                print(f"     Skipping {fid}: {e}")

        if not train_dfs:
            _nn_cache[key] = None
            return None

        df_train = pd.concat(train_dfs, ignore_index=True)

        A_train   = build_tl_matrix(df_train)
        att_train = df_train[['ins_pitch', 'ins_roll', 'ins_yaw']].values
        X_train   = np.hstack([A_train, att_train])
        y_train   = extract_ac(df_train['mag_1_lag'].values)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_sc = scaler_X.fit_transform(X_train)
        y_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        nn = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu',
            max_iter=100, early_stopping=True, validation_fraction=0.1,
            verbose=False, random_state=42
        )
        nn.fit(X_sc, y_sc)
        _nn_cache[key] = (scaler_X, scaler_y, nn)
        print(f"   NN trained. Validation score: {nn.best_validation_score_:.4f}")
        return _nn_cache[key]
    except Exception as e:
        print(f"   NN training failed: {e}")
        _nn_cache[key] = None
        return None

def nn_compensate(df_test, test_flight_id):
    """Apply NN model to test segment → returns mag_1_nn array."""
    result = get_nn_model(test_flight_id)
    if result is None:
        return tl_compensate(df_test)          # fallback
    scaler_X, scaler_y, nn = result
    A    = build_tl_matrix(df_test)
    att  = df_test[['ins_pitch', 'ins_roll', 'ins_yaw']].values
    X    = np.hstack([A, att])
    X_sc = scaler_X.transform(X)
    y_sc = nn.predict(X_sc)
    y_ac = scaler_y.inverse_transform(y_sc.reshape(-1, 1)).ravel()
    return df_test['mag_1_lag'].values - y_ac

# ── Real EKF (from notebook cell 21) ─────────────────────────────────────────
def run_ekf(ins_lat_deg, ins_lon_deg, z_all, mag_map, dt,
            R_noise=2150.0, q_drift=None, label='EKF'):
    """2-state position-correction EKF for magnetic map navigation."""
    N = len(ins_lat_deg)
    if q_drift is None:
        q_drift = (0.5 * dt) ** 2

    avg_lat_rad   = np.mean(ins_lat_deg) * DEG2RAD
    m_per_deg_lat = R_EARTH * DEG2RAD
    m_per_deg_lon = R_EARTH * np.cos(avg_lat_rad) * DEG2RAD

    corr_n  = np.zeros(N)
    corr_e  = np.zeros(N)
    mag_bias_arr = np.zeros(N)
    innov   = np.zeros(N)
    P_trace = np.zeros(N)

    x = np.array([0.0, 0.0])
    F = np.eye(2)
    P = np.diag([300.0**2, 300.0**2])
    Q = np.diag([q_drift, q_drift])
    R = np.array([[float(R_noise)]])

    start_map_val = map_query(mag_map, ins_lat_deg[0], ins_lon_deg[0])
    bias          = float(z_all[0] - start_map_val)
    alpha_bias    = 0.95

    mag_bias_arr[0] = bias
    P_trace[0]      = np.trace(P)

    for k in range(1, N):
        est_lat = ins_lat_deg[k] + x[0] / m_per_deg_lat
        est_lon = ins_lon_deg[k] + x[1] / m_per_deg_lon

        x = F @ x
        P = F @ P @ F.T + Q

        grad_n, grad_e, map_val = map_gradient(mag_map, est_lat, est_lon)

        if np.sqrt(grad_n**2 + grad_e**2) < 1e-6:
            corr_n[k] = x[0]; corr_e[k] = x[1]
            innov[k]  = 0.0;  P_trace[k] = np.trace(P)
            mag_bias_arr[k] = bias
            continue

        H          = np.array([[grad_n, grad_e]])
        z_pred     = map_val + bias
        innovation = float(z_all[k]) - z_pred

        bias = alpha_bias * bias + (1 - alpha_bias) * float(z_all[k] - map_val)

        S  = float(H @ P @ H.T + R)
        K  = (P @ H.T) / S
        x  = x + K.ravel() * innovation

        ImKH = np.eye(2) - K @ H
        P    = ImKH @ P @ ImKH.T + float(R_noise) * (K @ K.T)
        x    = np.clip(x, -800.0, 800.0)

        corr_n[k]       = x[0]
        corr_e[k]       = x[1]
        innov[k]        = innovation
        P_trace[k]      = np.trace(P)
        mag_bias_arr[k] = bias

    ekf_lat = ins_lat_deg + corr_n / m_per_deg_lat
    ekf_lon = ins_lon_deg + corr_e / m_per_deg_lon

    return {
        'label': label,
        'lat': ekf_lat, 'lon': ekf_lon,
        'corr_n': corr_n, 'corr_e': corr_e,
        'innov': innov, 'P_trace': P_trace,
        'mag_bias': mag_bias_arr,
    }

def pos_error_m(est_lat, est_lon, gt_lat, gt_lon):
    avg_lat = np.mean(gt_lat) * DEG2RAD
    dn = (est_lat - gt_lat) * R_EARTH * DEG2RAD
    de = (est_lon - gt_lon) * R_EARTH * np.cos(avg_lat) * DEG2RAD
    return np.sqrt(dn**2 + de**2)

# ── Downsampling helper ───────────────────────────────────────────────────────
def to_list(arr, max_pts=4000):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n > max_pts:
        idx = np.linspace(0, n-1, max_pts, dtype=int)
        arr = arr[idx]
    arr = np.where(np.isfinite(arr), arr, None)
    return arr.tolist()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/flights")
def list_flights():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Flt*.h5")))
    result = []
    for path in files:
        name   = os.path.basename(path)
        flt_id = name.replace("Flt", "").replace("_train.h5", "")
        result.append({"id": flt_id, "file": name})
    return jsonify(result)

@app.route("/api/flight/<flight_id>")
def flight_info(flight_id):
    path = os.path.join(DATA_DIR, f"Flt{flight_id}_train.h5")
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    df   = get_flight_df(flight_id)
    lines = sorted(df['line'].unique())
    line_info = []
    for ln in lines:
        seg = df[df['line'] == ln]
        dur = float(seg['tt'].max() - seg['tt'].min())
        line_info.append({
            "line": float(ln),
            "rows": int(len(seg)),
            "duration_s": round(dur, 1),
            "is_cal": dur > 2000
        })
    return jsonify({"flight": flight_id, "lines": line_info})

@app.route("/api/flight/<flight_id>/line/<line_id>")
def line_data(flight_id, line_id):
    try:
        path = os.path.join(DATA_DIR, f"Flt{flight_id}_train.h5")
        if not os.path.exists(path):
            return jsonify({"error": "File not found"}), 404

        df_full = get_flight_df(flight_id)
        ln = float(line_id)
        df = df_full[df_full['line'] == ln].copy().reset_index(drop=True)
        if len(df) == 0:
            return jsonify({"error": "No data for this line"}), 404

        # ── Time ────────────────────────────────────────────────────────────
        t0 = df['tt'].values[0]
        t  = df['tt'].values - t0
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1

        # ── Convert INS lat/lon (radians → degrees) ──────────────────────
        ins_lat = df['ins_lat'].values
        ins_lon = df['ins_lon'].values
        # detect radians: lat should be roughly 44–45° for this dataset
        if np.abs(ins_lat).max() < 2.0:   # radians
            ins_lat = ins_lat * RAD2DEG
            ins_lon = ins_lon * RAD2DEG

        gps_lat = df['lat'].values
        gps_lon = df['lon'].values

        # ── TL compensation ────────────────────────────────────────────────
        mag_tl = tl_compensate(df)
        df['mag_1_tl'] = mag_tl

        # ── NN compensation ────────────────────────────────────────────────
        mag_nn = nn_compensate(df, flight_id)
        df['mag_1_nn'] = mag_nn

        # ── EKF (3 variants) ───────────────────────────────────────────────
        mag_map = get_mag_map()

        ekf_igrf_result = ekf_tl_result = ekf_nn_result = None
        has_ekf = mag_map is not None

        if has_ekf:
            z_igrf = df['mag_1_igrf'].values.astype(np.float64) if 'mag_1_igrf' in df.columns else None
            z_tl   = mag_tl - mag_tl.mean()
            z_nn   = mag_nn - mag_nn.mean()

            if z_igrf is not None:
                ekf_igrf_result = run_ekf(ins_lat, ins_lon, z_igrf, mag_map, dt, label='EKF+IGRF')
            ekf_tl_result = run_ekf(ins_lat, ins_lon, z_tl, mag_map, dt, label='EKF+TL')
            ekf_nn_result = run_ekf(ins_lat, ins_lon, z_nn, mag_map, dt, label='EKF+NN')

        # ── Position errors ────────────────────────────────────────────────
        ins_err = pos_error_m(ins_lat, ins_lon, gps_lat, gps_lon)

        def ekf_err(r):
            if r is None: return None
            return pos_error_m(r['lat'], r['lon'], gps_lat, gps_lon)

        err_igrf = ekf_err(ekf_igrf_result)
        err_tl   = ekf_err(ekf_tl_result)
        err_nn   = ekf_err(ekf_nn_result)

        # ── Attitude ───────────────────────────────────────────────────────
        pitch = df['ins_pitch'].values
        roll  = df['ins_roll'].values
        yaw   = df['ins_yaw'].values
        if np.abs(pitch).max() < 0.5:      # likely radians
            pitch = pitch * RAD2DEG
            roll  = roll  * RAD2DEG
            yaw   = yaw   * RAD2DEG

        # ── Stats ──────────────────────────────────────────────────────────
        def safe_mean(arr):
            if arr is None: return None
            return round(float(np.nanmean(arr)), 1)

        def safe_rmse(a, b):
            if a is None or b is None: return None
            diff = (a - a.mean()) - (b - b.mean())
            return round(float(np.sqrt(np.mean(diff**2))), 2)

        ref = df['mag_1_c'].values if 'mag_1_c' in df.columns else None

        stats = {
            "n_points":     int(len(df)),
            "duration_s":   round(float(t[-1]), 1),
            "has_ekf":      has_ekf,
            "raw_std":      round(float(df['mag_1_lag'].std()), 2),
            "tl_std":       round(float(mag_tl.std()), 2),
            "nn_std":       round(float(mag_nn.std()), 2),
            "mit_std":      round(float(ref.std()), 2) if ref is not None else None,
            "tl_rmse":      safe_rmse(mag_tl, ref) if ref is not None else None,
            "nn_rmse":      safe_rmse(mag_nn, ref) if ref is not None else None,
            "ins_err_mean_m":    safe_mean(ins_err),
            "ekf_igrf_err_m":    safe_mean(err_igrf),
            "ekf_tl_err_m":      safe_mean(err_tl),
            "ekf_nn_err_m":      safe_mean(err_nn),
        }

        MAX = 4000
        payload = {
            "stats": stats,
            "t":         to_list(t, MAX),
            # Mag signals
            "mag_raw":   to_list(df['mag_1_lag'].values, MAX),
            "mag_tl":    to_list(mag_tl, MAX),
            "mag_nn":    to_list(mag_nn, MAX),
            "mag_mit":   to_list(ref, MAX) if ref is not None else None,
            "mag_igrf":  to_list(df['mag_1_igrf'].values, MAX) if 'mag_1_igrf' in df.columns else None,
            # GPS (ground truth)
            "gps_lat":   to_list(gps_lat, MAX),
            "gps_lon":   to_list(gps_lon, MAX),
            # INS dead-reckoning
            "ins_lat":   to_list(ins_lat, MAX),
            "ins_lon":   to_list(ins_lon, MAX),
            # EKF tracks
            "ekf_igrf_lat": to_list(ekf_igrf_result['lat'], MAX) if ekf_igrf_result else None,
            "ekf_igrf_lon": to_list(ekf_igrf_result['lon'], MAX) if ekf_igrf_result else None,
            "ekf_tl_lat":   to_list(ekf_tl_result['lat'],  MAX) if ekf_tl_result  else None,
            "ekf_tl_lon":   to_list(ekf_tl_result['lon'],  MAX) if ekf_tl_result  else None,
            "ekf_nn_lat":   to_list(ekf_nn_result['lat'],  MAX) if ekf_nn_result  else None,
            "ekf_nn_lon":   to_list(ekf_nn_result['lon'],  MAX) if ekf_nn_result  else None,
            # EKF internals
            "ekf_igrf_innov": to_list(ekf_igrf_result['innov'],   MAX) if ekf_igrf_result else None,
            "ekf_tl_innov":   to_list(ekf_tl_result['innov'],     MAX) if ekf_tl_result  else None,
            "ekf_nn_innov":   to_list(ekf_nn_result['innov'],     MAX) if ekf_nn_result  else None,
            # Position errors
            "ins_err":       to_list(ins_err,  MAX),
            "err_igrf":      to_list(err_igrf, MAX) if err_igrf is not None else None,
            "err_tl":        to_list(err_tl,   MAX) if err_tl   is not None else None,
            "err_nn":        to_list(err_nn,   MAX) if err_nn   is not None else None,
            # Attitude
            "pitch": to_list(pitch,               MAX),
            "roll":  to_list(roll,                MAX),
            "yaw":   to_list(yaw,                 MAX),
            "alt":   to_list(df['ins_alt'].values, MAX),
            "vn":    to_list(df['ins_vn'].values,  MAX) if 'ins_vn' in df.columns else None,
            "ve":    to_list(df['ins_vw'].values,  MAX) if 'ins_vw' in df.columns else None,
        }
        return jsonify(payload)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("\n🧭  MagNav Dashboard  (Real EKF Pipeline)")
    print(f"   Data dir : {DATA_DIR}")
    files = glob.glob(os.path.join(DATA_DIR, "Flt*.h5"))
    print(f"   Flights  : {len(files)} file(s) found")
    print(f"   EMAG2    : {'✓ found' if os.path.exists(MAP_PATH) else '✗ NOT found — EKF will be disabled'}")
    print("   Open     : http://localhost:5050\n")
    app.run(debug=True, port=5050)
