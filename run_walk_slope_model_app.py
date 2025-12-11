import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import math
import altair as alt

# ---------------------------------------------------------
# НАСТРОЙКИ (за бягане и ходене)
# ---------------------------------------------------------
T_SEG = 30.0           # дължина на сегмента [s]
MIN_D_SEG = 10.0       # минимум хоризонтална дистанция [m]
MIN_T_SEG = 15.0       # минимум продължителност [s]
MAX_ABS_SLOPE = 15.0   # макс. наклон [%]

V_JUMP_KMH = 15.0      # праг за "скачане" на скоростта между сегменти
V_JUMP_MIN = 10.0      # гледаме спайкове само над тази скорост [km/h]

SLOPE_POLY_DEG = 2     # степен на полинома за наклон

# Зонна система като % от критичната скорост
ZONE_BOUNDS = [0.0, 0.75, 0.85, 0.95, 1.05, 1.15, np.inf]
ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]

# Граници за наклоновия коефициент F(s)
F_MIN_SLOPE = 0.7
F_MAX_SLOPE = 1.7


# ---------------------------------------------------------
# ВСПОМОГАТЕЛНИ ФУНКЦИИ
# ---------------------------------------------------------
def poly_to_str(poly, var="s"):
    """Форматира np.poly1d като четим стринг (до квадратична степен)."""
    if poly is None:
        return "няма модел (недостатъчно данни)"

    coeffs = poly.coefficients
    deg = poly.order

    def fmt_coef(c):
        return f"{c:.4f}"

    if deg == 2:
        a, b, c = coeffs
        return (f"{fmt_coef(a)}·{var}² "
                f"{'+ ' if b >= 0 else '- '}{fmt_coef(abs(b))}·{var} "
                f"{'+ ' if c >= 0 else '- '}{fmt_coef(abs(c))}")
    elif deg == 1:
        a, b = coeffs
        return (f"{fmt_coef(a)}·{var} "
                f"{'+ ' if b >= 0 else '- '}{fmt_coef(abs(b))}")
    else:
        return " + ".join(
            f"{fmt_coef(c)}·{var}^{p}"
            for p, c in zip(range(deg, -1, -1), coeffs)
        )


def seconds_to_hhmmss(seconds: float) -> str:
    """Превръща секунди във формат ч:мм:сс."""
    if pd.isna(seconds):
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"


def haversine(lat1, lon1, lat2, lon2):
    """Ако се наложи да смятаме дистанция от координати (рядко за този app)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------------------------------------------------------
# TCX PARSER – за бягане / ходене
# ---------------------------------------------------------
def parse_tcx(file, activity_label):
    """
    Парсва TCX файл и връща точки:
      time, lat, lon, elev, dist, hr
    """
    content = file.read()
    tree = ET.parse(BytesIO(content))
    root = tree.getroot()

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    rows = []
    for lap in root.findall(".//tcx:Lap", ns):
        for tp in lap.findall(".//tcx:Trackpoint", ns):
            t_el = tp.find("tcx:Time", ns)
            if t_el is None:
                continue
            time = datetime.fromisoformat(t_el.text.replace("Z", "+00:00"))

            pos_el = tp.find("tcx:Position", ns)
            lat = lon = None
            if pos_el is not None:
                lat_el = pos_el.find("tcx:LatitudeDegrees", ns)
                lon_el = pos_el.find("tcx:LongitudeDegrees", ns)
                if lat_el is not None and lon_el is not None:
                    lat = float(lat_el.text)
                    lon = float(lon_el.text)

            alt_el = tp.find("tcx:AltitudeMeters", ns)
            elev = float(alt_el.text) if alt_el is not None else None

            dist_el = tp.find("tcx:DistanceMeters", ns)
            dist = float(dist_el.text) if dist_el is not None else None

            hr_el = tp.find(".//tcx:HeartRateBpm/tcx:Value", ns)
            hr = float(hr_el.text) if hr_el is not None else np.nan

            rows.append({
                "activity": activity_label,
                "time": time,
                "lat": lat,
                "lon": lon,
                "elev": elev,
                "dist": dist,
                "hr": hr,
            })

    if not rows:
        return pd.DataFrame(columns=["activity", "time", "lat", "lon", "elev", "dist", "hr"])

    df = pd.DataFrame(rows)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ако няма дистанция – смятаме от lat/lon (по-рядко, но да го има)
    if df["dist"].isna().all():
        df["dist"] = 0.0
        for i in range(1, len(df)):
            if None in (df.at[i-1, "lat"], df.at[i-1, "lon"],
                        df.at[i, "lat"], df.at[i, "lon"]):
                df.at[i, "dist"] = df.at[i-1, "dist"]
                continue
            d = haversine(
                df.at[i-1, "lat"], df.at[i-1, "lon"],
                df.at[i, "lat"], df.at[i, "lon"]
            )
            df.at[i, "dist"] = df.at[i-1, "dist"] + d

    df["dist"] = df["dist"].ffill()
    return df


# ---------------------------------------------------------
# СЕГМЕНТИРАНЕ НА 30 s (с hr_mean)
# ---------------------------------------------------------
def build_segments_30s(df_activity, activity_label):
    """
    Сегментиране на 30-секундни сегменти за бягане/ходене.
    """
    if df_activity.empty:
        return pd.DataFrame(columns=[
            "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
            "slope_pct", "v_kmh", "hr_mean"
        ])

    df_activity = df_activity.sort_values("time").reset_index(drop=True)

    times = df_activity["time"].to_numpy()
    elevs = df_activity["elev"].to_numpy()
    dists = df_activity["dist"].to_numpy()
    hrs = df_activity["hr"].to_numpy()

    n = len(df_activity)
    start_idx = 0
    seg_idx = 0
    seg_rows = []

    while start_idx < n - 1:
        t0 = times[start_idx]

        # търсим индекс, при който dt достига T_SEG
        end_idx = start_idx + 1
        while end_idx < n:
            dt_tmp = (times[end_idx] - t0) / np.timedelta64(1, "s")
            if dt_tmp >= T_SEG:
                break
            end_idx += 1

        if end_idx >= n:
            break

        t1 = times[end_idx]
        dt = (t1 - t0) / np.timedelta64(1, "s")

        d0 = dists[start_idx]
        d1 = dists[end_idx]
        elev0 = elevs[start_idx]
        elev1 = elevs[end_idx]
        d_m = max(0.0, d1 - d0)

        if dt < MIN_T_SEG or d_m < MIN_D_SEG:
            start_idx = end_idx
            continue

        if elev0 is None or elev1 is None or np.isnan(elev0) or np.isnan(elev1):
            slope = np.nan
        else:
            slope = (elev1 - elev0) / d_m * 100.0 if d_m > 0 else np.nan

        v_kmh = (d_m / dt) * 3.6
        hr_mean = float(np.nanmean(hrs[start_idx:end_idx + 1]))

        seg_rows.append({
            "activity": activity_label,
            "seg_idx": seg_idx,
            "t_start": pd.to_datetime(t0),
            "t_end": pd.to_datetime(t1),
            "dt_s": float(dt),
            "d_m": float(d_m),
            "slope_pct": float(slope) if not np.isnan(slope) else np.nan,
            "v_kmh": float(v_kmh),
            "hr_mean": hr_mean
        })

        seg_idx += 1
        start_idx = end_idx

    if not seg_rows:
        return pd.DataFrame(columns=[
            "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
            "slope_pct", "v_kmh", "hr_mean"
        ])

    return pd.DataFrame(seg_rows)


# ---------------------------------------------------------
# ФИЛТРИ ЗА НЕРЕАЛИСТИЧНИ СЕГМЕНТИ
# ---------------------------------------------------------
def apply_basic_filters(segments):
    seg = segments.copy()

    # филтър по наклон
    valid_slope = seg["slope_pct"].between(-MAX_ABS_SLOPE, MAX_ABS_SLOPE)
    valid_slope &= seg["slope_pct"].notna()
    seg["valid_basic"] = valid_slope

    # маркиране на скоростни спайкове (по активности)
    def mark_speed_spikes(group):
        group = group.sort_values("seg_idx").copy()
        spike = np.zeros(len(group), dtype=bool)
        v = group["v_kmh"].values
        for i in range(1, len(group)):
            dv = abs(v[i] - v[i-1])
            vmax = max(v[i], v[i-1])
            if dv > V_JUMP_KMH and vmax > V_JUMP_MIN:
                spike[i] = True
        group["speed_spike"] = spike
        return group

    seg = seg.groupby("activity", group_keys=False).apply(mark_speed_spikes)
    seg["speed_spike"] = seg["speed_spike"].fillna(False)
    seg.loc[seg["speed_spike"], "valid_basic"] = False

    return seg


# ---------------------------------------------------------
# МОДЕЛ ЗА НАКЛОН – регресия v(slope) върху целия диапазон
# ---------------------------------------------------------
def get_slope_training_data(seg_f):
    """
    Обучаващи данни за наклоновия модел:
    - всички валидни сегменти със v_kmh > 0
    - наклон в диапазон [-MAX_ABS_SLOPE, +MAX_ABS_SLOPE]
    """
    df = seg_f.copy()
    mask = (
        df["valid_basic"]
        & df["slope_pct"].between(-MAX_ABS_SLOPE, MAX_ABS_SLOPE)
        & (df["v_kmh"] > 0)
        & df["slope_pct"].notna()
    )
    train = df.loc[mask, ["activity", "slope_pct", "v_kmh"]].copy()
    return train


def fit_slope_poly(train_df):
    """
    Фитва полином v_model(s) върху v_kmh ~ slope_pct
    (регресия върху целия диапазон).
    """
    if train_df.empty:
        return None

    x = train_df["slope_pct"].values.astype(float)
    y = train_df["v_kmh"].values.astype(float)

    if len(x) <= SLOPE_POLY_DEG:
        return None

    coeffs = np.polyfit(x, y, SLOPE_POLY_DEG)
    return np.poly1d(coeffs)


def compute_slope_F(slopes, slope_poly, alpha_slope):
    """
    Изчислява корекционния коефициент F(s) за наклона:

      1) v_model(s) = poly(s)
      2) V0 = v_model(0) – типична скорост на равно
      3) F_raw(s) = V0 / v_model(s)
      4) клипваме до [F_MIN_SLOPE, F_MAX_SLOPE]
      5) правила:
          |s| <= 1% → F = 1
          s < -1%  → F <= 1 (спускане)
          s >  1%  → F >= 1 (изкачване)
      6) омекотяване: F = 1 + α * (F - 1)
    """
    slopes = np.asarray(slopes, dtype=float)
    if slope_poly is None:
        return np.ones_like(slopes, dtype=float)

    # 1) v_model(s)
    v_model = slope_poly(slopes)

    # 2) V0 = v_model(0) – скорост на равно според регресията
    V0 = float(slope_poly(0.0))

    # защита от делене на 0/отрицателни модели
    eps = 1e-6
    v_model_safe = np.where(v_model > eps, v_model, eps)

    # 3) суров коефициент
    F_raw = V0 / v_model_safe

    # 4) клип
    F = np.clip(F_raw, F_MIN_SLOPE, F_MAX_SLOPE)

    # 5) правила около 0% наклон
    abs_s = np.abs(slopes)

    # |s| <= 1% → без корекция
    mask_mid = abs_s <= 1.0
    F[mask_mid] = 1.0

    # спускане: s < -1 → F не може да е > 1
    mask_down = slopes < -1.0
    F[mask_down] = np.minimum(F[mask_down], 1.0)

    # изкачване: s > 1 → F не може да е < 1
    mask_up = slopes > 1.0
    F[mask_up] = np.maximum(F[mask_up], 1.0)

    # 6) омекотяване към 1.0
    F = 1.0 + alpha_slope * (F - 1.0)

    return F


def apply_slope_modulation(seg_f, slope_poly, alpha_slope, V_crit=None):
    """
    Прилага корекция по наклон върху v_kmh, за да получим v_flat_eq.
    Няма glide-модел – тук работим директно със сегментната скорост v_kmh.

    Ако V_crit е подаден, може да се добавят допълнителни ограничения за
    много стръмни спускания (по желание).
    """
    df = seg_f.copy()

    if slope_poly is None:
        df["v_flat_eq"] = df["v_kmh"]
        return df

    slopes = df["slope_pct"].values.astype(float)
    F_vals = compute_slope_F(slopes, slope_poly, alpha_slope)

    v_flat_eq = df["v_kmh"].values * F_vals

    # по желание: ограничение за много силни спускания
    if V_crit is not None and V_crit > 0:
        idx_below = df["slope_pct"] < -5.0
        v_flat_eq[idx_below] = np.minimum(v_flat_eq[idx_below], 0.8 * V_crit)

    df["v_flat_eq"] = v_flat_eq
    return df


# ---------------------------------------------------------
# ЗОНИ ПО СКОРОСТ И ПУЛС
# ---------------------------------------------------------
def assign_speed_zones(seg_slope, V_crit):
    df = seg_slope.copy()
    if V_crit is None or V_crit <= 0:
        df["rel_crit"] = np.nan
        df["zone"] = None
        return df

    df["rel_crit"] = df["v_flat_eq"] / V_crit

    zones = []
    for r in df["rel_crit"]:
        if pd.isna(r):
            zones.append(None)
            continue
        z_name = None
        for i in range(len(ZONE_NAMES)):
            if ZONE_BOUNDS[i] <= r < ZONE_BOUNDS[i + 1]:
                z_name = ZONE_NAMES[i]
                break
        zones.append(z_name)
    df["zone"] = zones
    return df


def summarize_speed_zones(seg_zones):
    df = seg_zones.dropna(subset=["zone"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["zone", "n_segments", "total_time_s", "mean_v_flat_eq"])
    agg = df.groupby("zone").agg(
        n_segments=("seg_idx", "count"),
        total_time_s=("dt_s", "sum"),
        mean_v_flat_eq=("v_flat_eq", "mean"),
    ).reset_index()
    agg = agg.sort_values("zone")
    return agg


def compute_zone_hr_from_counts(seg_df, zone_counts):
    """
    seg_df: DataFrame със сегментите
    zone_counts: dict {zone: n_segments}, на база скоростните зони.

    Алгоритъм:
      - филтрираме само:
          * сегменти без speed_spike
          * сегменти с наличен hr_mean
      - сортираме тези сегменти по hr_mean ↑
      - за Z1 взимаме първите N1, за Z2 – следващите N2, ...
    """
    df_hr = seg_df.copy()
    if "speed_spike" in df_hr.columns:
        df_hr = df_hr[~df_hr["speed_spike"].fillna(False)]

    df_hr = df_hr.dropna(subset=["hr_mean"]).copy()

    if df_hr.empty:
        rows = [{"zone": z, "mean_hr_zone": np.nan} for z in ZONE_NAMES]
        return pd.DataFrame(rows)

    df_hr = df_hr.sort_values("hr_mean").reset_index(drop=True)

    results = []
    start_idx = 0
    for z in ZONE_NAMES:
        n = int(zone_counts.get(z, 0))
        if n <= 0 or start_idx >= len(df_hr):
            results.append({"zone": z, "mean_hr_zone": np.nan})
            continue
        end_idx = min(start_idx + n, len(df_hr))
        subset = df_hr.iloc[start_idx:end_idx]
        mean_hr = subset["hr_mean"].mean() if not subset.empty else np.nan
        results.append({"zone": z, "mean_hr_zone": mean_hr})
        start_idx = end_idx

    return pd.DataFrame(results)


def build_zone_speed_hr_table(seg_zones, V_crit, activity=None):
    """
    Връща таблица по зони:
      Зона | Брой сегменти | Време [ч:мм:сс] | Средна скорост | Среден пулс
    Ако activity е None -> всички активности.
    """
    if activity is not None:
        df = seg_zones[seg_zones["activity"] == activity].copy()
    else:
        df = seg_zones.copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "Зона", "Брой сегменти", "Време [ч:мм:сс]",
            "Средна скорост [km/h]", "Среден пулс [bpm]"
        ])

    speed_summary = summarize_speed_zones(df)
    if speed_summary.empty:
        return pd.DataFrame(columns=[
            "Зона", "Брой сегменти", "Време [ч:мм:сс]",
            "Средна скорост [km/h]", "Среден пулс [bpm]"
        ])

    zone_counts = dict(zip(speed_summary["zone"], speed_summary["n_segments"]))
    hr_summary = compute_zone_hr_from_counts(df, zone_counts)

    merged = pd.merge(speed_summary, hr_summary, on="zone", how="left")

    merged["time_hhmmss"] = merged["total_time_s"].apply(seconds_to_hhmmss)

    merged = merged.rename(columns={
        "zone": "Зона",
        "n_segments": "Брой сегменти",
        "time_hhmmss": "Време [ч:мм:сс]",
        "mean_v_flat_eq": "Средна скорост [km/h]",
        "mean_hr_zone": "Среден пулс [bpm]",
    })

    merged = merged[[
        "Зона", "Брой сегменти", "Време [ч:мм:сс]",
        "Средна скорост [km/h]", "Среден пулс [bpm]"
    ]]

    return merged


# ---------------------------------------------------------
# STREAMLIT APP – Бягане и ходене (TCX)
# ---------------------------------------------------------
st.set_page_config(page_title="Run/Walk Slope Normalization", layout="wide")
st.title("Модел за приравняване на скоростта при бягане и ходене (наклон)")

st.markdown(
    """
Този app:
1. Чете TCX файлове от бягане или ходене  
2. Сегментира ги на 30-секундни сегменти  
3. Обучава регресионен модел v(slope) върху целия диапазон наклони  
4. Нормализира скоростта към „еквивалентна на равно“ v_flat_eq  
5. Изчислява зони по скорост и пулс  
"""
)

# ---------- Sidebar: параметри ----------
st.sidebar.header("Параметри на модела")

V_crit = st.sidebar.number_input(
    "Критична скорост V_crit [km/h]",
    min_value=4.0,
    max_value=30.0,
    value=12.0,
    step=0.5,
    help=(
        "Референтна „критична“ скорост, спрямо която се определят "
        "скоростните зони (Z1–Z6) след нормализация по наклон."
    ),
)

ALPHA_SLOPE = st.sidebar.slider(
    "Омекотяване на наклона β",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help=(
        "Коефициент за омекотяване на корекцията по наклон.\n"
        "β = 1 → пълна корекция според регресията.\n"
        "β = 0 → без корекция по наклон (v_flat_eq = v_kmh)."
    ),
)

st.sidebar.markdown("---")
st.sidebar.info(
    "За по-чиста регресия е добре да качваш отделно бегови и ходещи "
    "тренировки (да не смесваш много различни скорости в един модел)."
)

# ---------- File uploader ----------
uploaded_files = st.file_uploader(
    "Качи един или няколко TCX файла (бягане / ходене):",
    type=["tcx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Качи поне един TCX файл, за да започнем.")
    st.stop()

# 1) Парсване на файловете
all_points = []
for f in uploaded_files:
    label = f.name
    df_act = parse_tcx(f, label)
    if df_act.empty:
        continue
    all_points.append(df_act)

if not all_points:
    st.error("Не успях да извлека данни от файловете.")
    st.stop()

points = pd.concat(all_points, ignore_index=True)

# 2) Сегментиране на 30 s
seg_list = []
for act, g in points.groupby("activity"):
    seg_df = build_segments_30s(g, act)
    if not seg_df.empty:
        seg_list.append(seg_df)

segments = pd.concat(seg_list, ignore_index=True) if seg_list else pd.DataFrame()
if segments.empty:
    st.error("Не успях да създам сегменти. Провери TCX файловете.")
    st.stop()

# 3) Базови филтри
segments_f = apply_basic_filters(segments)

# 4) Обучение на наклоновия модел
slope_train = get_slope_training_data(segments_f)
slope_poly = fit_slope_poly(slope_train)

if slope_poly is None:
    st.warning("Няма достатъчно данни за надежден наклонов модел. "
               "Ще използваме суровата сегментна скорост (без корекция).")

# 5) Прилагане на наклоновия модел
seg_slope = apply_slope_modulation(segments_f, slope_poly, ALPHA_SLOPE, V_crit=V_crit)

# 6) Зони по скорост
seg_zones = assign_speed_zones(seg_slope, V_crit)

# ---------------------------------------------------------
# Обобщена таблица по активности
# ---------------------------------------------------------
st.subheader("Обобщение по активности")

summary = (
    seg_slope[seg_slope["valid_basic"]]
    .groupby("activity")
    .agg(
        v_real_mean=("v_kmh", "mean"),
        v_flat_mean=("v_flat_eq", "mean"),
        n_segments=("seg_idx", "count"),
        time_total_s=("dt_s", "sum"),
    )
    .reset_index()
)

summary["time_total_hhmmss"] = summary["time_total_s"].apply(seconds_to_hhmmss)

summary = summary.rename(columns={
    "activity": "Активност",
    "v_real_mean": "Ср. реална скорост [km/h]",
    "v_flat_mean": "Ср. еквивалентна скорост на равно [km/h]",
    "n_segments": "Брой сегменти",
    "time_total_hhmmss": "Общо време [ч:мм:сс]",
})

summary = summary[[
    "Активност",
    "Ср. реална скорост [km/h]",
    "Ср. еквивалентна скорост на равно [km/h]",
    "Брой сегменти",
    "Общо време [ч:мм:сс]",
]]

st.dataframe(summary, use_container_width=True)

# ---------------------------------------------------------
# ГРАФИКА 1 – Наклонов модел v(s)
# ---------------------------------------------------------
st.subheader("Зависимост скорост–наклон (регресионен модел)")

if not slope_train.empty and slope_poly is not None:
    s_min = slope_train["slope_pct"].min()
    s_max = slope_train["slope_pct"].max()
    s_grid = np.linspace(s_min, s_max, 200)
    df_curve = pd.DataFrame({
        "slope_pct": s_grid,
        "v_model": slope_poly(s_grid),
    })

    chart_points = alt.Chart(slope_train).mark_circle(size=30).encode(
        x=alt.X("slope_pct", title="Наклон [%]"),
        y=alt.Y("v_kmh", title="Скорост [km/h]"),
        color="activity:N"
    )

    chart_curve = alt.Chart(df_curve).mark_line().encode(
        x="slope_pct",
        y="v_model"
    )

    st.altair_chart(chart_points + chart_curve, use_container_width=True)

    st.markdown(
        f"**Модел за наклон:**  v_model(s) = {poly_to_str(slope_poly, var='s')}  \n"
        "Нормализацията използва V(0)/v_model(s) с клип и омекотяване β."
    )
else:
    st.info("Няма достатъчно данни за визуализация на наклоновия модел.")

# ---------------------------------------------------------
# ГРАФИКА 2 – Времева серия v_kmh vs v_flat_eq
# ---------------------------------------------------------
st.subheader("Времеви профил – реална vs еквивалентна скорост")

act_list = sorted(seg_slope["activity"].unique())
act_selected = st.selectbox(
    "Избери активност:",
    act_list,
    key="time_series_act"
)

g_plot = seg_slope[seg_slope["activity"] == act_selected].copy()
if not g_plot.empty:
    g_plot = g_plot.sort_values("t_start")
    g_plot["time_s"] = (g_plot["t_start"] - g_plot["t_start"].min()) / np.timedelta64(1, "s")

    base = alt.Chart(g_plot).encode(
        x=alt.X("time_s:Q", title="Време [s]")
    )

    line_real = base.mark_line().encode(
        y=alt.Y("v_kmh:Q", title="Скорост [km/h]"),
        color=alt.value("#1f77b4")
    )

    line_flat = base.mark_line(strokeDash=[4, 4]).encode(
        y="v_flat_eq:Q",
        color=alt.value("#ff7f0e")
    )

    st.altair_chart(line_real + line_flat, use_container_width=True)

    st.caption("Плътна линия – реална сегментна скорост; "
               "пунктирана – еквивалентна скорост на равно след нормализация по наклон.")

# ---------------------------------------------------------
# Зони – всички активности
# ---------------------------------------------------------
st.subheader("Разпределение по зони – скорост и пулс (всички активности)")

zone_table_all = build_zone_speed_hr_table(seg_zones, V_crit, activity=None)
st.dataframe(zone_table_all, use_container_width=True)

# ---------------------------------------------------------
# Зони – избрана активност
# ---------------------------------------------------------
st.subheader("Разпределение по зони – скорост и пулс (избрана активност)")

act_selected_z = st.selectbox(
    "Избери активност за зонен анализ:",
    act_list,
    key="zone_act_select"
)

zone_table_act = build_zone_speed_hr_table(seg_zones, V_crit, activity=act_selected_z)
st.dataframe(zone_table_act, use_container_width=True)

# ---------------------------------------------------------
# Експорт на сегментите
# ---------------------------------------------------------
st.subheader("Експорт на сегментите (след нормализация по наклон)")

export_cols = [
    "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
    "slope_pct", "v_kmh", "valid_basic", "speed_spike",
    "v_flat_eq", "hr_mean"
]

available_export_cols = [c for c in export_cols if c in seg_slope.columns]
export_df = seg_slope[available_export_cols].copy()

csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Свали сегментите като CSV",
    data=csv_data,
    file_name="segments_run_walk_slope_normalized.csv",
    mime="text/csv"
)
