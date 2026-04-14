import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_lidur8():
    #Lesa gögn
    rennsli_file = "lamah_ice/D_gauges/2_timeseries/daily/ID_12.csv"
    vedur_file = "lamah_ice/A_basins_total_upstrm/2_timeseries/daily/meteorological_data/ID_12.csv"

    df_q = pd.read_csv(rennsli_file, sep=";")
    df_w = pd.read_csv(vedur_file, sep=";")

    #Búa til dagsetningar
    df_q["date"] = pd.to_datetime(dict(year=df_q["YYYY"], month=df_q["MM"], day=df_q["DD"]))
    df_w["date"] = pd.to_datetime(dict(year=df_w["YYYY"], month=df_w["MM"], day=df_w["DD"]))

    #Velja dálka
    #Rennsli
    q_col = "qobs"
    #Veðurgögn
    p_col = "prec"
    t_col = "2m_temp_mean"

    df_q = df_q[["date", q_col]].copy()
    df_q = df_q.rename(columns={q_col: "Q"})

    df_w = df_w[["date", p_col, t_col]].copy()
    df_w = df_w.rename(columns={p_col: "P", t_col: "T"})

    #Sameina gögn

    data = pd.merge(df_q, df_w, on="date", how="inner")
    data = data.sort_values("date").reset_index(drop=True)

    # Breyta í numeric ef eitthvað kemur inn sem texti
    for col in ["Q", "P", "T"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Henda út röðum þar sem Q vantar
    data = data.dropna(subset=["Q"]).reset_index(drop=True)

    #Finna staðbundna toppa í rennsli
    data["is_peak"] = (
    (data["Q"] > data["Q"].shift(1)) &
    (data["Q"] > data["Q"].shift(-1))
    )

    peaks = data[data["is_peak"]].copy()

    #5 hæstu toppar
    top5 = peaks.nlargest(5, "Q")[["date", "Q"]].reset_index(drop=True)
    print("5 hæstu rennslistoppar:")
    print(top5)

    #Velja einn atburð
    #Hæsti toppur valinn
    peak_date = top5.loc[0, "date"]
    peak_idx = data.index[data["date"] == peak_date][0]
    peak_Q = data.loc[peak_idx, "Q"]

    print("\nValinn atburður:")
    print(f"Toppdagur: {peak_date.date()} | Qmax = {peak_Q:.3f}")

    #Velja tímabil í kringum topp
    days_before = 10
    days_after = 10

    start_window = peak_date - pd.Timedelta(days=days_before)
    end_window = peak_date + pd.Timedelta(days=days_after)

    event = data[(data["date"] >= start_window) & (data["date"] <= end_window)].copy()
    event = event.reset_index(drop=True)

    #Grunnrennsli fyrir atburð
    #Notum lægsta Q fyrstu 5 dagana í glugganum sem nálgun á grunnrennsli
    n_base = min(5, len(event))
    baseline_Q = event.loc[:n_base-1, "Q"].min()

    #Finna upphaf rennslisaukningar
    #Skilgreinum upphaf sem fyrsta skipti sem Q fer yfir:
    #baseline + 10% af (Qmax - baseline)
    threshold_start = baseline_Q + 0.10 * (peak_Q - baseline_Q)

    start_candidates = event.index[event["Q"] > threshold_start].tolist()

    if len(start_candidates) > 0:
        start_idx = start_candidates[0]
    else:
        start_idx = 0

    start_date = event.loc[start_idx, "date"]
    start_Q = event.loc[start_idx, "Q"]

    time_to_peak_days = (peak_date - start_date).days

    #Finna lok úrkomu
    #Lok úrkomu = síðasti dagur í atburðinum þar sem P > rain_threshold
    rain_threshold = 1.0

    rain_days = event.index[event["P"].fillna(0) > rain_threshold].tolist()

    if len(rain_days) > 0:
        rain_end_idx = rain_days[-1]
        rain_end_date = event.loc[rain_end_idx, "date"]
    else:
        rain_end_idx = None
        rain_end_date = None

    #Finna hvenær rennsli er aftur komið nálægt grunnástandi
    #Skilgreinum það sem: Q <= baseline + 5% af (Qmax - baseline)
    return_threshold = baseline_Q + 0.05 * (peak_Q - baseline_Q)

    if rain_end_idx is not None:
        after_rain = event.loc[rain_end_idx + 1:].copy()
    else:
        after_rain = event.copy()

    return_candidates = after_rain.index[after_rain["Q"] <= return_threshold].tolist()

    if len(return_candidates) > 0:
        return_idx = return_candidates[0]
        return_date = event.loc[return_idx, "date"]
        excess_rain_release_time_days = (return_date - rain_end_date).days
    else:
        return_idx = None
        return_date = None
        excess_rain_release_time_days = np.nan

    #Prenta niðurstöður
    print("\nNiðurstöður:")
    print(f"Grunnrennsli (baseline Q): {baseline_Q:.3f}")
    print(f"Upphaf rennslisaukningar: {start_date.date()}")
    print(f"Hámarksrennsli Qmax: {peak_date.date()} ({peak_Q:.3f})")
    print(f"Time-to-peak: {time_to_peak_days} dagar")

    if rain_end_date is not None:
        print(f"Lok úrkomu: {rain_end_date.date()}")
    else:
        print("Enginn úrkomudagur yfir þröskuldi fannst í glugganum.")

    if return_date is not None:
        print(f"Rennsli aftur nálægt grunnástandi: {return_date.date()}")
        print(f"Excess rain release time: {int(excess_rain_release_time_days)} dagar")
    else:
        print("Rennsli náði ekki aftur grunnástandi innan tímabilsins.")

    #Plotta Q, P og T

    fig, axes = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    #Q
    axes[0].plot(event["date"], event["Q"], linewidth=2, label="Q")
    axes[0].axvline(start_date, linestyle="--", label="Upphaf rennslisaukningar")
    axes[0].axvline(peak_date, linestyle="--", label="Qmax")

    if rain_end_date is not None:
        axes[0].axvline(rain_end_date, linestyle="--", label="Lok úrkomu")

    if return_date is not None:
        axes[0].axvline(return_date, linestyle="--", label="Aftur í grunnástandi")

    axes[0].axhline(baseline_Q, linestyle=":", label="Grunnrennsli")
    axes[0].set_ylabel("Q")
    axes[0].set_title("Rennslisatburður: Q, P og T")
    axes[0].legend(loc="best")

    text_lines = [f"Time-to-peak = {time_to_peak_days} dagar"]
    if not np.isnan(excess_rain_release_time_days):
        text_lines.append(f"Excess rain release time = {int(excess_rain_release_time_days)} dagar")

    axes[0].text(
    0.01, 0.95,
    "\n".join(text_lines),
    transform=axes[0].transAxes,
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    #P
    axes[1].bar(event["date"], event["P"].fillna(0), width=0.8, label="Úrkoma P")
    axes[1].set_ylabel("P")
    axes[1].legend(loc="best")

    #T
    axes[2].plot(event["date"], event["T"], linewidth=2, label="Hitastig T")
    axes[2].axhline(0, linestyle=":")
    axes[2].set_ylabel("T (°C)")
    axes[2].set_xlabel("Dagsetning")
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.savefig("figures/rennslisatburðir.png")
