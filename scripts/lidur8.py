import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    q_col = "qobs"
    p_col = "prec"
    t_col = "2m_temp_mean"

    df_q = df_q[["date", q_col]].copy()
    df_q = df_q.rename(columns={q_col: "Q"})

    df_w = df_w[["date", p_col, t_col]].copy()
    df_w = df_w.rename(columns={p_col: "P", t_col: "T"})

    #Sameina gögn
    data = pd.merge(df_q, df_w, on="date", how="inner")
    data = data.sort_values("date").reset_index(drop=True)

    for col in ["Q", "P", "T"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Q"]).reset_index(drop=True)

    #Finna staðbundna toppa í rennsli
    data["is_peak"] = (
    (data["Q"] > data["Q"].shift(1)) &
    (data["Q"] > data["Q"].shift(-1))
    )

    peaks = data[data["is_peak"]].copy()
    top5 = peaks.nlargest(5, "Q")[["date", "Q"]].reset_index(drop=True)

    print("5 hæstu rennslistoppar:")
    print(top5)

    #Velja einn atburð
    peak_date = top5.loc[0, "date"]
    peak_idx  = data.index[data["date"] == peak_date][0]
    peak_Q    = data.loc[peak_idx, "Q"]

    print("\nValinn atburður:")
    print(f"Toppdagur: {peak_date.date()} | Qmax = {peak_Q:.3f}")

    #Grunnrennsli
    days_before = 10
    pre_window = data[
    (data["date"] >= peak_date - pd.Timedelta(days=days_before)) &
    (data["date"] <= peak_date)
    ].copy()

    n_base     = min(5, len(pre_window))
    baseline_Q = pre_window.iloc[:n_base]["Q"].min()
    return_threshold = baseline_Q + 0.05 * (peak_Q - baseline_Q)

    for days_after in range(10, 121, 5):
        end_window  = peak_date + pd.Timedelta(days=days_after)
        after_check = data[(data["date"] > peak_date) & (data["date"] <= end_window)]
        if (after_check["Q"] <= return_threshold).any():
            break

    start_window = peak_date - pd.Timedelta(days=days_before)
    end_window   = peak_date + pd.Timedelta(days=days_after)

    event = data[(data["date"] >= start_window) & (data["date"] <= end_window)].copy()
    event = event.reset_index(drop=True)

    peak_event_idx = event.index[event["date"] == peak_date][0]

    print(f"Tímabilsglugginn: {start_window.date()} → {end_window.date()} ({days_after} dagar eftir topp)")

    #Finna upphaf rennslisaukningar
    threshold_start  = baseline_Q + 0.10 * (peak_Q - baseline_Q)
    start_candidates = event.index[event["Q"] > threshold_start].tolist()
    start_idx  = start_candidates[0] if start_candidates else 0
    start_date = event.loc[start_idx, "date"]
    time_to_peak_days = (peak_date - start_date).days

    #Finna lok úrkomu
    rain_threshold = 1.0
    rain_search = event[
    (event["date"] >= peak_date - pd.Timedelta(days=5)) &
    (event["date"] <= peak_date + pd.Timedelta(days=10))
    ]
    rain_days = rain_search.index[rain_search["P"].fillna(0) > rain_threshold].tolist()

    if len(rain_days) > 0:
        rain_end_idx  = rain_days[-1]
        rain_end_date = event.loc[rain_end_idx, "date"]
    else:
        rain_end_idx  = None
        rain_end_date = None

    #Finna hvenær rennsli er aftur komið nálægt grunnástandi
    after_peak        = event.loc[peak_event_idx + 1:].copy()
    return_candidates = after_peak.index[after_peak["Q"] <= return_threshold].tolist()

    if len(return_candidates) > 0:
        return_idx  = return_candidates[0]
        return_date = event.loc[return_idx, "date"]
    else:
        return_idx  = None
        return_date = None

    #Reikna tíma
    if return_date is not None and rain_end_date is not None:
        excess_rain_release_time_days = (return_date - rain_end_date).days
    else:
        excess_rain_release_time_days = np.nan

    recession_time_days = (return_date - peak_date).days if return_date is not None else np.nan

    #Prenta niðurstöður
    print("\nNiðurstöður:")
    print(f"Grunnrennsli (baseline Q):         {baseline_Q:.3f}")
    print(f"Upphaf rennslisaukningar:           {start_date.date()}")
    print(f"Hámarksrennsli Qmax:                {peak_date.date()} ({peak_Q:.3f})")
    print(f"Time-to-peak:                       {time_to_peak_days} dagar")
    if rain_end_date is not None:
        print(f"Lok úrkomu:                         {rain_end_date.date()}")
    if return_date is not None:
        print(f"Rennsli aftur nálægt grunnástandi:  {return_date.date()}")
        print(f"Excess rain release time:           {int(excess_rain_release_time_days)} dagar")
        print(f"Recession time (Qmax → base):       {int(recession_time_days)} dagar")

    #Plotta Q, P og T
    fig, axes = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    ax = axes[0]

    #Q
    ax.plot(event["date"], event["Q"], linewidth=2, color="steelblue", label="Q", zorder=3)
    ax.axhline(baseline_Q, color="gray", linestyle=":", linewidth=1.2, label="Grunnrennsli", zorder=2)
    ax.axvline(start_date,  color="green",  linestyle="--", linewidth=1.4, label="Upphaf rennslisaukningar")
    ax.axvline(peak_date,   color="red",    linestyle="--", linewidth=1.4, label="Qmax")
    if rain_end_date is not None:
        ax.axvline(rain_end_date, color="orange", linestyle="--", linewidth=1.4, label="Lok úrkomu")
    if return_date is not None:
        ax.axvline(return_date, color="purple", linestyle="--", linewidth=1.4, label="Aftur í grunnástandi")
    y_max   = event["Q"].max()
    y_range = y_max - baseline_Q
    level1 = y_max + y_range * 0.08   # time-to-peak      (top)
    level2 = y_max + y_range * 0.22   # recession time    (middle)
    level3 = y_max + y_range * 0.36   # excess rain rel.  (bottom — only if available)

    arrow_kw = dict(arrowstyle="<->", color="black", lw=1.2)

    def draw_span(ax, x_start, x_end, y_level, label, color):
        """Draw a horizontal double-arrow with a centred label."""
        ax.annotate(
            "", xy=(x_end, y_level), xytext=(x_start, y_level),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.4)
        )
        mid = x_start + (x_end - x_start) / 2
        ax.text(
            mid, y_level, label,
            ha="center", va="bottom", fontsize=8.5, color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1)
        )

    draw_span(ax, start_date, peak_date, level1,
            f"Time-to-peak = {time_to_peak_days} d", "green")

    #Recession time
    if return_date is not None:
        draw_span(ax, peak_date, return_date, level2,
                f"Recession time = {int(recession_time_days)} d", "purple")
        
    if rain_end_date is not None and return_date is not None and not np.isnan(excess_rain_release_time_days):
        draw_span(ax, rain_end_date, return_date, level3,
                f"Excess rain release = {int(excess_rain_release_time_days)} d", "orange")
    ax.set_ylim(bottom=0, top=y_max + y_range * 0.55)
    ax.set_ylabel("Q (m³/s)")
    ax.set_title("Rennslisatburður: Q, P og T")
    ax.legend(loc="upper right", fontsize=8)
    #P
    axes[1].bar(event["date"], event["P"].fillna(0), width=0.8, color="cornflowerblue", label="Úrkoma P")
    axes[1].set_ylabel("P (mm)")
    axes[1].legend(loc="upper right", fontsize=8)
    #T
    axes[2].plot(event["date"], event["T"], linewidth=2, color="tomato", label="Hitastig T")
    axes[2].axhline(0, color="gray", linestyle=":")
    axes[2].set_ylabel("T (°C)")
    axes[2].set_xlabel("Dagsetning")
    axes[2].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/flood_event.png", dpi=150)
