import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

def run_baseflow():
    skra = "lamah_ice/D_gauges/2_timeseries/daily/ID_12.csv"

    df = pd.read_csv(skra, sep=";")

    df.columns = [c.strip() for c in df.columns]

    df["date"] = pd.to_datetime(
        dict(year=df["YYYY"], month=df["MM"], day=df["DD"]),
        errors="coerce")

    df["Q"] = pd.to_numeric(df["qobs"], errors="coerce")

    df = df.dropna(subset=["date", "Q"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    def lyne_hollick_forward(Q, alpha=0.925):
        """
        Einn áfram-pass af Lyne-Hollick síu.
        Skilar:
        q  = quickflow
        qb = baseflow
        """
        Q = np.asarray(Q, dtype=float)
        n = len(Q)

        q = np.zeros(n)
        qb = np.zeros(n)
   
        q[0] = 0.0
        qb[0] = Q[0]

        for t in range(1, n):
            q[t] = alpha * q[t-1] + ((1 + alpha) / 2.0) * (Q[t] - Q[t-1])

            if q[t] < 0:
                q[t] = 0.0
            if q[t] > Q[t]:
                q[t] = Q[t]

            qb[t] = Q[t] - q[t]

            if qb[t] < 0:
                qb[t] = 0.0
            if qb[t] > Q[t]:
                qb[t] = Q[t]

        return q, qb

    def ladson_three_pass(Q, alpha=0.925):
        """
        3-pass síun samkvæmt algengri framkvæmd Ladson et al.:
        1) áfram
        2) afturábak
        3) áfram
        """
        Q = np.asarray(Q, dtype=float)

        _, qb1 = lyne_hollick_forward(Q, alpha=alpha)

        _, qb2_rev = lyne_hollick_forward(qb1[::-1], alpha=alpha)
        qb2 = qb2_rev[::-1]

        _, qb3 = lyne_hollick_forward(qb2, alpha=alpha)

        qb3 = np.clip(qb3, 0, Q)
        qf3 = Q - qb3

        return qf3, qb3
    def compute_bfi(Q, Qb):
        """
        Baseflow Index:
        BFI = sum(Qb) / sum(Q)
        """
        Q = np.asarray(Q, dtype=float)
        Qb = np.asarray(Qb, dtype=float)

        valid = np.isfinite(Q) & np.isfinite(Qb) & (Q >= 0)
        return np.sum(Qb[valid]) / np.sum(Q[valid])

    alpha = 0.925
    Q = df["Q"].values
    quickflow, baseflow = ladson_three_pass(Q, alpha=alpha)

    df["Qb"] = baseflow
    df["Qq"] = quickflow
    BFI = compute_bfi(df["Q"], df["Qb"])

    print(f"Fjöldi punkta: {len(df)}")
    print(f"BFI: {BFI:.4f}")

    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["Q"], label="Heildarrennsli Q", linewidth=1.0)
    plt.plot(df["date"], df["Qb"], label="Baseflow Qb", linewidth=2.0)
    plt.xlabel("Dagsetning")
    plt.ylabel("Rennsli (m³/s)")
    plt.title("Baseflow separation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/baseflow.png", dpi=300)

def run_recession():

    q_df = pd.read_csv("data/rennslisgogn.csv", sep=";")
    w_df = pd.read_csv("data/vedurgogn.csv", sep=";")

    q_df["date"] = pd.to_datetime(dict(year=q_df["YYYY"], month=q_df["MM"], day=q_df["DD"]))
    w_df["date"] = pd.to_datetime(dict(year=w_df["YYYY"], month=w_df["MM"], day=w_df["DD"]))

    q_df["qobs"] = pd.to_numeric(q_df["qobs"], errors="coerce")

    w_df["prec_rav"] = pd.to_numeric(w_df["prec_rav"], errors="coerce")

    df = q_df[["date", "qobs"]].merge(
        w_df[["date", "prec_rav"]],
        on="date",
        how="left"
    )

    df = df.rename(columns={"prec_rav": "prec"})
    df = df.sort_values("date").reset_index(drop=True)

    df = df.dropna(subset=["qobs"])
    df = df[df["qobs"] > 0].copy()
    df["prec"] = df["prec"].fillna(0)

    df["dq"] = df["qobs"].diff()
    df["dry"] = df["prec"] < 1.0
    df["recede"] = df["dq"] < 0
    df["candidate"] = df["dry"] & df["recede"]

    group_id = (df["candidate"] != df["candidate"].shift()).cumsum()

    segments = []
    for _, seg in df.groupby(group_id):
        if seg["candidate"].iloc[0] and len(seg) >= 5:
            segments.append(seg.copy())

    print(f"Fjöldi recession-skeiða: {len(segments)}")

    results = []

    for i, seg in enumerate(segments, start=1):
        x = np.arange(len(seg))
        y = np.log(seg["qobs"].values)

        slope, intercept, lo_slope, hi_slope = theilslopes(y, x, 0.95)

        k = -slope
        K_daily = np.exp(-k)

        results.append({
            "segment": i,
            "start": seg["date"].iloc[0],
            "end": seg["date"].iloc[-1],
            "n_days": len(seg),
            "slope_lnQ_day": slope,
            "k_day^-1": k,
            "K_daily": K_daily,
            "k_low_95": -hi_slope,
            "k_high_95": -lo_slope
        })

    res_df = pd.DataFrame(results)

    print("\nNiðurstöður fyrir hvert skeið:")
    print(res_df[["segment", "start", "end", "n_days", "k_day^-1", "K_daily"]])

    k_median = res_df["k_day^-1"].median()
    k_mean = res_df["k_day^-1"].mean()
    K_median = np.exp(-k_median)

    print("\nHeildarmat:")
    print(f"Miðgildi recession constant, k = {k_median:.4f} 1/dag")
    print(f"Meðaltal recession constant, k = {k_mean:.4f} 1/dag")
    print(f"Samsvarandi daglegur recession factor, K = exp(-k) = {K_median:.4f}")

    res_df["distance_from_median"] = (res_df["k_day^-1"] - k_median).abs()
    best_idx = res_df["distance_from_median"].idxmin()
    best_seg_no = int(res_df.loc[best_idx, "segment"])
    best_seg = segments[best_seg_no - 1]

    x = np.arange(len(best_seg))
    y = np.log(best_seg["qobs"].values)
    slope, intercept, _, _ = theilslopes(y, x, 0.95)
    q_fit = np.exp(intercept + slope * x)

    plt.figure(figsize=(8, 5))
    plt.plot(best_seg["date"], best_seg["qobs"], marker="o", label="Mælt rennsli")
    plt.plot(best_seg["date"], q_fit, linestyle="--", label="Fitted")
    plt.xlabel("Dagsetning")
    plt.ylabel("Rennsli Q (m³/s)")
    plt.title(f"Recession-skeið {best_seg_no}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/recession_skeid.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="ln(Q) mælt")
    plt.plot(x, intercept + slope * x, label=f"Fit: slope = {slope:.4f}")
    plt.xlabel("Tími frá upphafi skeiðs (dagar)")
    plt.ylabel("ln(Q)")
    plt.title(f"ln(Q) fyrir recession-skeið {best_seg_no}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/recession_ln.png")

    k_rep = -slope
    tau_rep = -1 / slope   

    print("\nDæmigert skeið:")
    print(f"Segment nr.: {best_seg_no}")
    print(f"Tímabil: {best_seg['date'].iloc[0].date()} til {best_seg['date'].iloc[-1].date()}")
    print(f"Hallatala = {slope:.4f}")
    print(f"Recession constant, k = {k_rep:.4f} 1/dag")
    print(f"Samsvarandi timescale = {-1/slope:.2f} dagar")

def run_lidur3():
    run_baseflow()
    run_recession()

