import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_lidur4():
    q = pd.read_csv("data/rennslisgogn.csv", sep=";")
    w = pd.read_csv("data/vedurgogn.csv", sep=";")
    v = pd.read_csv("data/vatnasvid.csv", sep=";")

    q["date"] = pd.to_datetime(dict(year=q["YYYY"], month=q["MM"], day=q["DD"]))
    w["date"] = pd.to_datetime(dict(year=w["YYYY"], month=w["MM"], day=w["DD"]))

    basin = v.loc[v["id"] == 12].iloc[0]

    def scale_metadata(value, divisor_if_large, threshold):
        if pd.isna(value):
            return np.nan
        return value / divisor_if_large if abs(value) > threshold else value

    area_km2 = scale_metadata(basin["area_calc"], divisor_if_large=100, threshold=10000)
    elev_mean_m = scale_metadata(basin["elev_mean"], divisor_if_large=1000, threshold=10000)
    p_mean_mm_yr = scale_metadata(basin["p_mean"], divisor_if_large=10, threshold=1000)
    slope_mean = scale_metadata(basin["slope_mean"], divisor_if_large=10000, threshold=1000)

    print("Vatnasvið 12")
    print(f"Flatarmál ~ {area_km2:.2f} km²")
    print(f"Meðalhæð ~ {elev_mean_m:.1f} m")
    print(f"Meðalúrkoma ~ {p_mean_mm_yr:.1f} mm/ár")
    print(f"Meðalhalli ~ {slope_mean:.2f}")
    print()

    carra_cols = [
        "date",
        "prec_carra",
        "total_et_carra",
        "swe_carra",
        "solid_prec_carra",
        "runoff_carra",
        "percolation_carra",
        "2m_temp_carra",
        "2m_temp_min_carra",
        "2m_temp_max_carra",
    ]

    df = q.merge(w[carra_cols], on="date", how="inner")

    df = df.dropna(subset=["qobs", "prec_carra", "total_et_carra", "swe_carra"]).copy()

    print("Sameiginlegt tímabil qobs + CARRA:")
    print("Frá", df["date"].min(), "til", df["date"].max())
    print("Fjöldi daga:", len(df))
    print()

    def scale_carra(series, threshold=100, divisor=1000):
        """
        Ef gildi eru mjög stór er líklegt að þau séu í röngum kvarða
        og þurfi að deilast með 1000.
        """
        s = series.copy()
        mask = s.abs() > threshold
        s.loc[mask] = s.loc[mask] / divisor
        return s

    df["P"] = scale_carra(df["prec_carra"], threshold=100, divisor=1000)

    df["ET"] = -scale_carra(df["total_et_carra"], threshold=100, divisor=1000)

    df["SWE"] = scale_carra(df["swe_carra"], threshold=1000, divisor=1000)

    df["dS_snow"] = df["SWE"].diff()

    df["Q_m3s"] = df["qobs"] / 1000

    area_m2 = area_km2 * 1e6
    df["Q_mm_day"] = df["Q_m3s"] * 86400 / area_m2 * 1000

    monthly = (
        df.set_index("date")
        .resample("MS")
        .agg({
            "P": "sum",              
            "ET": "sum",            
            "dS_snow": "sum",        
            "SWE": "mean",          
            "qobs": "mean",          
            "Q_mm_day": "sum"        
        })
        .reset_index()
    )

    monthly = monthly.rename(columns={
        "qobs": "qobs_mean",
        "Q_mm_day": "Q_mm"
    })

    monthly["residual_mm"] = monthly["P"] - monthly["ET"] - monthly["Q_mm"] - monthly["dS_snow"]

    print("Mánaðarleg samantekt:")
    print(monthly.head())
    print()

    annual = (
        df.set_index("date")
        .resample("YS")
        .agg({
            "P": "sum",
            "ET": "sum",
            "Q_mm_day": "sum",
            "dS_snow": "sum"
        })
        .reset_index()
        .rename(columns={"Q_mm_day": "Q_mm"})
    )

    annual["residual_mm"] = annual["P"] - annual["ET"] - annual["Q_mm"] - annual["dS_snow"]

    print("Meðaltöl yfir allt tímabilið:")
    print("P    =", annual["P"].mean())
    print("ET   =", annual["ET"].mean())
    print("Q    =", annual["Q_mm"].mean())
    print("dS   =", annual["dS_snow"].mean())
    print("Leif =", annual["residual_mm"].mean())
    print()

    uncertainty = pd.DataFrame({
        "Liður": ["P", "Q", "ET", "ΔS", "SWE", "Leif"],
        "Uppruni": [
            "CARRA endurgreining",
            "Mælt rennsli, umbreytt í mm/dag",
            "CARRA endurgreining / reiknað",
            "Reiknað úr breytingu í SWE",
            "CARRA endurgreining",
            "Reiknað sem lokunarvilla"
        ],
        "Tegund": [
            "Reiknað/líkan",
            "Mælt + umbreyting",
            "Reiknað/líkan",
            "Að hluta reiknað, að hluta ófullkomið",
            "Reiknað/líkan",
            "Óþekkt / samsett óvissa"
        ],
        "Helstu óvissuþættir": [
            "Grid-upplausn, staðsetning úrkomu, snjó/úrkomuskil, fjalllendi",
            "Mælivilla, rating-curve, einingar qobs, flatarmál vatnasviðs",
            "Líkangerð, formerki, geislun, vindur, rakastig",
            "SWE nær aðeins snjóforða, ekki jarðvegsvatni/grunnvatni",
            "Háð CARRA-líkani og skölun gagna",
            "Safnar saman öllum skekkjum í P, ET, Q og ΔS"
        ]
    })

    print(uncertainty)
    print()

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["date"], monthly["P"], label="P (úrkoma)")
    plt.plot(monthly["date"], monthly["ET"], label="ET")
    plt.plot(monthly["date"], monthly["dS_snow"], label="ΔS_snjór")
    plt.legend()
    plt.title("Mánaðarlegar stærðir í grunnlíkingu vatnafræðinnar")
    plt.xlabel("Ár")
    plt.ylabel("mm/mánuð")
    plt.tight_layout()
    plt.savefig("figures/manadarlegar_stærðir.png")

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["date"], monthly["qobs_mean"])
    plt.title("Mælt rennsli (qobs), mánaðarlegt meðaltal")
    plt.xlabel("Ár")
    plt.ylabel("qobs")
    plt.tight_layout()
    plt.savefig("figures/manadarlegt_medaltal_rennsli.png")

    plt.figure(figsize=(12, 4))
    plt.plot(monthly["date"], monthly["residual_mm"])
    plt.axhline(0, linestyle="--")
    plt.title("Leif vatnajafnvægis: P - ET - Q - ΔSsnjór")
    plt.xlabel("Ár")
    plt.ylabel("mm/mánuð")
    plt.tight_layout()
    plt.savefig("figures/leif_vatnajafnvaegis.png")   

    plt.figure(figsize=(12, 4))
    plt.plot(monthly["date"], monthly["SWE"])
    plt.title("Meðalsnjóforði (SWE) eftir mánuðum")
    plt.xlabel("Ár")
    plt.ylabel("mm")
    plt.tight_layout()
    plt.savefig("figures/medalsnjofordi.png")   
