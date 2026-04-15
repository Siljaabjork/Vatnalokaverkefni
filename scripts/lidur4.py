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
    print("Vatnasvið 12")
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

    df["P"] = pd.to_numeric(df["prec_carra"], errors="coerce")
    df["ET"] = pd.to_numeric(df["total_et_carra"], errors="coerce")
    df["SWE"] = pd.to_numeric(df["swe_carra"], errors="coerce")
    df["dS_snow"] = df["SWE"].diff()
    df["Q_m3s"] = pd.to_numeric(df["qobs"], errors="coerce")
    df = df.dropna(subset=["P", "ET", "SWE", "Q_m3s"]).copy()

    monthly = (
        df.set_index("date")
        .resample("MS")
        .agg({
            "P": "sum",
            "ET": "sum",
            "dS_snow": "sum",
            "SWE": "mean",
            "qobs": "mean",
            "Q_m3s": "mean"
        })
        .reset_index()
    )

    monthly = monthly.rename(columns={
        "qobs": "qobs_mean",
        "Q_m3s": "Q_m3s_mean"
    })

    annual = (
        df.set_index("date")
        .resample("YS")
        .agg({
            "P": "sum",
            "ET": "sum",
            "dS_snow": "sum",
            "Q_m3s": "mean"
        })
        .reset_index()
        .rename(columns={"Q_m3s": "Q_m3s_mean"})
    )

    print("Meðaltöl yfir allt tímabilið:")
    print("P       =", annual["P"].mean())
    print("ET      =", annual["ET"].mean())
    print("dS      =", annual["dS_snow"].mean())
    print("Q_m3s   =", annual["Q_m3s_mean"].mean())
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
    plt.savefig("figures/manaðarlegar_stærðir.png")

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["date"], monthly["Q_m3s_mean"])
    plt.title("Mælt rennsli, mánaðarlegt meðaltal")
    plt.xlabel("Ár")
    plt.ylabel("Q (m³/s)")
    plt.tight_layout()
    plt.savefig("figures/manaðarlegt_meðaltal_rennsli.png")

    plt.figure(figsize=(12, 4))
    plt.plot(monthly["date"], monthly["SWE"])
    plt.title("Meðalsnjóforði (SWE) eftir mánuðum")
    plt.xlabel("Ár")
    plt.ylabel("mm")
    plt.tight_layout()
    plt.savefig("figures/medalsnjoforði.png")

