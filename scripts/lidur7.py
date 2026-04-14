import pymannkendall as mk
import pandas as pd
import numpy as np
from scipy.stats import theilslopes
import matplotlib.pyplot as plt

def run_lidur7():
    #Lesa gögnin
    df = pd.read_csv("lamah_ice/D_gauges/2_timeseries/daily/ID_12.csv", sep=";")

    #Halda bara í dagsetningu og mælt rennsli
    df = df[["YYYY", "MM", "DD", "qobs"]].dropna().copy()

    #Búa til dagsetningu
    df["date"] = pd.to_datetime(
    dict(year=df["YYYY"], month=df["MM"], day=df["DD"])
    )

    #ÁR og MÁNUÐUR
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    #Skilgreina árstíðir
    def season_from_month(m):
        if m in [12, 1, 2]:
            return "Vetur"
        elif m in [3, 4, 5]:
            return "Vor"
        elif m in [6, 7, 8]:
            return "Sumar"
        else:
            return "Haust"

    df["season"] = df["month"].apply(season_from_month)

    #Látum desember tilheyra næsta vetri
    df["season_year"] = df["year"]
    df.loc[df["month"] == 12, "season_year"] = df.loc[df["month"] == 12, "year"] + 1

    #Búa til árs- og árstíðagögn

    #Árlegt meðalrennsli
    annual = df.groupby("year", as_index=False)["qobs"].mean()
    annual.columns = ["year", "Q_mean"]

    #Árstíðabundið meðalrennsli
    seasonal = df.groupby(["season_year", "season"], as_index=False)["qobs"].mean()
    seasonal.columns = ["year", "season", "Q_mean"]

    #Fall sem reiknar Theil-Sen + modified Mann-Kendall
    def trend_analysis(data, time_col="year", value_col="Q_mean"):
        x = data[time_col].values
        y = data[value_col].values
        #Theil-Sen estimator
        slope, intercept, low_slope, high_slope = theilslopes(y, x, 0.95)
        #Modified Mann-Kendall test (Hamed & Rao)
        mk_result = mk.hamed_rao_modification_test(y)
        #Pakka niðurstöðum
        result = {
        "slope": slope,
        "intercept": intercept,
        "low_slope_95CI": low_slope,
        "high_slope_95CI": high_slope,
        "trend": mk_result.trend,
        "p_value": mk_result.p,
        "tau": mk_result.Tau,
        "z": mk_result.z,
        "significant": "Já" if mk_result.p < 0.05 else "Nei"
        }
        return result

    #Reikna leitni fyrir ársgrunngögn
    annual_result = trend_analysis(annual)

    #Reikna leitni fyrir hverja árstíð
    season_results = []

    for season_name in ["Vetur", "Vor", "Sumar", "Haust"]:
        temp = seasonal[seasonal["season"] == season_name].copy()
        res = trend_analysis(temp)
        res["season"] = season_name
        season_results.append(res)

    season_results_df = pd.DataFrame(season_results)

    #Sýna niðurstöður
    print("ÁRSGRUNNUR")
    print(f"Theil-Sen halli: {annual_result['slope']:.4f} m³/s á ári")
    print(f"95% CI: [{annual_result['low_slope_95CI']:.4f}, {annual_result['high_slope_95CI']:.4f}]")
    print(f"Mann-Kendall trend: {annual_result['trend']}")
    print(f"p-gildi: {annual_result['p_value']:.4f}")
    print(f"Marktækt (p < 0.05): {annual_result['significant']}")
    print()

    print("ÁRSTÍÐAGRUNNUR")
    print(season_results_df[[
    "season", "slope", "low_slope_95CI", "high_slope_95CI",
    "trend", "p_value", "significant"
    ]])

    #Teikna ársleitni
    plt.figure(figsize=(8, 5))
    plt.plot(annual["year"], annual["Q_mean"], "o-", label="Árlegt meðalrennsli")
    plt.plot(
    annual["year"],
    annual_result["intercept"] + annual_result["slope"] * annual["year"],
    label="Theil-Sen leitnilína"
    )
    plt.xlabel("Ár")
    plt.ylabel("Meðalrennsli (m³/s)")
    plt.title("Leitni í árlegu meðalrennsli")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/arsleitni.png")

    #Teikna árstíðaleitni
    for season_name in ["Vetur", "Vor", "Sumar", "Haust"]:
        temp = seasonal[seasonal["season"] == season_name].copy()
        res = trend_analysis(temp)

        plt.figure(figsize=(8, 5))
        plt.plot(temp["year"], temp["Q_mean"], "o-", label=f"{season_name}")
        plt.plot(
            temp["year"],
            res["intercept"] + res["slope"] * temp["year"],
            label="Theil-Sen leitnilína"
        )
        plt.xlabel("Ár")
        plt.ylabel("Meðalrennsli (m³/s)")
        plt.title(f"Leitni í {season_name.lower()}rennsli")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"figures/arstidaleitni_{season_name}.png")
