import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_lidur5():
    #Lesa skrár
    flow_file = "lamah_ice/D_gauges/2_timeseries/daily/ID_12.csv"
    attr_file = "lamah_ice/A_basins_total_upstrm/1_attributes/Catchment_attributes.csv"

    df = pd.read_csv(flow_file, sep=";")
    attr = pd.read_csv(attr_file, sep=";")

    #Búa til dagsetningu
    df["date"] = pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))

    #Vel rennslisdálk
    if "qobs" in df.columns:
        q_col = "qobs"
    else:
        raise ValueError("Fann ekki dálkinn 'qobs' í ID_12_rennsli.csv")

    #Hreinsa gögnin
    data = df[["date", q_col]].copy()
    data = data.rename(columns={q_col: "Q"})
    data["Q"] = pd.to_numeric(data["Q"], errors="coerce")
    data = data.dropna(subset=["Q"]).reset_index(drop=True)

    #Reikna flow duration curve
    fdc = data.sort_values("Q", ascending=False).reset_index(drop=True)

    n = len(fdc)
    fdc["rank"] = np.arange(1, n + 1)

    #Exceedance probability
    fdc["exceedance"] = 100 * fdc["rank"] / (n + 1)

    #Reikna Q5, Q50 og Q95
    Q5 = np.interp(5, fdc["exceedance"], fdc["Q"])
    Q50 = np.interp(50, fdc["exceedance"], fdc["Q"])
    Q95 = np.interp(95, fdc["exceedance"], fdc["Q"])

    print("Niðurstöður úr flow duration curve:")
    print(f"Q5  (hárennsli) = {Q5:.3f}")
    print(f"Q50 (miðgildi)  = {Q50:.3f}")
    print(f"Q95 (lágrennsli)= {Q95:.3f}")

    #Sækja lýsigögn fyrir vatnasvið 12
    catchment_id = 12
    row = attr[attr["id"] == catchment_id].copy()

    if row.empty:
        raise ValueError(f"Fann ekki id = {catchment_id} í Catchment_attributes.csv")

    row = row.iloc[0]

    #Ná í hjálparbreytur ef þær eru til
    def get_value(col):
        return row[col] if col in row.index else np.nan

    q_mean = get_value("q_mean")
    slope_fdc = get_value("slope_fdc")
    bfi = get_value("baseflow_index_ladson")
    glac_fra = get_value("glac_fra")
    lake_fra = get_value("lake_fra")
    wetl_fra = get_value("wetl_fra")
    frac_snow = get_value("frac_snow")
    runoff_ratio = get_value("runoff_ratio")
    area_calc = get_value("area_calc")
    elev_mean = get_value("elev_mean")

    print("\nLýsigögn úr Catchment_attributes.csv:")
    print(f"q_mean                = {q_mean}")
    print(f"slope_fdc             = {slope_fdc}")
    print(f"baseflow_index        = {bfi}")
    print(f"glac_fra              = {glac_fra}")
    print(f"lake_fra              = {lake_fra}")
    print(f"wetl_fra              = {wetl_fra}")
    print(f"frac_snow             = {frac_snow}")
    print(f"runoff_ratio          = {runoff_ratio}")
    print(f"area_calc             = {area_calc}")
    print(f"elev_mean             = {elev_mean}")

    #Teikna flow duration curve   
    plt.figure(figsize=(10, 6))
    plt.plot(fdc["exceedance"], fdc["Q"], linewidth=2, label="Flow duration curve")
    plt.scatter([5, 50, 95], [Q5, Q50, Q95], zorder=5)
    plt.axvline(5, linestyle="--")
    plt.axvline(50, linestyle="--")
    plt.axvline(95, linestyle="--")
    plt.text(5, Q5, f"  Q5 = {Q5:.3f}", va="bottom")
    plt.text(50, Q50, f"  Q50 = {Q50:.3f}", va="bottom")
    plt.text(95, Q95, f"  Q95 = {Q95:.3f}", va="bottom")
    plt.xlabel("Exceedance probability (%)")
    plt.ylabel("Rennsli Q")
    plt.title("Langæislína rennslis (Flow Duration Curve)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/langaesilna_fdc.png", dpi=300)

    #Teikna á log-skala
    plt.figure(figsize=(10, 6))
    plt.plot(fdc["exceedance"], fdc["Q"], linewidth=2, label="Flow duration curve")
    plt.yscale("log")
    plt.scatter([5, 50, 95], [Q5, Q50, Q95], zorder=5)
    plt.axvline(5, linestyle="--")
    plt.axvline(50, linestyle="--")
    plt.axvline(95, linestyle="--")
    plt.text(5, Q5, f"  Q5 = {Q5:.3f}", va="bottom")
    plt.text(50, Q50, f"  Q50 = {Q50:.3f}", va="bottom")
    plt.text(95, Q95, f"  Q95 = {Q95:.3f}", va="bottom")
    plt.xlabel("Exceedance probability (%)")
    plt.ylabel("Rennsli Q (log-skali)")
    plt.title("Langæislína rennslis (log-skali)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/langaegislina_log.png")

    print("\nTúlkun:")
    #Sveiflukennd/stöðug út frá FDC hallanum
    if pd.notna(slope_fdc):
        if slope_fdc > 1.5:
            print("- slope_fdc er tiltölulega hár, sem bendir til fremur sveiflukennds rennslis.")
        elif slope_fdc > 1.0:
            print("- slope_fdc bendir til miðlungs breytilegs rennslis.")
        else:
            print("- slope_fdc er fremur lágur, sem bendir til stöðugra rennslis.")
    else:
        print("- slope_fdc vantar, þannig að lögun ferilsins þarf að meta beint af myndinni.")

    #Grunnvatn
    if pd.notna(bfi):
        if bfi >= 0.6:
            print("- Baseflow index er hár, sem bendir til umtalsverðs grunnvatnsframlags og stuðnings við lágrennsli.")
        elif bfi >= 0.4:
            print("- Baseflow index bendir til nokkurs grunnvatnsframlags.")
        else:
            print("- Baseflow index er lágur, sem bendir til takmarkaðs grunnvatnsframlags.")

    #Jöklar
    if pd.notna(glac_fra):
        if glac_fra > 0.1:
            print("- Jöklar gætu haft veruleg áhrif á rennslið.")
        elif glac_fra > 0:
            print("- Jöklar gætu haft einhver áhrif á rennslið.")
        else:
            print("- Jöklar virðast ekki skipta verulegu máli á þessu vatnasviði.")

    #Stöðuvötn
    if pd.notna(lake_fra):
        if lake_fra > 0.1:
            print("- Stöðuvötn gætu jafnað rennslið töluvert og dregið úr flóðtoppum.")
        elif lake_fra > 0:
            print("- Stöðuvötn gætu haft lítilsháttar jöfnunaráhrif á rennslið.")
        else:
            print("- Stöðuvötn virðast hafa lítil eða engin áhrif á rennslið.")

    #Snjór
    if pd.notna(frac_snow):
        if frac_snow > 0.5:
            print("- Hátt hlutfall snjókomu bendir til að snjóbráð geti haft mikil áhrif á árstíðasveiflu rennslis.")
        elif frac_snow > 0.2:
            print("- Snjór gæti haft nokkur áhrif á rennslið.")
        else:
            print("- Snjór virðist ekki vera ráðandi þáttur í rennslissvöruninni.")