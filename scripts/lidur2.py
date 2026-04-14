import pandas as pd
import matplotlib.pyplot as plt

def run_lidur2():

    months = ["Jan", "Feb", "Mar", "Apr", "Maí", "Jún",
          "Júl", "Ágú", "Sep", "Okt", "Nóv", "Des"]

    #Lesa inn gögnin fyrir rennsli
    q_path = "lamah_ice/D_gauges/2_timeseries/daily/ID_12.csv"
    q = pd.read_csv(q_path, sep=';')

    q['date'] = pd.to_datetime({
    'year': q['YYYY'],
    'month': q['MM'],
    'day': q['DD']
    })

    q = q[(q['date'] >= '1993-10-01') & (q['date'] <= '2023-09-30')]
    q['month'] = q['date'].dt.month

    #Meðaltalsár 
    monthly_q = q.groupby('month')['qobs'].mean()

    #Plot fyrir rennsli
    plt.figure(figsize=(8,5))
    plt.plot(monthly_q.index, monthly_q.values)
    plt.xticks(range(1,13), months)
    plt.xlabel("Mánuður")
    plt.ylabel("Rennsli (m³/s)")
    plt.title("Meðaltalsár - Rennsli")
    plt.savefig("figures/rennsli.png")

    #Lesa inn veðurgögnin
    vedur_path = "lamah_ice/A_basins_total_upstrm/2_timeseries/daily/meteorological_data/ID_12.csv"
    vedur = pd.read_csv(vedur_path, sep=';')

    vedur['date'] = pd.to_datetime({
    'year': vedur['YYYY'],
    'month': vedur['MM'],
    'day': vedur['DD']
    })

    df = pd.merge(q, vedur, on='date')
    df['month'] = df['date'].dt.month

    #Meðaltalsár
    monthly = df.groupby('month').mean()

    #Plot fyrir úrkomu
    plt.figure(figsize=(8,5))
    plt.plot(monthly.index, monthly['prec'])
    plt.xticks(range(1,13), months)
    plt.xlabel("Mánuður")
    plt.ylabel("Úrkoma (mm/dag)")
    plt.title("Meðaltalsár - Úrkoma")
    plt.savefig("figures/urkoma.png")

    #Plot fyrir hitastig
    plt.figure(figsize=(8,5))
    plt.plot(monthly.index, monthly['2m_temp_mean'])
    plt.xticks(range(1,13), months)
    plt.xlabel("Mánuður")
    plt.ylabel("Hitastig (°C)")
    plt.title("Meðaltalsár - Hitastig")
    plt.savefig("figures/hitastig.png")