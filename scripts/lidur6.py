import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings

def run_flood_seasonality():
    file_path = "data/rennslidate.xlsx"  
    df = pd.read_excel(file_path)

    df['date'] = pd.to_datetime(df['date'])

    df = df[['date', 'qobs']].dropna()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    annual_peaks = df.loc[df.groupby('year')['qobs'].idxmax()]

    monthly_counts = annual_peaks['month'].value_counts().sort_index()

    plt.figure()
    monthly_counts.plot(kind='bar')
    plt.xlabel("Mánuður")
    plt.ylabel("Fjöldi annual peaks")
    plt.title("Flood Seasonality – Fjöldi árlegra flóða eftir mánuðum")
    plt.xticks(
        ticks=range(12),
        labels=["Jan", "Feb", "Mar", "Apr", "Maí", "Jún",
                "Júl", "Ágú", "Sep", "Okt", "Nóv", "Des"],
        rotation=45
    )
    plt.tight_layout()
    plt.savefig("figures/flood_seasonality.png")

    print("Fjöldi annual peaks í hverjum mánuði:")
    print(monthly_counts)

def run_flood_frequency_analysis():    
    warnings.filterwarnings("ignore")

    file_path = "data/rennslidate.xlsx" 
    df = pd.read_excel(file_path)

    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'qobs']].dropna().copy()

    df['qobs'] = pd.to_numeric(df['qobs'], errors='coerce')
    df = df.dropna(subset=['qobs'])

    df['year'] = df['date'].dt.year

    annual_peaks = df.loc[df.groupby('year')['qobs'].idxmax()].copy()
    annual_peaks = annual_peaks.sort_values('year').reset_index(drop=True)

    peaks = annual_peaks['qobs'].values
    years = annual_peaks['year'].values
    n = len(peaks)

    print("Fjöldi annual peaks:", n)
    print("\nAnnual peaks:")
    print(annual_peaks[['year', 'date', 'qobs']])

    x_sorted = np.sort(peaks)
    i = np.arange(1, n + 1)

    F_gringorten = (i - 0.44) / (n + 0.12)

    T_gringorten = 1 / (1 - F_gringorten)

    params_gumbel = stats.gumbel_r.fit(peaks)

    params_lognorm3 = stats.lognorm.fit(peaks)

    if np.any(peaks <= 0):
        raise ValueError("Log-dreifingar krefjast jákvæðra peak-gilda.")
    log_peaks = np.log10(peaks)
    params_lp3 = stats.pearson3.fit(log_peaks)

    def q_gumbel(F, params):
        loc, scale = params
        return stats.gumbel_r.ppf(F, loc=loc, scale=scale)

    def q_lognorm3(F, params):
        s, loc, scale = params
        return stats.lognorm.ppf(F, s=s, loc=loc, scale=scale)

    def q_lp3(F, params):
        skew, loc, scale = params
        qlog = stats.pearson3.ppf(F, skew=skew, loc=loc, scale=scale)
        return 10 ** qlog

    def loglik_gumbel(x, params):
        loc, scale = params
        return np.sum(stats.gumbel_r.logpdf(x, loc=loc, scale=scale))

    def loglik_lognorm3(x, params):
        s, loc, scale = params
        return np.sum(stats.lognorm.logpdf(x, s=s, loc=loc, scale=scale))

    def loglik_lp3(x, params):
        skew, loc, scale = params
        y = np.log10(x)
        logpdf_y = stats.pearson3.logpdf(y, skew=skew, loc=loc, scale=scale)
        log_jac = -np.log(x * np.log(10))
        return np.sum(logpdf_y + log_jac)

    def aic(loglik, k):
        return 2 * k - 2 * loglik

    xhat_gumbel = q_gumbel(F_gringorten, params_gumbel)
    xhat_lognorm3 = q_lognorm3(F_gringorten, params_lognorm3)
    xhat_lp3 = q_lp3(F_gringorten, params_lp3)

    rmse_gumbel = np.sqrt(np.mean((x_sorted - xhat_gumbel) ** 2))
    rmse_lognorm3 = np.sqrt(np.mean((x_sorted - xhat_lognorm3) ** 2))
    rmse_lp3 = np.sqrt(np.mean((x_sorted - xhat_lp3) ** 2))

    ll_gumbel = loglik_gumbel(peaks, params_gumbel)
    ll_lognorm3 = loglik_lognorm3(peaks, params_lognorm3)
    ll_lp3 = loglik_lp3(peaks, params_lp3)

    results = pd.DataFrame({
        'Distribution': ['Gumbel', 'Log Normal 3', 'Log Pearson 3'],
        'RMSE': [rmse_gumbel, rmse_lognorm3, rmse_lp3],
        'LogLik': [ll_gumbel, ll_lognorm3, ll_lp3],
        'AIC': [aic(ll_gumbel, 2), aic(ll_lognorm3, 3), aic(ll_lp3, 3)]
    }).sort_values('RMSE').reset_index(drop=True)

    print("\nSamanburður dreifinga:")
    print(results)

    best_dist = results.loc[0, 'Distribution']
    print(f"\nBest fitting distribution samkvæmt RMSE: {best_dist}")

    return_periods = np.array([10, 50, 100], dtype=float)
    F_return = 1 - 1 / return_periods

    def estimate_quantiles(dist_name, F):
        if dist_name == 'Gumbel':
            return q_gumbel(F, params_gumbel)
        elif dist_name == 'Log Normal 3':
            return q_lognorm3(F, params_lognorm3)
        elif dist_name == 'Log Pearson 3':
            return q_lp3(F, params_lp3)
        else:
            raise ValueError("Óþekkt dreifing")

    Q_est = estimate_quantiles(best_dist, F_return)

    q_table = pd.DataFrame({
        'Return period (years)': return_periods.astype(int),
        'Non-exceedance probability F': F_return,
        'Estimated flow': Q_est
    })

    print("\nHönnunarrennsli fyrir bestu dreifingu:")
    print(q_table)

    rng = np.random.default_rng(42)
    B = 500
    boot_q = np.zeros((B, len(return_periods)))

    for b in range(B):
        sample = rng.choice(peaks, size=n, replace=True)
        try:
            if best_dist == 'Gumbel':
                p = stats.gumbel_r.fit(sample)
                boot_q[b, :] = q_gumbel(F_return, p)
            elif best_dist == 'Log Normal 3':
                p = stats.lognorm.fit(sample)
                boot_q[b, :] = q_lognorm3(F_return, p)
            elif best_dist == 'Log Pearson 3':
                if np.any(sample <= 0):
                    boot_q[b, :] = np.nan
                else:
                    lp = stats.pearson3.fit(np.log10(sample))
                    boot_q[b, :] = q_lp3(F_return, lp)
        except Exception:
            boot_q[b, :] = np.nan

    boot_q = boot_q[~np.isnan(boot_q).any(axis=1)]

    lower = np.percentile(boot_q, 5, axis=0)
    upper = np.percentile(boot_q, 95, axis=0)

    ci_table = pd.DataFrame({
        'Return period (years)': return_periods.astype(int),
        'Q estimate': Q_est,
        '90% CI lower': lower,
        '90% CI upper': upper
    })

    print("\n90% confidence interval (bootstrap):")
    print(ci_table)

    T_plot = np.linspace(1.01, 200, 500)
    F_plot = 1 - 1 / T_plot

    q_plot_gumbel = q_gumbel(F_plot, params_gumbel)
    q_plot_lognorm3 = q_lognorm3(F_plot, params_lognorm3)
    q_plot_lp3 = q_lp3(F_plot, params_lp3)

    plt.figure(figsize=(10, 6))
    plt.scatter(T_gringorten, x_sorted, label='Observed annual peaks')
    plt.plot(T_plot, q_plot_gumbel, '-', linewidth=2, label='Gumbel', color='y')
    plt.plot(T_plot, q_plot_lognorm3, '--', linewidth=2, label='Log Normal 3', color='m')
    plt.plot(T_plot, q_plot_lp3, ':', linewidth=3, label='Log Pearson 3', color='g')
    plt.xscale('log')
    plt.xlabel('Return period T (ár)')
    plt.ylabel('Hámarksrennsli (m3/s)')
    plt.title('Flóðatíðnigreining með Gringorten')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/gringorten.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(T_gringorten, x_sorted, label='Observed annual peaks')
    plt.plot(return_periods, Q_est, marker='o', label=f'{best_dist} estimate')
    plt.fill_between(return_periods, lower, upper, alpha=0.3, label='90% CI')
    plt.xscale('log')
    plt.xlabel('Return period T (ár)')
    plt.ylabel('Hámarksrennsli (m3/s)')
    plt.title(f'Design floods and 90% CI ({best_dist})')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/design_floods.png")

def run_lidur6():
    run_flood_seasonality()
    run_flood_frequency_analysis()
