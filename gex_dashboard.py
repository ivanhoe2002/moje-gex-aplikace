Skvělé, váš původní kód je velmi čistě napsaný. Upravil jsem ho přesně tak, aby všechny výpočty na pozadí dál běžely z robustních opčních dat QQQ, ale uživatelské rozhraní, graf a vypočítané hladiny se automaticky přepočítaly na ekvivalent futures kontraktu MNQ.
Aplikace si nově stáhne i aktuální cenu Micro E-mini Nasdaq futures (MNQ=F) z Yahoo Finance a spočítá přesný koeficient převodu, takže hladiny budou perfektně sedět i v případě intradenních pohybů.
Zde je kompletní upravený kód. Celý ho zkopírujte a přepište jím obsah souboru gex_dashboard.py na vašem GitHubu:
python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Nasdaq MNQ GEX Dashboard", layout="wide")
st.title("📊 Nasdaq Micro E-mini Futures (MNQ) Gamma Exposure Dashboard")
st.markdown("Tato aplikace stahuje opční data z QQQ, ale automaticky je přepočítává a vizualizuje pro hladiny **MNQ Futures**.")

ticker_symbol = "QQQ"
futures_symbol = "MNQ=F"

@st.cache_resource(ttl=3600)
def load_options_data(ticker, futures_tick):
    tk = yf.Ticker(ticker)
    ft = yf.Ticker(futures_tick)
    
    expirations = tk.options
    if not expirations:
        return None, None, None
    
    # Vybereme nejbližší expiraci opcí
    closest_exp = expirations[0]
    opt = tk.option_chain(closest_exp)
    
    # Získání aktuální ceny podkladu (QQQ)
    history = tk.history(period="1d")
    spot_price = history['Close'].iloc[-1]
    
    # Získání aktuální ceny futures (MNQ)
    try:
        ft_history = ft.history(period="1d")
        futures_price = ft_history['Close'].iloc[-1]
    except Exception:
        # Záložní plán: pokud by Yahoo futures symbol zrovna stávkoval, použijeme přibližný násobek indexu
        futures_price = spot_price * 40.0
        
    # Převodní faktor mezi QQQ a MNQ Futures
    conversion_factor = futures_price / spot_price
    
    return opt, spot_price, conversion_factor

opt_data, spot_qqq, conversion_factor = load_options_data(ticker_symbol, futures_symbol)

if opt_data is None:
    st.error("Nepodařilo se načíst opční nebo futures data. Zkontrolujte připojení.")
else:
    # Přepočet aktuální spotové ceny z QQQ na MNQ
    spot = spot_qqq * conversion_factor
    
    calls = opt_data.calls.copy()
    puts = opt_data.puts.copy()
    
    # Přepočet realizačních cen (strikes) v opčním řetězci z QQQ na MNQ ceny
    calls['strike'] = calls['strike'] * conversion_factor
    puts['strike'] = puts['strike'] * conversion_factor
    
    # Zjednodušený výpočet Implikované Volatility a Gammy pomocí Black-Scholes (přizpůsobený pro MNQ strikes)
    def calculate_gex(df, is_call, spot_p):
        gex_list = []
        S = spot_p
        T = 1/252 # Předpoklad 1 den do expirace u nejbližší opce
        r = 0.04
        
        for idx, row in df.iterrows():
            K = row['strike']
            oi = row['openInterest']
            iv = row['impliedVolatility']
            
            if pd.isna(oi) or oi <= 0 or pd.isna(iv) or iv <= 0:
                gex_list.append(0)
                continue
                
            d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
            
            # Výpočet expozice normalizovaný na body indexu
            if is_call:
                gex = oi * gamma * S * S * 0.1 * 100
            else:
                gex = -oi * gamma * S * S * 0.1 * 100
                
            gex_list.append(gex)
        return gex_list

    calls['GEX'] = calculate_gex(calls, is_call=True, spot_p=spot)
    puts['GEX'] = calculate_gex(puts, is_call=False, spot_p=spot)
    
    # Sloučení dat
    df_calls = calls[['strike', 'GEX']].copy()
    df_puts = puts[['strike', 'GEX']].copy()
    
    df_all = pd.merge(df_calls, df_puts, on='strike', suffixes=('_call', '_put'), how='outer').fillna(0)
    df_all['Total_GEX'] = df_all['GEX_call'] + df_all['GEX_put']
    
    # Filtrování strikes kolem spotové ceny MNQ (rozmezí +-10 %)
    df_filtered = df_all[(df_all['strike'] >= spot * 0.9) & (df_all['strike'] <= spot * 1.1)]
    
    # Určení klíčových hladin na cenách MNQ futures
    call_wall = df_filtered.loc[df_filtered['GEX_call'].idxmax()]['strike']
    put_wall = df_filtered.loc[df_filtered['GEX_put'].idxmin()]['strike']
    
    # Gamma Flip (Bod, kde se GEX mění z kladného do záporného poblíž spotu MNQ)
    df_sorted = df_filtered.sort_values(by='strike')
    gamma_flip = spot
    for i in range(len(df_sorted)-1):
        g1 = df_sorted.iloc[i]['Total_GEX']
        g2 = df_sorted.iloc[i+1]['Total_GEX']
        if (g1 < 0 and g2 > 0) or (g1 > 0 and g2 < 0):
            gamma_flip = df_sorted.iloc[i]['strike']
            break

    # Metriky navrchu stránky přepočítané na MNQ1!
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Aktuální MNQ Spot", f"{spot:.1f}")
    col2.metric("🟢 Call Wall (Strop pro MNQ)", f"{call_wall:.1f}")
    col3.metric("🔴 Put Wall (Beton pro MNQ)", f"{put_wall:.1f}")
    col4.metric("⚖️ MNQ Gamma Flip Line", f"{gamma_flip:.1f}")
    
    # Vykreslení grafu pomocí Plotly
    fig = go.Figure()
    
    # Total GEX profil
    fig.add_trace(go.Bar(
        y=df_filtered['strike'],
        x=df_filtered['Total_GEX'],
        orientation='h',
        name='Čistý GEX (Net Gamma)',
        marker_color=np.where(df_filtered['Total_GEX'] >= 0, 'green', 'red')
    ))
    
    # Přidání linky pro nulovou osu
    fig.add_shape(type="line", y0=df_filtered['strike'].min(), y1=df_filtered['strike'].max(), x0=0, x1=0,
                  line=dict(color="black", width=1, dash="dot"))
    
    fig.add_hline(y=spot, line_color="orange", line_width=2, line_dash="solid", annotation_text="Aktuální MNQ SPOT")
    fig.add_hline(y=gamma_flip, line_color="purple", line_width=1.5, line_dash="dash", annotation_text="MNQ Gamma Flip")
    
    fig.update_layout(
        title=f"GEX Profil přepočítaný pro MNQ Futures (Ceny kontraktu vs Expozice)",
        xaxis_title="Gamma Exposure (Záporná = Volatilita, Kladná = Klid)",
        yaxis_title="MNQ Realizační Cena (Points)",
        height=700,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("Tip pro MNQ trading: Sledujte, jak se ranní hladiny kryjí s POC nebo VAL/VAH na vašem Volume Profilu v platformě TradingView.")
Při použití kódu buďte obezřetní.

