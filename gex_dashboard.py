import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Nasdaq GEX Dashboard", layout="wide")
st.title("📊 Nasdaq (QQQ) Gamma Exposure (GEX) Dashboard")
st.markdown("Tato aplikace stahuje bezplatná opční data z Yahoo Finance a počítá odhad denních GEX hladin.")

ticker_symbol = "QQQ"

@st.cache_resource(ttl=3600)
def load_options_data(ticker):
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        return None, None
    
    # Vybereme nejbližší expiraci
    closest_exp = expirations[0]
    opt = tk.option_chain(closest_exp)
    
    # Získání aktuální ceny podkladu
    history = tk.history(period="1d")
    spot_price = history['Close'].iloc[-1]
    
    return opt, spot_price

opt_data, spot = load_options_data(ticker_symbol)

if opt_data is None:
    st.error("Nepodařilo se načíst opční data. Zkontrolujte připojení nebo ticker.")
else:
    calls = opt_data.calls
    puts = opt_data.puts
    
    # Zjednodušený výpočet Implikované Volatility a Gammy pomocí Black-Scholes (přibližný dopočet)
    def calculate_gex(df, is_call, spot_p):
        gex_list = []
        # Parametry pro přibližnou gammu (zjednodušený model, pokud chybí řecká písmena)
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
            
            # Velikost kontraktu je 100 akcí. 
            # Předpoklad: Retail nakupuje Puts (MM je long gamma = +) a nakupuje Calls (MM je short gamma = -)
            # Častý standardní přístup: SpotGamma/GEX reprezentace z pohledu celkového open interestu:
            if is_call:
                gex = oi * gamma * S * S * 0.1 * 100  # Normalizovaný GEX v milionech USD na bod změny
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
    
    # Filtrování strikes kolem spotové ceny
    df_filtered = df_all[(df_all['strike'] >= spot * 0.9) & (df_all['strike'] <= spot * 1.1)]
    
    # Určení klíčových hladin
    call_wall = df_filtered.loc[df_filtered['GEX_call'].idxmax()]['strike']
    put_wall = df_filtered.loc[df_filtered['GEX_put'].idxmin()]['strike']
    
    # Gamma Flip (Bod, kde se GEX mění z kladného do záporného poblíž spotu)
    df_sorted = df_filtered.sort_values(by='strike')
    gamma_flip = spot
    for i in range(len(df_sorted)-1):
        g1 = df_sorted.iloc[i]['Total_GEX']
        g2 = df_sorted.iloc[i+1]['Total_GEX']
        if (g1 < 0 and g2 > 0) or (g1 > 0 and g2 < 0):
            gamma_flip = df_sorted.iloc[i]['strike']
            break

    # Metriky navrchu stránky
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Aktuální QQQ Spot", f"${spot:.2f}")
    col2.metric("🟢 Call Wall (Strop)", f"${call_wall:.2f}")
    col3.metric("🔴 Put Wall (Beton)", f"${put_wall:.2f}")
    col4.metric("⚖️ Gamma Flip Line", f"${gamma_flip:.2f}")
    
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
    
    # Přidání linky pro aktuální spot cenu
    fig.add_shape(type="line", y0=df_filtered['strike'].min(), y1=df_filtered['strike'].max(), x0=0, x1=0,
                  line=dict(color="black", width=1, dash="dot"))
    
    fig.add_hline(y=spot, line_color="orange", line_width=2, line_dash="solid", annotation_text="Aktuální SPOT")
    fig.add_hline(y=gamma_flip, line_color="purple", line_width=1.5, line_dash="dash", annotation_text="Gamma Flip")
    
    fig.update_layout(
        title=f"GEX Profil pro QQQ (Strikes vs Expozice v mil. $)",
        xaxis_title="Gamma Exposure (Záporná = Volatilita, Kladná = Klid)",
        yaxis_title="Strike Cena ($)",
        height=700,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("Tip pro trading: Pokud jsme NAD fialovou linkou (Gamma Flip), trh má tendenci se uklidňovat. Pokud propadneme POD ni, volatilita prudce roste.")
