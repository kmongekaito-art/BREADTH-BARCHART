import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import asyncio
import nest_asyncio
from io import StringIO
from playwright.async_api import async_playwright
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Permitir asyncio en Streamlit
nest_asyncio.apply()

# Configuración de la página
st.set_page_config(page_title="Market Breadth Analyzer (Barchart)", layout="wide")

# ==========================================
# MAPEO DE TICKERS A SÍMBOLOS DE BARCHART
# ==========================================
BARCHART_SYMBOLS = {
    'SPY': ["$AVVT", "$DVCT", "$HIGN", "$LOWN"],
    'QQQ': ["$AVVQ", "$DVCQ", "$HIGQ", "$LOWQ"]
}

# ==========================================
# PASO 1: SIDEBAR (Selección de Mercado)
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuración")
    st.markdown("### Selección de Índice")
    ticker = st.selectbox("Mercado a analizar:", ["SPY", "QQQ"], index=0)
    st.markdown("---")
    st.info("💡 *La primera vez tardará ~15s. Si falla, usa el menú (⋮) -> 'Clear cache' y recarga.*")

# ==========================================
# PASO 2: DESCARGA DE DATOS (Lógica original de Colab + Espera Segura)
# ==========================================
async def capturar_csv_barchart(simbolos):
    """Función asíncrona robusta basada en tu lógica de Colab, con sincronización para evitar cierres prematuros"""
    resultados = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
        )
        
        for simbolo in simbolos:
            url = f"https://www.barchart.com/stocks/quotes/{simbolo}/interactive-chart"
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                viewport={"width": 1920, "height": 1080}, locale="en-US"
            )
            page = await context.new_page()
            raw_text = [None]
            
            # Listener original que funciona en Colab
            async def handle_response(response):
                if "queryeod" in response.url and response.ok:
                    raw_text[0] = await response.text()
            
            page.on("response", lambda resp: asyncio.ensure_future(handle_response(resp)))
            await page.goto(url, wait_until="networkidle")
            
            # ESPERA SEGURA: Esperamos hasta que raw_text[0] tenga datos o pasen 20 segundos.
            # Esto evita el 'TargetClosedError' porque NO cerramos el navegador hasta que la 
            # tarea en segundo plano ha terminado de leer el texto exitosamente.
            waited = 0
            while raw_text[0] is None and waited < 20000:
                await asyncio.sleep(0.5)
                waited += 500
                
            if raw_text[0]:
                try:
                    header = "Symbol,Date,Open,High,Low,Close,Volume"
                    csv_completo = header + "\n" + raw_text[0].strip()
                    df = pd.read_csv(StringIO(csv_completo))
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date').set_index('Date')
                    resultados[simbolo] = df['Close']
                except Exception as e:
                    print(f"⚠️ Error procesando {simbolo}: {e}")
            else:
                print(f"⚠️ Timeout (20s) esperando respuesta de {simbolo}")
                
            # Cerramos el contexto de forma ordenada para este símbolo
            await context.close()
            
        await browser.close()
    return resultados

@st.cache_data(show_spinner="🕷️ Scraping Barchart (Primera vez puede tardar 10-20s)...", ttl=3600)
def get_barchart_data(ticker_selected):
    simbolos = BARCHART_SYMBOLS[ticker_selected]
    cierres_dict = asyncio.run(capturar_csv_barchart(simbolos))
    
    # VERIFICACIÓN DE INTEGRIDAD
    if len(cierres_dict) < len(simbolos):
        missing = set(simbolos) - set(cierres_dict.keys())
        st.error(f"⚠️ **Fallo en la descarga:** No se pudieron obtener los datos de {missing}. Por favor, recarga la página (F5).")
        st.stop()
        
    if cierres_dict:
        df = pd.concat(cierres_dict, axis=1)
        df.columns = [col.replace('$', '') for col in df.columns]
        return df
    return None

# ==========================================
# PASO 3: CÁLCULO DE INDICADORES
# ==========================================
@st.cache_data(show_spinner="🧮 Calculando indicadores de amplitud...")
def calculate_indicators(df_barchart, ticker_selected):
    SPADVDEC = pd.DataFrame(index=df_barchart.index)
    
    # Mapeo seguro y explícito (Corregido: HIGN/LOWN para SPY, HIGQ/LOWQ para QQQ)
    if ticker_selected == 'SPY':
        col_map = {'Avance': 'AVVT', 'Descenso': 'DVCT', 'NewHigh': 'HIGN', 'NewLow': 'LOWN'}
    else: # QQQ
        col_map = {'Avance': 'AVVQ', 'Descenso': 'DVCQ', 'NewHigh': 'HIGQ', 'NewLow': 'LOWQ'}

    for new_name, old_name in col_map.items():
        if old_name in df_barchart.columns:
            SPADVDEC[new_name] = df_barchart[old_name]
        else:
            st.error(f"⚠️ Error interno: La columna {old_name} no existe en el DataFrame descargado.")
            st.stop()

    # Cálculos (Lógica de tu script de Colab)
    SPADVDEC['LIN'] = (SPADVDEC['Avance'] - SPADVDEC['Descenso']).cumsum()
    SPADVDEC['RATIO'] = (SPADVDEC['Avance'] - SPADVDEC['Descenso']) / (SPADVDEC['Avance'] + SPADVDEC['Descenso'])
    SPADVDEC['LINADN'] = SPADVDEC['RATIO'].cumsum()
    
    roll_min = SPADVDEC['LINADN'].rolling(window=21).min()
    roll_max = SPADVDEC['LINADN'].rolling(window=21).max()
    SPADVDEC['FORMULA'] = ((SPADVDEC['LINADN'] - roll_min) / (roll_max - roll_min)) * 100
    SPADVDEC['ADn'] = SPADVDEC['FORMULA'].ewm(span=7).mean()
    
    SPADVDEC['MIMACD'] = SPADVDEC['LIN'].ewm(span=3).mean() - SPADVDEC['LIN'].ewm(span=8).mean()
    SPADVDEC['EMAMIMACD'] = SPADVDEC['MIMACD'].ewm(span=2, adjust=False).mean()
    
    SPADVDEC['SUMM_SHORT'] = (SPADVDEC['Avance'] - SPADVDEC['Descenso']).ewm(span=19).mean()
    SPADVDEC['SUMM_LONG'] = (SPADVDEC['Avance'] - SPADVDEC['Descenso']).ewm(span=39).mean()
    SPADVDEC['SUMM'] = (SPADVDEC['SUMM_SHORT'] - SPADVDEC['SUMM_LONG']).cumsum()
    
    SPADVDEC['AvanceEMA19'] = SPADVDEC['Avance'].ewm(span=19).mean()
    SPADVDEC['AvanceEMA39'] = SPADVDEC['Avance'].ewm(span=39).mean()
    SPADVDEC['McCellan'] = SPADVDEC['AvanceEMA19'] - SPADVDEC['AvanceEMA39']
    
    SPADVDEC['longMIMACD'] = SPADVDEC['LIN'].ewm(span=12).mean() - SPADVDEC['LIN'].ewm(span=26).mean()
    SPADVDEC['longEMAMIMACD'] = SPADVDEC['longMIMACD'].ewm(span=9).mean()
    
    return SPADVDEC

# ==========================================
# EJECUCIÓN DE CARGA DE DATOS
# ==========================================
# 1. Precio del ETF (yfinance)
stock = yf.download(ticker, start='2020-01-01', interval='1d', progress=False, auto_adjust=True)
if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.get_level_values(0)

# 2. Datos de Barchart e Indicadores
df_barchart = get_barchart_data(ticker)
if df_barchart is not None:
    SPADVDEC = calculate_indicators(df_barchart, ticker)
else:
    st.error("No se pudieron descargar los datos de Barchart.")
    st.stop()

# ==========================================
# PASO 4: INTERFAZ (Layout y Controles)
# ==========================================
INDICATOR_GROUPS = {
    'MIMACD + Señal': ['MIMACD', 'EMAMIMACD'],
    'McClellan Oscillator': ['McCellan'],
    'SUMM Index': ['SUMM'],
    'Long MIMACD + Señal': ['longMIMACD', 'longEMAMIMACD'],
    'ADn (Accumulation/Distribution)': ['ADn'],
    'Avances vs Descensos': ['Avance', 'Descenso'],
    'New Highs vs New Lows': ['NewHigh', 'NewLow']
}

col_chart, col_controls = st.columns([4, 1])

with col_controls:
    st.subheader("📊 Config. Paneles")
    num_panels = st.selectbox("Nº de paneles inferiores:", [1, 2, 3], index=1)
    
    group_names = list(INDICATOR_GROUPS.keys())
    selected_indicators = []
    
    default_options = ['MIMACD + Señal', 'New Highs vs New Lows', 'McClellan Oscillator']
    for i in range(num_panels):
        default_idx = group_names.index(default_options[i]) if i < len(default_options) else 0
        ind = st.selectbox(f"Panel {i+1}:", group_names, index=default_idx, key=f"ind_{i}")
        selected_indicators.append(ind)

# ==========================================
# PASO 5: GRÁFICO DINÁMICO PLOTLY
# ==========================================
with col_chart:
    total_rows = 1 + num_panels
    row_heights = [0.6] + [0.4 / num_panels] * num_panels
    specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * num_panels

    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=row_heights, specs=specs
    )

    # Panel Principal
    fig.add_trace(go.Candlestick(
        x=stock.index, open=stock['Open'], high=stock['High'],
        low=stock['Low'], close=stock['Close'], name=ticker,
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=stock.index, y=stock['Volume'], name='Volumen', opacity=0.3, marker_color='gray'
    ), row=1, col=1, secondary_y=True)

    # Paneles de Indicadores
    colors = ['#2962FF', '#FF6D00', '#00C853', '#D50000', '#9C27B0']
    for i, group_name in enumerate(selected_indicators):
        row_num = i + 2
        indicators_in_group = INDICATOR_GROUPS[group_name]
        
        for j, col_name in enumerate(indicators_in_group):
            color = colors[j % len(colors)]
            fig.add_trace(go.Scatter(
                x=SPADVDEC.index, y=SPADVDEC[col_name],
                line=dict(color=color, width=1.5), name=col_name
            ), row=row_num, col=1)

    # Layout general
    fig.update_layout(
        title=f"Análisis de Amplitud de Mercado: {ticker}",
        xaxis_rangeslider_visible=False,
        height=850,
        hovermode="x",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Precio", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Volumen", secondary_y=True, row=1, col=1, showgrid=False)
    
    for i in range(num_panels):
        fig.update_yaxes(title_text=selected_indicators[i], row=i+2, col=1)
        
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    # ==========================================
    # PASO 6: INYECCIÓN DE JAVASCRIPT
    # ==========================================
    js_code = """
    <script>
    function _tv() {
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if(!gd) return setTimeout(_tv, 200);
        var lbl = document.createElement('div');
        Object.assign(lbl.style, {
            position:'absolute', right:'45px', backgroundColor:'#2a2e39',
            color:'#d1d4dc', padding:'3px 6px', borderRadius:'3px',
            fontSize:'11px', zIndex:'1000', display:'none', pointerEvents:'none'
        });
        gd.appendChild(lbl);
        gd.addEventListener('mousemove', (e) => {
            var ya = gd._fullLayout.yaxis, box = gd.getBoundingClientRect(),
            yM = e.clientY - box.top, p = ya.p2c(yM - gd._fullLayout.margin.t);
            if (p >= ya.range[0] && p <= ya.range[1]) {
                lbl.style.top = (yM - 10) + 'px';
                lbl.innerHTML = p.toFixed(2);
                lbl.style.display = 'block';
            } else lbl.style.display = 'none';
        });
        gd.addEventListener('mouseleave', () => lbl.style.display = 'none');
    }
    setTimeout(_tv, 500);
    </script>
    """

    html_str = fig.to_html(
        include_plotlyjs=True,
        config={
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawrect', 'eraseshape'],
            'displaylogo': False
        }
    )
    
    components.html(html_str + js_code, height=870)
