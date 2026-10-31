import os
import sys
import asyncio
import subprocess
import threading

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
from io import StringIO
from playwright.async_api import async_playwright
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(page_title="Market Breadth Analyzer (Barchart)", layout="wide")


def run_coro_in_thread(coro, timeout=300):
    """Ejecuta una corrutina en un hilo con su propio event loop.

    Necesario porque el hilo de Streamlit ya puede tener un loop en marcha y
    NO queremos tocar el loop del servidor (nest_asyncio lo rompe).
    """
    result = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"La descarga superó {timeout}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


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
@st.cache_resource(show_spinner="🔧 Preparando Chromium (solo la primera vez, ~1 min)...")
def ensure_chromium_installed():
    """Streamlit Cloud no trae el binario de Chromium: lo instalamos una vez por proceso."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            print(f"⚠️ playwright install falló: {proc.stderr[-2000:]}")
        return proc.returncode == 0
    except Exception as exc:
        print(f"⚠️ No se pudo preparar Chromium: {exc}")
        return False


def _parsear_respuesta(raw_text):
    """Convierte la respuesta CSV de Barchart en una Serie de cierres indexada por fecha."""
    text = raw_text.strip()
    if not text:
        return None

    primera = text.splitlines()[0].lower()
    tiene_cabecera = "date" in primera and ("close" in primera or "symbol" in primera)

    if tiene_cabecera:
        df = pd.read_csv(StringIO(text))
    else:
        df = pd.read_csv(
            StringIO(text),
            header=None,
            names=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"],
        )

    df.columns = [str(c).strip().strip('"') for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("timestamp")
    close_col = cols.get("close") or cols.get("last")
    if not date_col or not close_col:
        raise ValueError(f"Columnas no reconocidas: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    return pd.to_numeric(df[close_col], errors="coerce")


async def capturar_csv_barchart(simbolos):
    """Descarga los cierres de cada símbolo interceptando la respuesta 'queryeod' de Barchart."""
    resultados = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
        )
        try:
            for simbolo in simbolos:
                url = f"https://www.barchart.com/stocks/quotes/{simbolo}/interactive-chart"
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}, locale="en-US"
                )
                try:
                    page = await context.new_page()
                    loop = asyncio.get_running_loop()
                    raw_future = loop.create_future()

                    def _on_response(response):
                        if raw_future.done():
                            return
                        if "queryeod" in response.url and response.ok:
                            async def _store():
                                try:
                                    data = await response.text()
                                    if not raw_future.done():
                                        raw_future.set_result(data)
                                except Exception as exc:
                                    if not raw_future.done():
                                        raw_future.set_exception(exc)
                            asyncio.ensure_future(_store())

                    page.on("response", _on_response)

                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    except Exception as exc:
                        print(f"⚠️ Navegación {simbolo}: {exc}")

                    if not raw_future.done():
                        try:
                            await page.wait_for_load_state("networkidle", timeout=8000)
                        except Exception:
                            pass

                    try:
                        raw_text = await asyncio.wait_for(raw_future, timeout=20)
                    except Exception:
                        raw_text = None
                        print(f"⚠️ Timeout (20s) esperando respuesta de {simbolo}")

                    if raw_text:
                        try:
                            serie = _parsear_respuesta(raw_text)
                            if serie is not None and not serie.empty:
                                resultados[simbolo] = serie
                            else:
                                print(f"⚠️ Respuesta vacía para {simbolo}")
                        except Exception as exc:
                            print(f"⚠️ Error procesando {simbolo}: {exc}")
                finally:
                    await context.close()
        finally:
            await browser.close()
    return resultados


@st.cache_data(show_spinner="🕷️ Scraping Barchart (Primera vez puede tardar 10-20s)...", ttl=3600)
def get_barchart_data(ticker_selected):
    simbolos = BARCHART_SYMBOLS[ticker_selected]
    ensure_chromium_installed()
    cierres_dict = run_coro_in_thread(capturar_csv_barchart(simbolos), timeout=300)

    if cierres_dict:
        df = pd.concat(cierres_dict, axis=1)
        df.columns = [str(col).replace('$', '') for col in df.columns]
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
    rango = roll_max - roll_min
    SPADVDEC['FORMULA'] = ((SPADVDEC['LINADN'] - roll_min) / rango.mask(rango == 0)) * 100
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
try:
    stock = yf.download(ticker, start='2020-01-01', interval='1d', progress=False, auto_adjust=True)
except Exception as exc:
    st.error(f"No se pudo descargar el precio de {ticker}: {exc}")
    st.stop()

if stock is None or stock.empty:
    st.error(f"yfinance no devolvió datos para {ticker}. Recarga la página en unos segundos.")
    st.stop()

if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.get_level_values(0)

# 2. Datos de Barchart e Indicadores
df_barchart = get_barchart_data(ticker)
if df_barchart is None:
    st.error("No se pudieron descargar los datos de Barchart. Pulsa F5 o usa 'Clear cache' y recarga.")
    st.stop()

faltan = [s.replace('$', '') for s in BARCHART_SYMBOLS[ticker] if s.replace('$', '') not in df_barchart.columns]
if faltan:
    st.error(f"⚠️ **Fallo en la descarga:** no se obtuvieron {', '.join(faltan)}. Recarga la página (F5) o usa 'Clear cache'.")
    st.stop()

SPADVDEC = calculate_indicators(df_barchart, ticker)

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

