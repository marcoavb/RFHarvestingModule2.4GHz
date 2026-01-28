import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (edita aquí)
# =========================
CFG = {
    "FONT_SIZE": 18,

    "S11_COLOR": "tab:orange",
    "S21_COLOR": "tab:blue",

    "GRID_ALPHA": 0.30,
    "LINE_WIDTH": 2.0,

    # Criterio BW típico en matching
    "S11_THRESHOLD_DB": -10.0,

    # Sombreado BW suave
    "BW_SHADE_ALPHA": 0.15,

    # Caja anotación
    "BOX_FACE": "#418BBF",
    "BOX_ALPHA": 0.95,
    "BOX_EDGE": "#3B6EA8",
    "TEXT_COLOR": "black",

    # Guardado HD
    "SAVE_DPI": 1600,
}

# === Archivos (ASCII exportados desde ADS) ===
S11_FILE = "Matching Network Stub Radial_S11.txt"  # <-- tu S11
S21_FILE = "Matching Network Stub Radial_S21.txt"   # <-- tu S21 (pon el nombre real)

# Salidas (tecla 's')
OUT_S11 = "S11_BW_highres.png"
OUT_S21 = "S21_highres.png"

plt.rcParams.update({
    "font.size": CFG["FONT_SIZE"],
    "axes.titlesize": CFG["FONT_SIZE"] + 2,
    "axes.labelsize": CFG["FONT_SIZE"],
})

# =========================
# Texto movible (drag & drop)
# =========================
def make_draggable(artist):
    state = {"press": None}

    def on_press(event):
        if event.inaxes != artist.axes:
            return
        contains, _ = artist.contains(event)
        if not contains:
            return
        x0, y0 = artist.get_position()
        state["press"] = (x0, y0, event.xdata, event.ydata)

    def on_release(event):
        state["press"] = None
        artist.figure.canvas.draw_idle()

    def on_motion(event):
        if state["press"] is None or event.inaxes != artist.axes:
            return
        x0, y0, xpress, ypress = state["press"]
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        artist.set_position((x0 + dx, y0 + dy))
        artist.figure.canvas.draw_idle()

    fig = artist.figure
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

# =========================
# Guardado HD con tecla "s"
# =========================
def enable_save_hotkey(fig, out_name: str):
    def on_key(event):
        if event.key and event.key.lower() == "s":
            fig.savefig(out_name, dpi=CFG["SAVE_DPI"], bbox_inches="tight")
            print(f"[OK] Guardado: {out_name} (dpi={CFG['SAVE_DPI']})")
    fig.canvas.mpl_connect("key_press_event", on_key)

# =========================
# Buscar columnas automáticamente
# =========================
def find_col(cols, candidates):
    cols = [c.strip() for c in cols]
    lower = [c.lower() for c in cols]

    # match exacto primero
    for cand in candidates:
        cl = cand.lower()
        for i, c in enumerate(lower):
            if c == cl:
                return cols[i]

    # match por contención
    for cand in candidates:
        cl = cand.lower()
        for i, c in enumerate(lower):
            if cl in c:
                return cols[i]
    return None

# =========================
# Parsear frecuencia a Hz (robusto)
# =========================
def parse_frequency_to_hz(series: pd.Series) -> np.ndarray:
    """
    Convierte freq que puede venir como:
      - 1.5000E9 (Hz)
      - 2.45e9
      - 2.45GHz, 2450MHz, etc.
    Devuelve numpy array en Hz (float).
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)

    # Normaliza unidades comunes
    s = s.str.replace("GHz", "e9", regex=False)
    s = s.str.replace("ghz", "e9", regex=False)
    s = s.str.replace("MHz", "e6", regex=False)
    s = s.str.replace("mhz", "e6", regex=False)
    s = s.str.replace("kHz", "e3", regex=False)
    s = s.str.replace("khz", "e3", regex=False)
    s = s.str.replace("Hz", "", regex=False)
    s = s.str.replace("hz", "", regex=False)

    f = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return f

# =========================
# Leer archivo S-parameter (freq + dB(Sij))
# =========================
def read_sparam_file(path: str, y_candidates: list[str]):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    freq_col = find_col(df.columns, ["freq", "fr", "frequency", "f"])
    y_col = find_col(df.columns, y_candidates)

    if freq_col is None:
        raise ValueError(f"[{path}] No se encontró columna de frecuencia (freq/fr/frequency). Columnas: {df.columns.tolist()}")
    if y_col is None:
        raise ValueError(f"[{path}] No se encontró columna {y_candidates}. Columnas: {df.columns.tolist()}")

    f_hz = parse_frequency_to_hz(df[freq_col])
    y_db = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    valid = ~np.isnan(f_hz) & ~np.isnan(y_db)
    f_hz = f_hz[valid]
    y_db = y_db[valid]

    # Heurística: si viene en GHz (valores pequeños), pasarlo a Hz
    # (Tu caso típico ADS: Hz ~ 1e9)
    if np.nanmax(f_hz) < 1e6:
        # probablemente está en GHz
        f_hz = f_hz * 1e9

    return f_hz, y_db, df.columns.tolist(), freq_col, y_col

# =========================
# BW con interpolación lineal (S11 <= umbral)
# =========================
def bandwidth_from_threshold(f_ghz: np.ndarray, s11_db: np.ndarray, thr_db: float):
    f = np.asarray(f_ghz, dtype=float)
    y = np.asarray(s11_db, dtype=float)

    order = np.argsort(f)
    f = f[order]
    y = y[order]

    mask = y <= thr_db
    if not np.any(mask):
        return None

    idx = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    segments_idx = np.split(idx, splits + 1)

    segments = []
    for seg in segments_idx:
        i0 = int(seg[0])
        i1 = int(seg[-1])

        f_low = f[i0]
        if i0 > 0 and (y[i0 - 1] > thr_db) and (y[i0] <= thr_db):
            y1, y2 = y[i0 - 1], y[i0]
            f1, f2 = f[i0 - 1], f[i0]
            t = (thr_db - y1) / (y2 - y1) if (y2 - y1) != 0 else 0.0
            f_low = f1 + (f2 - f1) * t

        f_high = f[i1]
        if i1 < len(f) - 1 and (y[i1] <= thr_db) and (y[i1 + 1] > thr_db):
            y1, y2 = y[i1], y[i1 + 1]
            f1, f2 = f[i1], f[i1 + 1]
            t = (thr_db - y1) / (y2 - y1) if (y2 - y1) != 0 else 0.0
            f_high = f1 + (f2 - f1) * t

        segments.append((float(f_low), float(f_high), float(f_high - f_low)))

    # banda más ancha
    segments.sort(key=lambda x: x[2], reverse=True)
    return segments[0]

# =========================
# ======== MAIN ===========
# =========================

# ---- Leer S11 ----
f_hz_s11, s11_db, cols_s11, fc_s11, yc_s11 = read_sparam_file(
    S11_FILE,
    y_candidates=["dB(S(1,1))", "db(s(1,1))", "S11", "dB(S11)"]
)
f_ghz_s11 = f_hz_s11 / 1e9

print("\n=== S11: Columnas detectadas ===")
print(cols_s11)
print(f"Usando: freq='{fc_s11}', y='{yc_s11}'")

# Métricas S11
i_s11_min = int(np.argmin(s11_db))
s11_min = float(s11_db[i_s11_min])
f_s11_min = float(f_ghz_s11[i_s11_min])

thr = float(CFG["S11_THRESHOLD_DB"])
bw_info = bandwidth_from_threshold(f_ghz_s11, s11_db, thr)

print("\n=== RESULTADOS S11 ===")
print(f"Mínimo S11 = {s11_min:.2f} dB en f = {f_s11_min:.6f} GHz")
if bw_info is None:
    print(f"No hay BW: no se cumple S11 <= {thr:.2f} dB")
else:
    fL, fH, BW = bw_info
    print(f"BW (S11 <= {thr:.2f} dB) = {BW*1000:.2f} MHz  |  [{fL:.6f}, {fH:.6f}] GHz")

# ---- Leer S21 ----
f_hz_s21, s21_db, cols_s21, fc_s21, yc_s21 = read_sparam_file(
    S21_FILE,
    y_candidates=["dB(S(2,1))", "db(s(2,1))", "S21", "dB(S21)", "dB(S(1,2))", "S12", "dB(S12)"]
)
f_ghz_s21 = f_hz_s21 / 1e9

print("\n=== S21: Columnas detectadas ===")
print(cols_s21)
print(f"Usando: freq='{fc_s21}', y='{yc_s21}'")

# Métricas S21 (máximo)
i_s21_max = int(np.argmax(s21_db))
s21_max = float(s21_db[i_s21_max])
f_s21_max = float(f_ghz_s21[i_s21_max])

print("\n=== RESULTADOS S21 ===")
print(f"Máximo S21 = {s21_max:.2f} dB en f = {f_s21_max:.6f} GHz")

# =========================
# FIGURA 1: S11 + BW sombreado
# =========================
fig1, ax1 = plt.subplots(figsize=(8.2, 4.8))
enable_save_hotkey(fig1, OUT_S11)

ax1.plot(f_ghz_s11, s11_db, lw=CFG["LINE_WIDTH"], color=CFG["S11_COLOR"])
ax1.set_xlabel("Frecuencia (GHz)")
ax1.set_ylabel(r"$S_{11}$ (dB)")
ax1.grid(True, alpha=CFG["GRID_ALPHA"])

# Umbral BW
ax1.axhline(thr, ls="--", lw=1.2)

# Marcador mínimo
ax1.axvline(f_s11_min, ls="--", lw=1.4)
ax1.scatter([f_s11_min], [s11_min], s=35)

# Sombreado BW suave
if bw_info is not None:
    fL, fH, BW = bw_info
    ax1.axvspan(fL, fH, alpha=CFG["BW_SHADE_ALPHA"])

# Texto movible
if bw_info is None:
    txt_s11 = (
        f"Mínimo S11\n"
        f"f = {f_s11_min:.4f} GHz\n"
        f"S11 = {s11_min:.2f} dB\n\n"
        f"BW: no cumple S11≤{thr:.0f} dB"
    )
else:
    txt_s11 = (
        f"Mínimo S11\n"
        f"f = {f_s11_min:.4f} GHz\n"
        f"S11 = {s11_min:.2f} dB\n\n"
        f"BW (S11≤{thr:.0f} dB)\n"
        f"{fL:.4f}–{fH:.4f} GHz\n"
        f"BW = {BW*1000:.1f} MHz"
    )

ann1 = ax1.annotate(
    txt_s11,
    xy=(f_s11_min, s11_min),
    xytext=(f_s11_min + 0.10, s11_min + 10),
    bbox=dict(boxstyle="round,pad=0.35", fc=CFG["BOX_FACE"], ec=CFG["BOX_EDGE"], alpha=CFG["BOX_ALPHA"]),
    arrowprops=dict(arrowstyle="->", lw=1.2),
    color=CFG["TEXT_COLOR"]
)
make_draggable(ann1)

# =========================
# FIGURA 2: S21 + máximo marcado
# =========================
fig2, ax2 = plt.subplots(figsize=(8.2, 4.8))
enable_save_hotkey(fig2, OUT_S21)

ax2.plot(f_ghz_s21, s21_db, lw=CFG["LINE_WIDTH"], color=CFG["S21_COLOR"])
ax2.set_xlabel("Frecuencia (GHz)")
ax2.set_ylabel(r"$S_{21}$ (dB)")
ax2.grid(True, alpha=CFG["GRID_ALPHA"])

# Marcador máximo
ax2.axvline(f_s21_max, ls="--", lw=1.4)
ax2.scatter([f_s21_max], [s21_max], s=35)

txt_s21 = (
    f"Máximo S21\n"
    f"f = {f_s21_max:.4f} GHz\n"
    f"S21 = {s21_max:.2f} dB"
)

ann2 = ax2.annotate(
    txt_s21,
    xy=(f_s21_max, s21_max),
    xytext=(f_s21_max + 0.10, s21_max - 5),
    bbox=dict(boxstyle="round,pad=0.35", fc=CFG["BOX_FACE"], ec=CFG["BOX_EDGE"], alpha=CFG["BOX_ALPHA"]),
    arrowprops=dict(arrowstyle="->", lw=1.2),
    color=CFG["TEXT_COLOR"]
)
make_draggable(ann2)

print("\nTIP: Mueve los textos con el mouse. Cuando quede como quieres,")
print("     haz click en la ventana y presiona 's' para guardar en HD.")
plt.show()
