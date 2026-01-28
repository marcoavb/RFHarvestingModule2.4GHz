import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# =========================
# CONFIG
# =========================
CFG = {
    "LINE_WIDTH": 2.4,
    "GRID_ALPHA_MAJOR": 0.35,
    "GRID_ALPHA_MINOR": 0.18,
    "SAVE_DPI": 1600,

    # Figsize para cada figura
    "FIGSIZE_PCE": (6.8, 5.2),
    "FIGSIZE_VOUT": (6.8, 5.2),

    # Nombres de salida (se guardan con tecla "s" en cada figura)
    "OUT_PCE": "Fig_PCE_compare.png",
    "OUT_VOUT": "Fig_Vout_compare.png",

    # Límites (ajusta si quieres)
    "PIN_XLIM": (-20, 10),     # dBm
    "PCE_YLIM": (0, 100),      # %
    "VOUT_YLIM": (0, 3.0),     # V

    # Si quieres forzar RL para convertir PCE_sim -> Vout_sim (ohms)
    # Si es None: se usa la mediana de RL del CSV (recomendado)
    "RL_FOR_VOUT_SIM": None,
}

plt.rcParams.update({
    # Match the screenshot style (Matplotlib default-like)
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],

    # Make math text match the same family
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",

    # Sizes closer to the screenshot (tweak if needed)
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,

    # Optional: improve look consistency
    "axes.titlesize": 14,
})

# =========================
# ARCHIVOS (si no están en la misma carpeta, usa ruta absoluta)
# =========================
FILES = {
    "sim_pce": "PCEFINAL.txt",
    "meas_csv": "Tesis RF Harvesting - Resultados Rectificador.csv",
}

# =========================
# GUARDAR con tecla "s"
# =========================
def enable_save_hotkey(fig, out_name: str):
    def on_key(event):
        if event.key and event.key.lower() == "s":
            fig.savefig(out_name, dpi=CFG["SAVE_DPI"], bbox_inches="tight")
            print(f"[OK] Guardado: {out_name} (dpi={CFG['SAVE_DPI']})")
    fig.canvas.mpl_connect("key_press_event", on_key)

def apply_paper_grid(ax, x_minor=4, y_minor=5):
    ax.grid(True, which="major", linestyle="--", alpha=CFG["GRID_ALPHA_MAJOR"])
    ax.grid(True, which="minor", linestyle=":",  alpha=CFG["GRID_ALPHA_MINOR"])
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))

# =========================
# Utils
# =========================
def dbm_to_w(dbm):
    return 1e-3 * (10.0 ** (dbm / 10.0))

def _clean_cols(cols):
    return [str(c).strip() for c in cols]

def _pick_col(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

# =========================
# LECTURA SIM: PCE (ASCII tipo ADS)
# Espera columnas: power  PCE
# - Si PCE viene 0..1, lo convierte a %
# =========================
def read_sim_pce_ascii(path: str):
    df = pd.read_csv(path, sep=r"\s+", engine="python", comment="!")
    df.columns = _clean_cols(df.columns)

    if "power" not in df.columns or "PCE" not in df.columns:
        raise ValueError(f"[SIM PCE] Columnas esperadas: 'power' y 'PCE'. Detectadas: {df.columns.tolist()}")

    pin_dbm = pd.to_numeric(df["power"], errors="coerce").to_numpy()
    pce = pd.to_numeric(df["PCE"], errors="coerce").to_numpy()

    m = np.isfinite(pin_dbm) & np.isfinite(pce)
    pin_dbm, pce = pin_dbm[m], pce[m]

    # Si está en 0..1 -> %
    if np.nanmax(pce) <= 1.2:
        pce = 100.0 * pce

    idx = np.argsort(pin_dbm)
    return pin_dbm[idx], pce[idx]

# =========================
# LECTURA MEAS: CSV
# Espera (ideal): PIN (dBm), PCE, Vout, RL
# =========================
def read_meas_csv(path: str):
    df = pd.read_csv(path)
    df.columns = _clean_cols(df.columns)

    pin_col = _pick_col(df, ["PIN (dBm)", "Pin (dBm)", "PIN_dBm", "Pin_dBm", "Pin", "PIN"])
    if pin_col is None:
        raise ValueError(f"[MEAS CSV] No encuentro columna de Pin en dBm. Columnas: {df.columns.tolist()}")

    pce_col = _pick_col(df, ["PCE", "pce", "PCE (%)", "PCE(%)"])
    vout_col = _pick_col(df, ["Vout", "Vout (V)", "V_out", "VOUT"])
    rl_col = _pick_col(df, ["RL", "R_L", "Load", "Rload", "R_load"])

    if pce_col is None:
        raise ValueError(f"[MEAS CSV] No encuentro columna PCE. Columnas: {df.columns.tolist()}")
    if vout_col is None:
        raise ValueError(f"[MEAS CSV] No encuentro columna Vout. Columnas: {df.columns.tolist()}")
    if rl_col is None:
        raise ValueError(f"[MEAS CSV] No encuentro columna RL. Columnas: {df.columns.tolist()}")

    pin_dbm = pd.to_numeric(df[pin_col], errors="coerce").to_numpy()
    pce = pd.to_numeric(df[pce_col], errors="coerce").to_numpy()
    vout = pd.to_numeric(df[vout_col], errors="coerce").to_numpy()
    rl = pd.to_numeric(df[rl_col], errors="coerce").to_numpy()

    m = np.isfinite(pin_dbm) & np.isfinite(pce) & np.isfinite(vout) & np.isfinite(rl)
    pin_dbm, pce, vout, rl = pin_dbm[m], pce[m], vout[m], rl[m]

    # Si PCE viene 0..1 -> %
    if np.nanmax(pce) <= 1.2:
        pce = 100.0 * pce

    idx = np.argsort(pin_dbm)
    return pin_dbm[idx], pce[idx], vout[idx], rl[idx]

# =========================
# CARGA DATOS
# =========================
pin_sim_dbm, pce_sim = read_sim_pce_ascii(FILES["sim_pce"])
pin_meas_dbm, pce_meas, vout_meas, rl_meas = read_meas_csv(FILES["meas_csv"])

# RL para convertir PCE_sim -> Vout_sim
RL_USED = CFG["RL_FOR_VOUT_SIM"]
if RL_USED is None:
    RL_USED = float(np.nanmedian(rl_meas)) if len(rl_meas) else 900.0

# Vout_sim (desde PCE_sim y RL):
# Pout = (PCE/100) * Pin(W),  Vout = sqrt(Pout * RL)
pin_sim_w = dbm_to_w(pin_sim_dbm)
pout_sim = (pce_sim / 100.0) * pin_sim_w
vout_sim = np.sqrt(np.maximum(pout_sim * RL_USED, 0.0))

# =========================
# FIGURA 1: PCE (SEPARADA)
# =========================
fig1, ax1 = plt.subplots(1, 1, figsize=CFG["FIGSIZE_PCE"])
enable_save_hotkey(fig1, CFG["OUT_PCE"])

ax1.plot(pin_meas_dbm, pce_meas, "-", lw=CFG["LINE_WIDTH"], label='meas. "PCE"')
ax1.plot(pin_sim_dbm,  pce_sim,  linestyle=(0, (4, 3)), lw=CFG["LINE_WIDTH"], label='sim. "PCE"')

ax1.set_xlim(*CFG["PIN_XLIM"])
ax1.set_ylim(*CFG["PCE_YLIM"])
ax1.set_xlabel(r"$P_{in}$ (dBm)")
ax1.set_ylabel("PCE (%)")
apply_paper_grid(ax1, x_minor=5, y_minor=5)

leg1 = ax1.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")
leg1.get_frame().set_alpha(1.0)
leg1.set_draggable(True)

fig1.tight_layout()

# =========================
# FIGURA 2: Vout (SEPARADA)
# =========================
fig2, ax2 = plt.subplots(1, 1, figsize=CFG["FIGSIZE_VOUT"])
enable_save_hotkey(fig2, CFG["OUT_VOUT"])

ax2.plot(pin_meas_dbm, vout_meas, "-", lw=CFG["LINE_WIDTH"], label='meas. "Vout"')
ax2.plot(pin_sim_dbm,  vout_sim,  linestyle=(0, (4, 3)), lw=CFG["LINE_WIDTH"],
         label=f'sim. "Vout" (desde PCE, $R_L$={RL_USED:.0f}$\\Omega$)')

ax2.set_xlim(*CFG["PIN_XLIM"])
ax2.set_ylim(*CFG["VOUT_YLIM"])
ax2.set_xlabel(r"$P_{in}$ (dBm)")
ax2.set_ylabel(r"$V_{out}$ (V)")
apply_paper_grid(ax2, x_minor=5, y_minor=5)

leg2 = ax2.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")
leg2.get_frame().set_alpha(1.0)
leg2.set_draggable(True)

fig2.tight_layout()

print("\nTIP: Arrastra las leyendas con el mouse.")
print("TIP: En la figura activa, presiona 's' para guardar en HD.")
plt.show()
