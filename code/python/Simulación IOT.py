import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# =========================
# CONFIG
# =========================
CFG = {
    "FILE": "rfharvesting_iot_uvlo.txt",  # <-- tu archivo

    "LINE_WIDTH": 2.4,
    "GRID_ALPHA_MAJOR": 0.35,
    "GRID_ALPHA_MINOR": 0.18,
    "SAVE_DPI": 1600,

    # Figsize por figura (igual estilo “paper”)
    "FIGSIZE_V": (6.8, 5.2),
    "FIGSIZE_I": (6.8, 5.2),

    # Guardado con tecla "s"
    "OUT_V": "Fig_UVLO_Voltajes.png",
    "OUT_I": "Fig_UVLO_Corrientes.png",

    # Escala de tiempo en el eje X (elige: "s", "ms", "us")
    "TIME_UNIT": "s",

    # Límites (pon None para auto)
    "XLIM": None,           # e.g. (0, 200) en segundos (ya escalados)
    "V_YLIM": None,         # e.g. (0, 3.0)
    "I_YLIM": None,         # e.g. (0, 25) en mA si I_UNIT="mA"

    # Corriente en el eje Y (elige: "A", "mA", "uA")
    "I_UNIT": "mA",
}

# =========================
# FONT / STYLE (como el plot ejemplo: sans-serif)
# =========================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",

    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

# =========================
# UTILIDADES
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

def time_scale_factor(unit: str) -> float:
    u = unit.lower().strip()
    if u == "s":
        return 1.0
    if u == "ms":
        return 1e3
    if u == "us":
        return 1e6
    raise ValueError("TIME_UNIT debe ser: 's', 'ms' o 'us'")

def current_scale_factor(unit: str) -> float:
    u = unit.lower().strip()
    if u == "a":
        return 1.0
    if u == "ma":
        return 1e3
    if u == "ua":
        return 1e6
    raise ValueError("I_UNIT debe ser: 'A', 'mA' o 'uA'")

# =========================
# LECTURA DEL ARCHIVO
# (tu header trae paréntesis: V(ctrl), etc.)
# =========================
def read_uvlo_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    required = ["time", "V(ctrl)", "V(ncap)", "I(Bburst)", "I(Bchg)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en {path}: {missing}\nEncontré: {list(df.columns)}")

    # Coerce numérico
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).copy()

    # Ordena por tiempo
    df = df.sort_values("time")
    return df

def detect_low_to_high_then_high_to_low(t, vctrl, vcap, frac_low=0.2, frac_high=0.8):
    """
    Detecta:
      - primer flanco de subida (bajo->alto) con histéresis
      - siguiente flanco de bajada (alto->bajo)
    Devuelve:
      t_rise, t_fall, vcap_at_rise, vcap_at_fall, thresholds (th_low, th_high, th_mid)
    """
    t = np.asarray(t, float)
    vctrl = np.asarray(vctrl, float)
    vcap = np.asarray(vcap, float)

    m = np.isfinite(t) & np.isfinite(vctrl) & np.isfinite(vcap)
    t, vctrl, vcap = t[m], vctrl[m], vcap[m]

    if t.size < 3:
        raise ValueError("No hay suficientes datos para detectar transiciones.")

    vmin, vmax = np.nanmin(vctrl), np.nanmax(vctrl)
    dv = vmax - vmin
    if dv <= 0:
        raise ValueError("Vcontrol no varía; no se pueden detectar flancos.")

    th_low  = vmin + frac_low  * dv
    th_high = vmin + frac_high * dv
    th_mid  = vmin + 0.5 * dv

    # Estados con histéresis: 0=LOW, 1=HIGH
    state = 1 if vctrl[0] >= th_high else 0
    t_rise = None
    t_fall = None
    idx_rise = None
    idx_fall = None

    # 1) Buscar rising: LOW -> HIGH
    for i in range(1, len(t)):
        if state == 0:
            if vctrl[i-1] < th_high and vctrl[i] >= th_high:
                t_rise = t[i]
                idx_rise = i
                state = 1
                break
        else:
            if vctrl[i] <= th_low:
                state = 0

    if t_rise is None:
        raise ValueError("No se encontró transición bajo->alto en Vcontrol.")

    # 2) Buscar falling después del rising: HIGH -> LOW
    for i in range(idx_rise + 1, len(t)):
        if state == 1:
            if vctrl[i-1] > th_low and vctrl[i] <= th_low:
                t_fall = t[i]
                idx_fall = i
                state = 0
                break
        else:
            if vctrl[i] >= th_high:
                state = 1

    if t_fall is None:
        raise ValueError("No se encontró el retorno (alto->bajo) luego del flanco de subida.")

    vcap_at_rise = vcap[idx_rise]
    vcap_at_fall = vcap[idx_fall]

    return t_rise, t_fall, vcap_at_rise, vcap_at_fall, (th_low, th_high, th_mid)


df = read_uvlo_file(CFG["FILE"])

# Escalas
t_factor = time_scale_factor(CFG["TIME_UNIT"])
i_factor = current_scale_factor(CFG["I_UNIT"])

t = df["time"].to_numpy() * t_factor
v_ctrl = df["V(ctrl)"].to_numpy()
v_cap  = df["V(ncap)"].to_numpy()
i_burst = df["I(Bburst)"].to_numpy() * i_factor
i_chg   = df["I(Bchg)"].to_numpy() * i_factor

# =========================
# FIGURA 1: VOLTAJES
# =========================
figV, axV = plt.subplots(1, 1, figsize=CFG["FIGSIZE_V"])
enable_save_hotkey(figV, CFG["OUT_V"])

axV.plot(t, v_cap,  "-",  lw=CFG["LINE_WIDTH"], label=r"$V_{cap}$")
axV.plot(t, v_ctrl, linestyle=(0, (4, 3)), lw=CFG["LINE_WIDTH"], label=r"$V_{ctrl}$")

# =========================
# ANOTACIÓN: intervalo Vctrl bajo->alto y retorno
# =========================
try:
    t_rise, t_fall, vcap_i, vcap_f, (th_low, th_high, th_mid) = detect_low_to_high_then_high_to_low(
        t, v_ctrl, v_cap, frac_low=0.2, frac_high=0.8
    )

    dt = t_fall - t_rise

    # Líneas verticales + sombreado del intervalo
    axV.axvline(t_rise, linestyle="--", linewidth=1.4)
    axV.axvline(t_fall, linestyle="--", linewidth=1.4)
    axV.axvspan(t_rise, t_fall, alpha=0.12)

    # Punto de referencia para la flecha (en la curva Vctrl)
    y_point = np.interp(t_rise, t, v_ctrl)

    # Texto estilo “caja” como tu ejemplo
    txt = (
        "Cambio Vcontrol (LOW→HIGH)\n"
        f"t_rise = {t_rise:.6g} {CFG['TIME_UNIT']}\n"
        f"t_fall = {t_fall:.6g} {CFG['TIME_UNIT']}\n"
        f"Δt (retorno) = {dt:.6g} {CFG['TIME_UNIT']}\n"
        f"Vcap inicial = {vcap_i:.4g} V\n"
        f"Vcap final   = {vcap_f:.4g} V"
    )

    ann = axV.annotate(
        txt,
        xy=(t_rise, y_point),
        xytext=(0.05, 0.95),            # esquina superior izquierda del axes
        textcoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="#4F94CD", ec="#2F5E8E", alpha=0.95),
        arrowprops=dict(arrowstyle="->", lw=1.6),
    )

    # ✅ Arrastrable (draggable) como pediste
    ann.draggable(True)

except Exception as e:
    print(f"[WARN] No se pudo crear la anotación de transición Vcontrol: {e}")


axV.set_xlabel(f"Tiempo ({CFG['TIME_UNIT']})")
axV.set_ylabel("Voltaje (V)")
apply_paper_grid(axV, x_minor=4, y_minor=5)

if CFG["XLIM"] is not None:
    axV.set_xlim(*CFG["XLIM"])
if CFG["V_YLIM"] is not None:
    axV.set_ylim(*CFG["V_YLIM"])

legV = axV.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
legV.get_frame().set_alpha(1.0)
legV.set_draggable(True)

figV.tight_layout()

# =========================
# FIGURA 2: CORRIENTES
# =========================
figI, axI = plt.subplots(1, 1, figsize=CFG["FIGSIZE_I"])
enable_save_hotkey(figI, CFG["OUT_I"])

axI.plot(t, i_chg,   "-",  lw=CFG["LINE_WIDTH"], label=r"$I_{chg}$")
axI.plot(t, i_burst, linestyle=(0, (1, 1)), lw=CFG["LINE_WIDTH"], label=r"$I_{burst}$")

axI.set_xlabel(f"Tiempo ({CFG['TIME_UNIT']})")
axI.set_ylabel(f"Corriente ({CFG['I_UNIT']})")
apply_paper_grid(axI, x_minor=4, y_minor=5)

if CFG["XLIM"] is not None:
    axI.set_xlim(*CFG["XLIM"])
if CFG["I_YLIM"] is not None:
    axI.set_ylim(*CFG["I_YLIM"])

legI = axI.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
legI.get_frame().set_alpha(1.0)
legI.set_draggable(True)

figI.tight_layout()

print("\nTIP: Arrastra las leyendas con el mouse.")
print("TIP: En la figura activa, presiona 's' para guardar en HD.")
plt.show()
