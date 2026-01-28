import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# CONFIG (edita aquí)
# =========================
CFG = {
    "FONT_SIZE": 18,

    # Estilo general
    "GRID_ALPHA": 0.30,
    "LINE_WIDTH": 2.4,

    # Dispersión (mediciones individuales)
    "SCATTER_ALPHA": 0.35,
    "SCATTER_SIZE": 22,

    # Banda de IC
    "CI_ALPHA": 0.22,

    # Caja de anotación (recomiendo clara para texto negro)
    "BOX_FACE": "#418BBF",
    "BOX_ALPHA": 0.95,
    "BOX_EDGE": "#3B6EA8",

    # Texto SIEMPRE negro
    "TEXT_COLOR": "black",

    # Guardado en alta definición
    "SAVE_DPI": 1600,

    # Tamaño de figura
    "FIGSIZE": (9, 6),

    # Nivel de confianza
    "CONF_LEVEL": 0.95,
}

# =========================
# ARCHIVO
# =========================
CSV_FILE = "Tesis RF Harvesting - Potencia Antena Turnstile.csv"

# Nombres de salida (se guardan al presionar "s")
OUT_PLOT = "Distancia_vs_Prx_IC95.png"

# Columnas esperadas (del CSV)
DIST_COL = "Distancia"
PRX_MAIN_COL = "Potencia de recepción (dBm)"

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
    """
    Presiona 's' con la ventana activa para guardar en alta definición.
    (Esto sirve justo DESPUÉS de que muevas los textos.)
    """
    def on_key(event):
        if event.key and event.key.lower() == "s":
            fig.savefig(out_name, dpi=CFG["SAVE_DPI"], bbox_inches="tight")
            print(f"[OK] Guardado: {out_name} (dpi={CFG['SAVE_DPI']})")

    fig.canvas.mpl_connect("key_press_event", on_key)

# =========================
# LECTURA CSV + PREPROCESO
# =========================
df = pd.read_csv(CSV_FILE)

# Detecta columnas de mediciones repetidas (en tu archivo vienen como "Unnamed: x")
unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]

# Lista total de columnas Prx (principal + repetidas)
prx_cols = [PRX_MAIN_COL] + unnamed_cols

# Convierte a numérico (por si hay vacíos o texto)
for c in [DIST_COL] + prx_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Pasa a formato largo: una fila por medición
long_df = df[[DIST_COL] + prx_cols].melt(
    id_vars=DIST_COL,
    value_vars=prx_cols,
    var_name="Medicion",
    value_name="Prx_dBm"
).dropna()

# =========================
# RESUMEN + IC (t-student)
# =========================
summary = long_df.groupby(DIST_COL)["Prx_dBm"].agg(["count", "mean", "std"]).reset_index()
summary["sem"] = summary["std"] / np.sqrt(summary["count"])

alpha = 1.0 - CFG["CONF_LEVEL"]

# t crítico por cada distancia (df = n-1)
summary["t_crit"] = summary.apply(
    lambda r: stats.t.ppf(1 - alpha/2, df=int(r["count"] - 1)) if r["count"] > 1 else np.nan,
    axis=1
)

summary["ci"] = summary["t_crit"] * summary["sem"]
summary["lower"] = summary["mean"] - summary["ci"]
summary["upper"] = summary["mean"] + summary["ci"]

# Arrays numéricos (evita problemas con fill_between)
x = summary[DIST_COL].to_numpy(dtype=float)
y = summary["mean"].to_numpy(dtype=float)
lower = summary["lower"].to_numpy(dtype=float)
upper = summary["upper"].to_numpy(dtype=float)

# =========================
# FIGURA: Distancia vs Prx con IC
# =========================
fig, ax = plt.subplots(figsize=CFG["FIGSIZE"])
enable_save_hotkey(fig, OUT_PLOT)

# Mediciones individuales
ax.scatter(
    long_df[DIST_COL].to_numpy(dtype=float),
    long_df["Prx_dBm"].to_numpy(dtype=float),
    s=CFG["SCATTER_SIZE"],
    alpha=CFG["SCATTER_ALPHA"],
    label="Mediciones individuales"
)

# Media
ax.plot(
    x, y,
    lw=CFG["LINE_WIDTH"],
    marker="o",
    label="Media"
)

# Banda IC
ax.fill_between(
    x, lower, upper,
    alpha=CFG["CI_ALPHA"],
    label=f"IC {int(CFG['CONF_LEVEL']*100)}% (t-student)"
)

ax.set_xlabel("Distancia (m)")
ax.set_ylabel("Potencia de recepción (dBm)")
ax.grid(True, alpha=CFG["GRID_ALPHA"])
ax.legend()

# =========================
# Anotación movible (resumen)
# =========================
# Toma la última distancia (máxima) para mostrar un resumen en la caja
i_last = int(np.argmax(x))
d_last = float(x[i_last])
mean_last = float(y[i_last])
ci_last = float(summary.loc[i_last, "ci"])
n_last = int(summary.loc[i_last, "count"])



# =========================
# Mostrar
# =========================
print("=== INFO ===")
print(f"Filas en long_df (mediciones totales): {len(long_df)}")
print("TIP: Mueve el cuadro con el mouse. Cuando quede como quieres,")
print("     haz click en la ventana y presiona 's' para guardar en HD.")
plt.tight_layout()
plt.show()

# (Opcional) imprime tabla resumen
print("\n=== RESUMEN POR DISTANCIA ===")
print(summary[[DIST_COL, "count", "mean", "std", "ci", "lower", "upper"]])
