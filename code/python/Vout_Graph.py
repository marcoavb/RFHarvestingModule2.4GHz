import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (edita aquí)
# =========================
CFG = {
    "FONT_SIZE": 18,

    # Color de curva Vout
    "VOUT_COLOR": "tab:orange",

    # Estilo general
    "GRID_ALPHA": 0.30,
    "LINE_WIDTH": 2.0,

    # Caja de anotación
    "BOX_FACE": "#418BBF",
    "BOX_ALPHA": 0.95,
    "BOX_EDGE": "#3B6EA8",

    # Texto negro
    "TEXT_COLOR": "black",

    # Guardado HD
    "SAVE_DPI": 1600,
}

# =========================
# MARCADOR EN dBm DEFINIDO POR TI
# =========================
PIN_MARK_DBM = 5  # <-- pon aquí el dBm que quieras marcar (ej: -10, 0, 5, etc.)

# =========================
# ARCHIVO ASCII exportado desde ADS
# =========================
VOUT_FILE = "VOUT.txt"          # <-- tu archivo
OUT_VOUT  = "VOUT_highres.png"  # salida HD (presionando "s")

# Columnas REALES de tu ASCII (según tu archivo)
PIN_COL  = "power"
VOUT_COL = "real(Vout[::,0])"

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
# LECTURA ASCII (tab/espacios)
# =========================
df = pd.read_csv(VOUT_FILE, sep=r"\s+", engine="python")
df.columns = [c.strip() for c in df.columns]

print("=== Columnas detectadas ===")
print(list(df.columns))

if PIN_COL not in df.columns:
    raise ValueError(f"No existe la columna '{PIN_COL}'. Revisa el nombre real en: {df.columns.tolist()}")
if VOUT_COL not in df.columns:
    raise ValueError(f"No existe la columna '{VOUT_COL}'. Revisa el nombre real en: {df.columns.tolist()}")

pin_dbm = pd.to_numeric(df[PIN_COL], errors="coerce").to_numpy()
vout    = pd.to_numeric(df[VOUT_COL], errors="coerce").to_numpy()

mask = np.isfinite(pin_dbm) & np.isfinite(vout)
pin_dbm = pin_dbm[mask]
vout    = vout[mask]

if len(pin_dbm) == 0:
    raise ValueError("No hay datos numéricos válidos tras limpiar NaNs.")

# =========================
# BUSCAR EL PUNTO MÁS CERCANO AL dBm QUE DEFINISTE
# =========================
idx = int(np.argmin(np.abs(pin_dbm - PIN_MARK_DBM)))
pin_mark  = float(pin_dbm[idx])
vout_mark = float(vout[idx])

print("=== MARCADOR SELECCIONADO ===")
print(f"Pin objetivo: {PIN_MARK_DBM:.2f} dBm")
print(f"Pin encontrado (más cercano): {pin_mark:.2f} dBm")
print(f"Vout en ese punto: {vout_mark:.6f} V")

# =========================
# FIGURA: Vout vs Pin
# =========================
fig, ax = plt.subplots(figsize=(7.6, 4.4))
enable_save_hotkey(fig, OUT_VOUT)

ax.plot(pin_dbm, vout, lw=CFG["LINE_WIDTH"], color=CFG["VOUT_COLOR"])
ax.set_xlabel("Potencia de entrada, Pin (dBm)")
ax.set_ylabel("Voltaje de salida, Vout (V)")
ax.grid(True, alpha=CFG["GRID_ALPHA"])

# Marcador vertical en el dBm definido (o el más cercano)
ax.axvline(pin_mark, ls="--", lw=1.4)
ax.scatter([pin_mark], [vout_mark], s=35)

# Anotación movible
txt = (
    "Pin = {pin_mark:.2f} dBm\n"
    f"Vout = {vout_mark:.4f} V"
)

ann = ax.annotate(
    txt,
    xy=(pin_mark, vout_mark),
    xytext=(pin_mark + 2, vout_mark),
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc=CFG["BOX_FACE"],
        ec=CFG["BOX_EDGE"],
        alpha=CFG["BOX_ALPHA"]
    ),
    arrowprops=dict(arrowstyle="->", lw=1.2),
    color=CFG["TEXT_COLOR"]
)
make_draggable(ann)

print("\nTIP: Mueve el texto con el mouse. Cuando quede como quieres,")
print("     haz click en la ventana y presiona 's' para guardar en HD.")
plt.show()
