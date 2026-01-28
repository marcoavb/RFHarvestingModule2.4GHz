import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (edita aquí)
# =========================
CFG = {
    "FONT_SIZE": 18,

    # Colores de curvas
    "S11_COLOR": "tab:orange",
    "PCE_COLOR": "tab:blue",

    # Estilo general
    "GRID_ALPHA": 0.30,
    "LINE_WIDTH": 2.0,

    # Caja de anotación (recomiendo clara para texto negro)
    "BOX_FACE": "#418BBF",
    "BOX_ALPHA": 0.95,
    "BOX_EDGE": "#3B6EA8",

    # Texto SIEMPRE negro (como pediste)
    "TEXT_COLOR": "black",

    # Guardado en alta definición
    "SAVE_DPI": 1600,
}

# Potencia a la que simulaste S11 (si quieres mostrarla en la caja). Si no, pon None.
PIN_S11_DBM = 0  # ejemplo: 0 dBm

# Archivos ASCII (exportados desde ADS)
PCE_FILE = "PCE_Rectificador.txt"
S11_FILE = "S11_Rectificador.txt"

# Nombres de salida (se guardan al presionar "s" en cada figura)
OUT_S11 = "S11_highres.png"
OUT_PCE = "PCE_highres.png"

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
        if event.key.lower() == "s":
            fig.savefig(out_name, dpi=CFG["SAVE_DPI"], bbox_inches="tight")
            print(f"[OK] Guardado: {out_name} (dpi={CFG['SAVE_DPI']})")

    fig.canvas.mpl_connect("key_press_event", on_key)

# =========================
# LECTURA ASCII (tab/espacios)
# =========================
df_pce = pd.read_csv(PCE_FILE, sep=r"\s+", engine="python")
df_s11 = pd.read_csv(S11_FILE, sep=r"\s+", engine="python")

df_pce.columns = [c.strip() for c in df_pce.columns]
df_s11.columns = [c.strip() for c in df_s11.columns]

# Extrae datos
pin_dbm = df_pce["power"].to_numpy()
pce = df_pce["PCE"].to_numpy()

f_hz = df_s11["fr"].to_numpy()
s11_db = df_s11["dB(S(1,1))"].to_numpy()
f_ghz = f_hz / 1e9

# =========================
# MÉTRICAS: máximo PCE y mínimo S11
# =========================
i_pce_max = int(np.argmax(pce))
pce_max = float(pce[i_pce_max])
pin_at_pce_max = float(pin_dbm[i_pce_max])

i_s11_min = int(np.argmin(s11_db))  # más negativo
s11_min = float(s11_db[i_s11_min])
f_at_s11_min = float(f_ghz[i_s11_min])

print("=== RESULTADOS ===")
print(f"Máximo PCE = {pce_max:.3f}   en Pin = {pin_at_pce_max:.2f} dBm")
print(f"Mínimo S11 = {s11_min:.2f} dB en f = {f_at_s11_min:.6f} GHz")
if PIN_S11_DBM is not None:
    print(f"(S11 simulado a Pin = {PIN_S11_DBM:.2f} dBm)")

# =========================
# FIGURA 1: S11 (independiente)
# =========================
fig1, ax1 = plt.subplots(figsize=(9, 6))
enable_save_hotkey(fig1, OUT_S11)

ax1.plot(f_ghz, s11_db, lw=CFG["LINE_WIDTH"], color=CFG["S11_COLOR"])
ax1.set_xlabel("Frecuencia (GHz)")
ax1.set_ylabel(r"$S_{11}$ (dB)")

ax1.grid(True, alpha=CFG["GRID_ALPHA"])

# Marcador VERTICAL en el mínimo
ax1.axvline(f_at_s11_min, ls="--", lw=1.4)

# Punto mínimo
ax1.scatter([f_at_s11_min], [s11_min], s=35)

# Anotación movible (texto negro)
txt_s11 = f"Mínimo S11\nf = {f_at_s11_min:.4f} GHz\nS11 = {s11_min:.2f} dB"


ann1 = ax1.annotate(
    txt_s11,
    xy=(f_at_s11_min, s11_min),
    xytext=(f_at_s11_min + 0.10, s11_min + 10),
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc=CFG["BOX_FACE"],
        ec=CFG["BOX_EDGE"],
        alpha=CFG["BOX_ALPHA"]
    ),
    arrowprops=dict(arrowstyle="->", lw=1.2),
    color=CFG["TEXT_COLOR"]
)
make_draggable(ann1)

# =========================
# FIGURA 2: PCE (independiente)
# =========================
fig2, ax2 = plt.subplots(figsize=(9, 6))
enable_save_hotkey(fig2, OUT_PCE)

ax2.plot(pin_dbm, pce, lw=CFG["LINE_WIDTH"], color=CFG["PCE_COLOR"])
ax2.set_xlabel("Potencia de entrada, Pin (dBm)")
ax2.set_ylabel("PCE")

ax2.grid(True, alpha=CFG["GRID_ALPHA"])

# Marcador VERTICAL en el máximo
ax2.axvline(pin_at_pce_max, ls="--", lw=1.4)

# Punto máximo
ax2.scatter([pin_at_pce_max], [pce_max], s=35)

# Anotación movible (texto negro)
txt_pce = f"Máximo PCE\nPin = {pin_at_pce_max:.2f} dBm\nPCE = {pce_max:.2f}"
ann2 = ax2.annotate(
    txt_pce,
    xy=(pin_at_pce_max, pce_max),
    xytext=(pin_at_pce_max + 2, pce_max - 10),
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc=CFG["BOX_FACE"],
        ec=CFG["BOX_EDGE"],
        alpha=CFG["BOX_ALPHA"]
    ),
    arrowprops=dict(arrowstyle="->", lw=1.2),
    color=CFG["TEXT_COLOR"]
)
make_draggable(ann2)

# =========================
# Mostrar ambas ventanas
# =========================
print("\nTIP: Mueve los textos con el mouse. Cuando quede como quieres,")
print("     haz click en la ventana y presiona 's' para guardar en HD.")
plt.show()
