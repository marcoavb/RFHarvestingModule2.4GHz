import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from matplotlib.legend import Legend

# ========= Config rápida =========
FONT_SIZE = 16
COL_TL   = "tab:blue"
COL_STUB = "tab:orange"
LW = 2.4
SAVE_DPI = 600
OUTFILE = "Smith_desplazamientos_curvos.png"

plt.rcParams.update({"font.size": FONT_SIZE})

# ========= Datos =========
Z0 = 50
ZL = 25 - 1j*66  # Zin (carga)

theta_tl_deg   = 32.812
theta_stub_deg = 116.828
NPTS = 260

# ========= Utilidades =========
def gamma_from_z(Z, Z0=50):
    return (Z - Z0) / (Z + Z0)

def zin_tline(ZL, Z0, theta_deg):
    th = np.deg2rad(theta_deg)
    t = np.tan(th)
    return Z0 * (ZL + 1j*Z0*t) / (Z0 + 1j*ZL*t)

def y_open_stub_shunt(Z0, theta_deg):
    th = np.deg2rad(theta_deg)
    return 1j * np.tan(th) / Z0

def enable_save_hotkey(fig):
    def on_key(event):
        if event.key.lower() == "s":
            fig.savefig(OUTFILE, dpi=SAVE_DPI, bbox_inches="tight")
            print(f"[OK] Guardado: {OUTFILE} (dpi={SAVE_DPI})")
    fig.canvas.mpl_connect("key_press_event", on_key)


def make_draggable(obj, use_axes_fraction=False):
    """
    Drag & drop para:
    - Annotations/Text (tienen get_position/set_position)
    - Legend (maneja su bbox_to_anchor)
    """
    fig = obj.figure
    state = {"press": None}

    # --- Legend: se mueve con bbox_to_anchor ---
    if isinstance(obj, Legend):
        def on_press(event):
            if event.inaxes is None:
                return
            bbox = obj.get_window_extent()
            if not bbox.contains(event.x, event.y):
                return
            state["press"] = (event.x, event.y,
                              obj.get_bbox_to_anchor()._bbox.x0,
                              obj.get_bbox_to_anchor()._bbox.y0)

        def on_motion(event):
            if state["press"] is None:
                return
            dx = event.x - state["press"][0]
            dy = event.y - state["press"][1]

            inv = fig.transFigure.inverted()
            p0 = inv.transform((0, 0))
            p1 = inv.transform((dx, dy))
            dfx, dfy = (p1 - p0)

            x0 = state["press"][2] + dfx
            y0 = state["press"][3] + dfy
            obj.set_bbox_to_anchor((x0, y0), transform=fig.transFigure)
            fig.canvas.draw_idle()

        def on_release(event):
            state["press"] = None
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_release_event", on_release)
        return

    # --- Text/Annotation ---
    def on_press(event):
        if event.inaxes != obj.axes:
            return
        ok, _ = obj.contains(event)
        if not ok:
            return
        x0, y0 = obj.get_position()
        if use_axes_fraction:
            state["press"] = (x0, y0, event.x, event.y)
        else:
            state["press"] = (x0, y0, event.xdata, event.ydata)

    def on_motion(event):
        if state["press"] is None:
            return
        if event.inaxes != obj.axes:
            return

        x0, y0, xpress, ypress = state["press"]

        if use_axes_fraction:
            inv = obj.axes.transAxes.inverted()
            x_now, y_now = inv.transform((event.x, event.y))
            x_old, y_old = inv.transform((xpress, ypress))
            obj.set_position((x0 + (x_now - x_old), y0 + (y_now - y_old)))
        else:
            obj.set_position((x0 + (event.xdata - xpress), y0 + (event.ydata - ypress)))

        fig.canvas.draw_idle()

    def on_release(event):
        state["press"] = None
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)


def fit_circle_complex(z):
    x = np.real(z)
    y = np.imag(z)
    M = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    A, B, C = np.linalg.lstsq(M, b, rcond=None)[0]
    cx = -A/2
    cy = -B/2
    r = np.sqrt((A**2 + B**2)/4 - C)
    return (cx + 1j*cy), r

def unwrap_long_arc(a0, a1):
    d = (a1 - a0 + np.pi) % (2*np.pi) - np.pi  # delta corto [-pi, pi]
    # arco largo (complementario) manteniendo endpoints
    if d >= 0:
        return d - 2*np.pi
    else:
        return d + 2*np.pi

# ========= Etapa 0 =========
g0 = gamma_from_z(ZL, Z0)

# ========= Etapa TL =========
phi = np.deg2rad(np.linspace(0.0, -2.0*theta_tl_deg, NPTS))
g_tl = g0 * np.exp(1j * phi)

g1 = g_tl[-1]
Z1 = zin_tline(ZL, Z0, theta_tl_deg)

# ========= Etapa Stub (corta) =========
Y1 = 1.0 / Z1
Ystub_final = y_open_stub_shunt(Z0, theta_stub_deg)
B_final = np.imag(Ystub_final)

B_path = np.linspace(0.0, B_final, NPTS)
Y_path = Y1 + 1j * B_path
Z_path = 1.0 / Y_path
g_stub_short = gamma_from_z(Z_path, Z0)

# ========= Stub (arco largo, mismo inicio/fin) =========
c, r = fit_circle_complex(g_stub_short)
a0 = np.angle(g_stub_short[0] - c)
a1 = np.angle(g_stub_short[-1] - c)
d_long = unwrap_long_arc(a0, a1)

angles = a0 + np.linspace(0.0, d_long, NPTS)
g_stub_long = c + r * np.exp(1j * angles)

g2 = g_stub_short[-1]  # endpoint físico (mismo)

# ========= Plot Smith =========
fig, ax = plt.subplots(figsize=(6.8, 6.8))
enable_save_hotkey(fig)

rf.plotting.smith(ax=ax)

# Curvas
ax.plot(g_tl.real, g_tl.imag, lw=LW, color=COL_TL,   label=f"Línea: {theta_tl_deg:.3f}°")
ax.plot(g_stub_long.real, g_stub_long.imag, lw=LW, color=COL_STUB, label=f"Stub abierto: {theta_stub_deg:.3f}°")

# ========= Marcadores solicitados =========
# 1) Zin = 25 - j66 (este es ZL)
ax.scatter([g0.real], [g0.imag], s=55, color="tab:green")
ann_zin = ax.annotate(
    "Zin = 25 - j66",
    xy=(g0.real, g0.imag),
    xytext=(0.08, 0.90),
    textcoords="axes fraction",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=0.95),
    color="black"
)

# 2) Zo = 50 (centro de la Smith: Γ=0)
g_zo = 0 + 0j
ax.scatter([g_zo.real], [g_zo.imag], s=65, color="tab:red")
ann_zo = ax.annotate(
    "Zo = 50",
    xy=(0.0, 0.0),
    xytext=(0.08, 0.82),
    textcoords="axes fraction",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=0.95),
    color="black"
)

# Otros puntos (intermedio/final) sin etiquetas
ax.scatter([g1.real], [g1.imag], s=40, color=COL_TL)
ax.scatter([g2.real], [g2.imag], s=55, color=COL_STUB)

# Leyenda solo con grados (y movible)
leg = ax.legend(loc="lower right")

# ========= Hacer movibles anotaciones y leyenda =========
make_draggable(ann_zin, use_axes_fraction=True)
make_draggable(ann_zo,  use_axes_fraction=True)
make_draggable(leg)  # leyenda drag & drop

print("TIP: Arrastra las anotaciones y la leyenda con el mouse.")
print("     Cuando quede como quieres, presiona 's' para guardar en HD.")
plt.show()
