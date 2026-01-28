import os
import re
import numpy as np
import matplotlib

# =======================
# BACKEND INTERACTIVO (PyCharm)
# =======================
BACKEND = "Qt5Agg"   # si no tienes PyQt5 -> "TkAgg"
matplotlib.use(BACKEND)

import matplotlib.pyplot as plt


# =======================
# CONFIGURACIÓN
# =======================
FILE = "Radiacion_turnstile.txt"          # <-- tu archivo de CST

# Corte fijo:
PHI_MAIN = 90                       # tu caso: phi fijo en 0
USE_PHI_180_IF_AVAILABLE = True      # si el archivo tiene phi=180, lo usa para completar el plano
PHI_BACK = 0.0                     # phi "opuesto"

PEAK_TARGET_DEG = 180.0




# Qué columna usar como patrón (en tu archivo existe Abs(Dir.)[dBi])
# Opciones típicas: "Abs(Dir.)", "Abs(Theta)", "Abs(Phi)"
PATTERN_COLUMN = "Abs(Dir.)"
ANNOTATION_FONTSIZE = 14

plt.rcParams.update({
    "font.size": 20,        # tamaño general
    "axes.titlesize": 14,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 20
})



# Estilo
TITLE_POLAR = "Patrón de radiación (corte φ = 0°)"
CURVE_COLOR = "tab:blue"

# HPBW
HPBW_DROP_DB = 3.0                   # -3 dB
NORMALIZE_TO_0DB = True              # normaliza al máximo (0 dB en el pico)

# Exportación HD (después de ajustar con mouse)
EXPORT_DIR = "exports"
EXPORT_DPI = 600
EXPORT_BASENAME = "Turnstile_Polar_HPBW"
# =======================


# =======================
# UTILIDADES INTERACTIVAS
# =======================
def ensure_export_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def make_draggable(artist):
    """Compatibilidad matplotlib viejo/nuevo."""
    if hasattr(artist, "set_draggable"):
        artist.set_draggable(True)
    elif hasattr(artist, "draggable"):
        artist.draggable(True)


def register_editing(fig, prompt_prefix="Nuevo texto: "):
    """
    - Click en una caja (annotation/text) para seleccionarla
    - Presiona 'e' y escribe por consola
    - Delete/Backspace para ocultar/mostrar
    """
    fig._selected_target = None

    def on_pick(event):
        fig._selected_target = event.artist
        print("✅ Texto seleccionado. Presiona 'e' para editar, 'delete' para ocultar/mostrar.")

    def on_key(event):
        tgt = getattr(fig, "_selected_target", None)
        if tgt is None:
            return
        if event.key == "e":
            try:
                new_text = input(prompt_prefix)
            except EOFError:
                return
            if new_text.strip():
                tgt.set_text(new_text)
                fig.canvas.draw_idle()
        elif event.key in ["delete", "backspace"]:
            tgt.set_visible(not tgt.get_visible())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)


def save_figure(fig, base_name: str) -> None:
    ensure_export_dir(EXPORT_DIR)
    png_path = os.path.join(EXPORT_DIR, f"{base_name}.png")
    pdf_path = os.path.join(EXPORT_DIR, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print("\n✅ Exportación completada (con posiciones finales):")
    print(f" - {png_path}")
    print(f" - {pdf_path}\n")


def register_export_shortcut(fig, base_name: str):
    def on_key(event):
        if event.key.lower() == "s":
            save_figure(fig, base_name)
    fig.canvas.mpl_connect("key_press_event", on_key)


# =======================
# LECTOR DEL TXT DE CST (como el tuyo)
# =======================
def parse_cst_txt(path: str):
    """
    Lee el TXT de CST con header tipo:
    Theta [deg.]  Phi [deg.]  Abs(Dir.)[dBi]  Abs(Theta)[dBi] ... Ax.Ratio[dB]
    y filas numéricas con notación científica.
    Devuelve:
      - header_cols: lista de nombres de columnas
      - data: ndarray (N, M) float
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1) Encontrar línea de header (contiene Theta y Phi)
    header_idx = None
    for i, ln in enumerate(lines):
        if "Theta" in ln and "Phi" in ln and "Abs" in ln:
            header_idx = i
            header_line = ln.strip()
            break
    if header_idx is None:
        raise ValueError("No se encontró header con Theta/Phi/Abs en el archivo.")

    # 2) Parsear nombres de columnas: tomamos tokens “base” sin unidades
    # Ej: "Theta [deg.]" -> "Theta"
    # Ej: "Abs(Dir.)[dBi" -> "Abs(Dir.)"
    raw_tokens = header_line.split()

    cols = []
    skip_next = False
    for t in raw_tokens:
        if skip_next:
            skip_next = False
            continue

        # Junta casos tipo: "Theta [deg.]" "Phi [deg.]"
        if t in ["Theta", "Phi", "Phase(Theta)[deg.]", "Phase(Phi", "Phase(Phi)[deg.]"]:
            cols.append(t)
            continue

        # Si el token es "Theta" y el siguiente es "[deg.]" lo ignoramos
        if t in ["Theta", "Phi"] and "[deg." in raw_tokens[raw_tokens.index(t) + 1]:
            cols.append(t)
            skip_next = True
            continue

        # Quitar unidades entre corchetes si vienen pegadas
        t_clean = re.sub(r"\[.*?\]", "", t).strip()
        # Quitar tokens vacíos / unidades sueltas
        if t_clean and t_clean not in ["[deg.]", "[deg.]", "[dBi", "[dB"]:
            cols.append(t_clean)

    # Normalizar nombres específicos del header real (más robusto)
    # En tu archivo: Theta, Phi, Abs(Dir.), Abs(Theta), Phase(Theta), Abs(Phi), Phase(Phi), Ax.Ratio
    # Forzamos si detectamos patrones:
    normalized_cols = []
    for c in cols:
        c = c.replace("Abs(Phi", "Abs(Phi)")
        c = c.replace("Phase(Phi", "Phase(Phi)")
        c = c.replace("Phase(Theta)[deg.]", "Phase(Theta)")
        c = c.replace("Phase(Phi)[deg.]", "Phase(Phi)")
        c = c.replace("Ax.Ratio[dB", "Ax.Ratio")
        normalized_cols.append(c)

    # 3) Leer filas numéricas (saltando header y guiones)
    num_rows = []
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    for ln in lines[header_idx + 1:]:
        s = ln.strip()
        if not s:
            continue
        if set(s) <= set("-"):
            continue  # línea de guiones
        # Extraer todos los números
        nums = float_re.findall(s)
        if len(nums) < 3:
            continue
        num_rows.append([float(x) for x in nums])

    data = np.array(num_rows, dtype=float)

    # Algunas veces cols puede no calzar perfecto con data.shape[1] por unidades/tokenización
    # Entonces construimos un mapeo “manual” por posiciones esperadas si hace falta.
    # En tu archivo, el orden real es:
    # 0 Theta, 1 Phi, 2 Abs(Dir.), 3 Abs(Theta), 4 Phase(Theta), 5 Abs(Phi), 6 Phase(Phi), 7 Ax.Ratio
    if data.shape[1] >= 8:
        header_cols = ["Theta", "Phi", "Abs(Dir.)", "Abs(Theta)", "Phase(Theta)", "Abs(Phi)", "Phase(Phi)", "Ax.Ratio"]
        data = data[:, :8]
    else:
        # fallback: usa lo que haya
        header_cols = normalized_cols[:data.shape[1]]

    return header_cols, data


def get_col(header_cols, data, name):
    if name not in header_cols:
        raise ValueError(f"No encontré la columna '{name}'. Disponibles: {header_cols}")
    j = header_cols.index(name)
    return data[:, j]


# =======================
# CONSTRUIR CORTE + HPBW
# =======================
def build_plane_cut(theta, phi, pat_db, phi_main=90.0, use_phi_180=True, phi_back=180.0, tol=1e-6):
    """
    Construye un corte “plano” combinando:
    - phi=0   -> ángulos +theta
    - phi=180 -> ángulos -theta (si está disponible y use_phi_180=True)
    Si NO existe phi=180, solo devuelve +theta (0..180).
    """
    m0 = np.isclose(phi, phi_main, atol=tol)
    th0 = theta[m0]
    p0 = pat_db[m0]

    angles = [th0]
    pats = [p0]

    if use_phi_180:
        m180 = np.isclose(phi, phi_back, atol=tol)
        if np.any(m180):
            th180 = theta[m180]
            p180 = pat_db[m180]
            angles.append(-th180)
            pats.append(p180)

    ang = np.concatenate(angles)
    pat = np.concatenate(pats)

    idx = np.argsort(ang)
    return ang[idx], pat[idx]


def compute_hpbw_circular(angle_deg, pat_db, drop_db=3.0, peak_target_deg=0.0, window_deg=60.0):
    ang = np.mod(np.array(angle_deg, dtype=float), 360.0)
    p = np.array(pat_db, dtype=float)

    idx = np.argsort(ang)
    ang = ang[idx]
    p = p[idx]

    def circ_dist(a, b):
        d = np.abs(a - b) % 360.0
        return np.minimum(d, 360.0 - d)

    # ✅ elegir datos cerca del ángulo objetivo (ventana ±window_deg)
    d = circ_dist(ang, peak_target_deg)
    mask = d <= window_deg
    if not np.any(mask):
        # si por alguna razón no hay puntos en ventana, caer al más cercano
        i_peak = int(np.argmin(d))
    else:
        # pico local dentro de la ventana
        i_peak = int(np.argmax(p[mask]))
        i_peak = np.where(mask)[0][i_peak]

    peak_val = float(p[i_peak])
    peak_ang = float(ang[i_peak])
    target = peak_val - drop_db

    # Duplicar para búsqueda circular
    ang2 = np.concatenate([ang, ang + 360.0])
    p2 = np.concatenate([p, p])

    start = i_peak
    angw = ang2[start:start + len(ang) + 1]
    pw = p2[start:start + len(ang) + 1]

    # Derecha
    right = None
    for i in range(0, len(pw) - 1):
        if pw[i] >= target and pw[i + 1] <= target:
            x0, y0 = angw[i], pw[i]
            x1, y1 = angw[i + 1], pw[i + 1]
            t = (target - y0) / (y1 - y0 + 1e-12)
            right = float(x0 + t * (x1 - x0))
            break

    # Izquierda
    left = None
    for i in range(0, len(pw) - 1):
        j0 = -i
        j1 = j0 - 1
        if abs(j1) > len(pw):
            break
        if pw[j0] >= target and pw[j1] <= target:
            x0, y0 = angw[j0], pw[j0]
            x1, y1 = angw[j1], pw[j1]
            t = (target - y0) / (y1 - y0 + 1e-12)
            left = float(x0 + t * (x1 - x0))
            break

    if left is None or right is None:
        return peak_ang, None, None, None, peak_val

    left %= 360.0
    right %= 360.0
    bw = circ_dist(right, left)
    return peak_ang, left, right, float(bw), peak_val



# =======================
# MAIN
# =======================
header_cols, data = parse_cst_txt(FILE)

theta = get_col(header_cols, data, "Theta")
phi = get_col(header_cols, data, "Phi")

# Patrón en dB/dBi según columna elegida
pat = get_col(header_cols, data, PATTERN_COLUMN)

# Construir corte plano
angle_deg, pat_db = build_plane_cut(
    theta, phi, pat,
    phi_main=PHI_MAIN,
    use_phi_180=USE_PHI_180_IF_AVAILABLE,
    phi_back=PHI_BACK
)

# =======================
# EXTENDER PATRÓN SI EL PICO QUEDA EN BORDE (caso típico 0..180)
# =======================
def extend_cut_0_180_to_0_360(theta_deg, patt_db):
    """
    Si CST exporta 0..180 (un solo corte), reflejamos para obtener 0..360
    y evitar que el pico quede en el borde (necesario para HPBW).
    """
    t = np.array(theta_deg, dtype=float)
    p = np.array(patt_db, dtype=float)

    # Si ya hay datos negativos o >180, no tocamos
    if np.min(t) < 0 or np.max(t) > 180:
        return t, p

    # Reflejo: segunda mitad = 360 - theta (invertido)
    t2 = 360.0 - t[::-1]
    p2 = p[::-1]

    t_full = np.concatenate([t, t2])
    p_full = np.concatenate([p, p2])

    order = np.argsort(t_full)
    return t_full[order], p_full[order]


angle_deg, pat_db = extend_cut_0_180_to_0_360(angle_deg, pat_db)


# Normalizar a 0 dB (si quieres patrón clásico)
if NORMALIZE_TO_0DB:
    peak_abs = float(np.max(pat_db))
    pat_db_plot = pat_db - peak_abs
    peak_label_value = peak_abs  # pico absoluto (dBi)
else:
    pat_db_plot = pat_db
    peak_label_value = float(np.max(pat_db))

# HPBW (sobre el patrón ABS, no sobre el normalizado; el resultado es el mismo)
peak_ang, left_ang, right_ang, hpbw_deg, peak_val = compute_hpbw_circular(
    angle_deg, pat_db,
    drop_db=HPBW_DROP_DB,
    peak_target_deg=PEAK_TARGET_DEG,
    window_deg=60.0   # puedes subir a 90 si tu lóbulo es muy ancho
)



# Polar requiere radianes
ang_rad = np.deg2rad(angle_deg)

# =======================
# PLOT POLAR
# =======================
fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})

ax.plot(ang_rad, pat_db_plot, color=CURVE_COLOR, linewidth=1.8)

#ax.set_title(TITLE_POLAR, pad=18)
ax.set_theta_zero_location("N")   # 0° arriba
ax.set_theta_direction(-1)        # sentido horario

# rango radial con margen
rmin = float(np.min(pat_db_plot))
ax.set_rlim(rmin - 3, 1.0)

# Línea de referencia -3 dB (si está normalizado)
if NORMALIZE_TO_0DB:
    theta_full = np.linspace(0, 2 * np.pi, 720)
    ax.plot(theta_full, np.full_like(theta_full, -HPBW_DROP_DB), linestyle="--", linewidth=1.2)

# Marcadores y anotación HPBW
# Pico
ax.scatter([np.deg2rad(peak_ang)], [0.0 if NORMALIZE_TO_0DB else peak_val], zorder=5)

# Texto principal (movible + editable)
if hpbw_deg is not None:
    info = (
        f"Pico @ θ = {peak_ang:.1f}°\n"
        f"Pico = {peak_label_value+ 2.07:.2f} dBi\n"
        f"HPBW (−3 dB) = {hpbw_deg:.2f}°\n"
        f"θL={left_ang:.1f}°, θR={right_ang:.1f}°"
    )

    # Puntos -3 dB en el plot normalizado
    if NORMALIZE_TO_0DB:
        ax.scatter([np.deg2rad(left_ang)],  [-HPBW_DROP_DB], zorder=5)
        ax.scatter([np.deg2rad(right_ang)], [-HPBW_DROP_DB], zorder=5)

        # Arco a -3 dB
        t_arc = np.linspace(left_ang, right_ang, 200)
        ax.plot(np.deg2rad(t_arc), np.full_like(t_arc, -HPBW_DROP_DB), linestyle="--", linewidth=1.4)

    ann = ax.annotate(
        info,
        xy=(np.deg2rad(peak_ang), 0.0 if NORMALIZE_TO_0DB else peak_val),
        xytext=(0.02, 0.98), textcoords="axes fraction",
        ha="left", va="top",
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.9),
        fontsize=ANNOTATION_FONTSIZE
    )
else:
    ann = ax.annotate(
        "No se pudo calcular HPBW (−3 dB).\n"
        "Tip: asegúrate de que el patrón cruce −3 dB a ambos lados del pico\n"
        "dentro del rango angular exportado.",
        xy=(0.02, 0.98), xycoords="axes fraction",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.9),
        fontsize=ANNOTATION_FONTSIZE
    )

make_draggable(ann)
ann.set_picker(True)


# Editor + export con tecla S
register_editing(fig, prompt_prefix="Nuevo texto: ")
register_export_shortcut(fig, base_name=EXPORT_BASENAME)

print("\n=== CONTROLES ===")
print("🖱️  Arrastra las cajas/textos con el mouse.")
print("🖱️  Click para seleccionar un texto.")
print("⌨️  'e' para editar (escribes en consola).")
print("💾  's' para exportar en HD (PNG 600 dpi + PDF).")
print(f"📁  Carpeta: ./{EXPORT_DIR}/\n")

print("=== DATOS LEÍDOS ===")
print(f"Columnas detectadas: {header_cols}")
print(f"Filtrado: φ={PHI_MAIN}°" + (" (+ φ=180°)" if USE_PHI_180_IF_AVAILABLE else ""))

if hpbw_deg is not None:
    print(f"\nHPBW (−3 dB): {hpbw_deg:.2f}° | θL={left_ang:.2f}°, θR={right_ang:.2f}° | Pico={peak_label_value:.2f} dBi")
else:
    print("\nHPBW: no calculado (no se hallaron cruces a −3 dB).")

plt.show()
