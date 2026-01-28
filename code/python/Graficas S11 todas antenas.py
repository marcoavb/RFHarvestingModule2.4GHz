import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CFG = {
    "FONT_SIZE": 18,
    "GRID_ALPHA": 0.30,
    "LINE_WIDTH": 2.2,
    "SAVE_DPI": 1600,

    "X_LABEL": "Frecuencia (GHz)",
    "Y_LABEL": r"$S_{11}$ (dB)",

    # Opcional: limita ejes si quieres
    "X_LIM": None,     # ejemplo: (2.2, 2.6)
    "Y_LIM": None,     # ejemplo: (-35, 0)

    "OUT_NAME": "S11_4archivos_highres.png",
}

# =========================
# TUS ARCHIVOS (4 CURVAS)
# =========================
CURVES = [
    {
        "path": "Yagi_2_4_final.s1p",
        "label": "Yagi (S1P)",
        "type": "touchstone",
        "port": 1,  # S11
    },
    {
        "path": "turnstile.s2p",
        "label": "Turnstile (S2P)",
        "type": "touchstone",
        "port": 1,  # S11
    },
    {
        "path": "meas_LaYUA_ASCII.txt",
        "label": "Medición LaYUA (ASCII)",
        "type": "ascii_ads",
    },
    {
        "path": "meas_TuSA_ASCII.txt",
        "label": "Medición TuSA (ASCII)",
        "type": "ascii_ads",
    },
]

plt.rcParams.update({
    "font.size": CFG["FONT_SIZE"],
    "axes.titlesize": CFG["FONT_SIZE"] + 2,
    "axes.labelsize": CFG["FONT_SIZE"],
})

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
# Lector ASCII (tus TXT)
# Espera columnas: freq_GHz  dB
# =========================
def read_ascii_ads(path: str):
    df = pd.read_csv(path, sep=r"\s+", engine="python", comment="!")
    df.columns = [c.strip() for c in df.columns]

    # tus archivos tienen: freq_GHz y dB
    if "freq_GHz" not in df.columns or "dB" not in df.columns:
        raise ValueError(f"[ASCII] Columnas no esperadas en {path}: {list(df.columns)}")

    f_ghz = pd.to_numeric(df["freq_GHz"], errors="coerce").to_numpy()
    s11_db = pd.to_numeric(df["dB"], errors="coerce").to_numpy()

    m = np.isfinite(f_ghz) & np.isfinite(s11_db)
    return f_ghz[m], s11_db[m]

# =========================
# Lector Touchstone (.s1p / .s2p)
# Soporta formatos: RI / MA / DB
# Devuelve frecuencia en GHz y S11 en dB
# =========================
def _to_complex(a, b, fmt: str):
    fmt = fmt.upper()
    if fmt == "RI":
        return a + 1j * b
    elif fmt == "MA":
        mag = a
        ang = np.deg2rad(b)
        return mag * (np.cos(ang) + 1j * np.sin(ang))
    elif fmt == "DB":
        mag = 10 ** (a / 20.0)
        ang = np.deg2rad(b)
        return mag * (np.cos(ang) + 1j * np.sin(ang))
    else:
        raise ValueError(f"Formato Touchstone no soportado: {fmt}")

def _freq_to_ghz(freq, unit: str):
    unit = unit.upper()
    if unit == "HZ":
        return freq / 1e9
    if unit == "KHZ":
        return freq / 1e6
    if unit == "MHZ":
        return freq / 1e3
    if unit == "GHZ":
        return freq
    raise ValueError(f"Unidad de frecuencia no soportada: {unit}")

def read_touchstone_s11(path: str):
    # Lee líneas, ignora comentarios "!" y vacías
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.strip() for ln in f.readlines()]

    data_lines = []
    unit = "GHZ"
    fmt = "MA"  # default común si no aparece, pero normalmente sí aparece

    for ln in raw:
        if not ln or ln.startswith("!"):
            continue
        if ln.startswith("#"):
            # Ejemplo: # GHZ S RI R 50
            parts = ln.split()
            # parts[1]=unidad, parts[3]=formato (RI/MA/DB) típicamente
            # Formato típico: # <unit> S <format> R <z0>
            if len(parts) >= 4:
                unit = parts[1].upper()
                fmt = parts[3].upper()
            continue
        data_lines.append(ln)

    if not data_lines:
        raise ValueError(f"No hay datos en: {path}")

    # Convierte a floats (maneja líneas con continuación)
    tokens = []
    for ln in data_lines:
        # algunos touchstone usan "D" en exponentes tipo Fortran
        ln = ln.replace("D", "E").replace("d", "E")
        tokens.extend(ln.split())

    vals = np.array([float(x) for x in tokens], dtype=float)

    # Determina tipo por extensión
    is_s1p = path.lower().endswith(".s1p")
    is_s2p = path.lower().endswith(".s2p")
    if not (is_s1p or is_s2p):
        raise ValueError(f"Extensión no soportada (usa .s1p o .s2p): {path}")

    if is_s1p:
        # por punto: freq + (S11: a b)
        cols = 3
        n = (len(vals) // cols) * cols
        vals = vals[:n].reshape(-1, cols)
        freq = vals[:, 0]
        a, b = vals[:, 1], vals[:, 2]
        s11 = _to_complex(a, b, fmt)

    else:
        # s2p: freq + S11(a b) S21(a b) S12(a b) S22(a b)
        cols = 9
        n = (len(vals) // cols) * cols
        vals = vals[:n].reshape(-1, cols)
        freq = vals[:, 0]
        a11, b11 = vals[:, 1], vals[:, 2]
        s11 = _to_complex(a11, b11, fmt)

    f_ghz = _freq_to_ghz(freq, unit)
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-15))
    return f_ghz, s11_db

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(8.8, 5.0))
enable_save_hotkey(fig, CFG["OUT_NAME"])

for c in CURVES:
    if c["type"] == "ascii_ads":
        f_ghz, s11_db = read_ascii_ads(c["path"])
    elif c["type"] == "touchstone":
        f_ghz, s11_db = read_touchstone_s11(c["path"])
    else:
        raise ValueError(f"Tipo no soportado: {c['type']}")

    ax.plot(
        f_ghz, s11_db,
        lw=CFG["LINE_WIDTH"],
        label=c["label"]
    )  # sin markers

ax.set_xlabel(CFG["X_LABEL"])
ax.set_ylabel(CFG["Y_LABEL"])
ax.grid(True, alpha=CFG["GRID_ALPHA"])

if CFG["X_LIM"] is not None:
    ax.set_xlim(CFG["X_LIM"])
if CFG["Y_LIM"] is not None:
    ax.set_ylim(CFG["Y_LIM"])

leg = ax.legend(loc="best", frameon=True)
leg.set_draggable(True)  # <- esto te deja mover la leyenda con el mouse

print("\nTIP:")
print("- Arrastra la leyenda con el mouse para ubicarla donde quieras.")
print("- Cuando quede como quieres, presiona 's' para guardar en HD.")
plt.show()
