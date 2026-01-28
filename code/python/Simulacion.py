import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CFG = {
    # Medición de salida del harvesting
    "V_DC": 1.65,          # V sobre la carga
    "R_LOAD": 900.0,       # ohm (tu carga medida)
    # Potencia disponible aproximada (se calcula con V^2/R)
    "ETA_CHG": 0.75,       # eficiencia desde harvesting -> supercap (boost + pérdidas)

    # Supercap
    "C_SUPERCAP": 0.47,    # F
    "V_CAP_INIT": 0.2,     # V (si quieres cold-start desde 0, pon 0.0)

    # Umbrales
    "V_ON": 2.50,
    "V_OFF": 2.20,

    # Cargas IoT (desde el supercap, ya en Vcap)
    "I_SLEEP": 20e-6,
    "I_SENSE": 3e-3,
    "T_SENSE": 0.15,
    "I_TX": 25e-3,
    "T_TX": 0.03,

    # Simulación
    "T_TOTAL": 1200.0,   # 20 min
    "DT": 1e-3,
}

Vdc = CFG["V_DC"]
R = CFG["R_LOAD"]
Pin = (Vdc * Vdc) / R              # W
Pchg = CFG["ETA_CHG"] * Pin        # W disponible para cargar el cap

dt = CFG["DT"]
t = np.arange(0, CFG["T_TOTAL"] + dt, dt)

C = CFG["C_SUPERCAP"]
Vcap = np.zeros_like(t)
Vcap[0] = CFG["V_CAP_INIT"]

state = "SLEEP"
phase = "IDLE"
phase_left = 0.0
event_times = []

def I_out(state, phase):
    if state == "SLEEP":
        return CFG["I_SLEEP"]
    if phase == "SENSE":
        return CFG["I_SENSE"]
    if phase == "TX":
        return CFG["I_TX"]
    return CFG["I_SLEEP"]

for k in range(1, len(t)):
    v = Vcap[k-1]

    # 1) Corriente de carga equivalente por potencia (boost idealizado)
    # I_chg = Pchg / Vcap  (si Vcap es muy pequeña, limitamos para evitar infinito numérico)
    v_eff = max(v, 0.05)   # evita división por cero
    I_chg = Pchg / v_eff

    # 2) Lógica UVLO / duty cycle
    if state == "SLEEP" and v >= CFG["V_ON"]:
        state = "ACTIVE"
        phase = "SENSE"
        phase_left = CFG["T_SENSE"]
        event_times.append(t[k-1])

    if state == "ACTIVE":
        phase_left -= dt
        if phase_left <= 0:
            if phase == "SENSE":
                phase = "TX"
                phase_left = CFG["T_TX"]
            else:
                state = "SLEEP"
                phase = "IDLE"
                phase_left = 0.0

        if v <= CFG["V_OFF"]:
            state = "SLEEP"
            phase = "IDLE"
            phase_left = 0.0

    # 3) Descarga por consumo
    Iload = I_out(state, phase)

    # 4) Dinámica del cap
    dv = (I_chg - Iload) * dt / C
    Vcap[k] = max(v + dv, 0.0)

Ecap = 0.5 * C * Vcap**2
event_times = np.array(event_times)
periods = np.diff(event_times) if len(event_times) >= 2 else np.array([])

print("=== Resultados ===")
print(f"Pin aprox = {Pin*1000:.3f} mW")
print(f"Eventos IoT = {len(event_times)}")
if len(periods) > 0:
    print(f"Periodo promedio = {periods.mean():.2f} s")
else:
    print("No hay suficientes eventos para periodo promedio.")

# PLOTS
fig1, ax1 = plt.subplots(figsize=(8.2, 4.8))
ax1.plot(t, Vcap, linewidth=2.0)
ax1.axhline(CFG["V_ON"], linestyle="--", linewidth=1.2)
ax1.axhline(CFG["V_OFF"], linestyle="--", linewidth=1.2)
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("Vcap (V)")
ax1.set_title("Vcap(t) con carga por potencia (boost idealizado)")
ax1.grid(True, alpha=0.3)

fig2, ax2 = plt.subplots(figsize=(8.2, 4.8))
ax2.plot(t, Ecap*1000.0, linewidth=2.0)
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("Energía (mJ)")
ax2.set_title("E(t)=0.5*C*Vcap^2")
ax2.grid(True, alpha=0.3)

plt.show()
