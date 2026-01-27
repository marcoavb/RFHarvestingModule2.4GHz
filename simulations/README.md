
# RF Energy Harvesting Simulations

This section contains all the electromagnetic and circuit-level simulations used in the development of the RF Energy Harvesting module.

The simulations cover the complete chain:

📡 Antenna → 🔧 Matching Network → 🔋 RF-DC Rectifier → ⚙️ Power Management

---

## 1. Simulation Domains

| Domain | Tool | Purpose |
|-------|------|---------|
| Electromagnetics | CST Studio Suite | Antenna design and radiation analysis |
| Microwave Circuits | ADS | Matching network and RF rectifier |
| Power Electronics | LTspice | Power management and storage behavior |

---

## 2. Folder Structure

```
simulations/
│
├── cst/
│   ├── antenna_design/
│   ├── radiation_patterns/
│   └── s11_results/
│
├── ads/
│   ├── matching_network/
│   ├── rectifier/
│   └── harmonic_balance/
│
└── ltspice/
    ├── rectifier_transient/
    └── power_management/
```

---

## 3. Antenna Simulations (CST)

These simulations focus on:

- Reflection coefficient (S11)
- Gain and radiation patterns
- Impedance behavior at 2.4 GHz
- Substrate and geometry optimization

Open with:
```
CST Studio Suite → Open Project → simulations/cst/
```

---

## 4. Matching Network (ADS)

Purpose:

- Impedance transformation to maximize RF power transfer
- S-parameter analysis
- Smith chart matching verification

Simulation types:

- Linear S-parameter simulation
- Optimization sweeps

---

## 5. RF-DC Rectifier (ADS + LTspice)

Includes:

- Schottky diode nonlinear modeling
- Harmonic balance simulation
- Efficiency vs input power analysis
- Output voltage behavior

---

## 6. Power Management Module (LTspice)

Focuses on:

- Boost converter operation
- Supercapacitor charging
- Startup transient behavior
- Load regulation

Run LTspice files in:
```
simulations/ltspice/
```

---

## 7. Notes on File Sizes

Electromagnetic simulation files can be large. Some CST and ADS projects may be:

- Compressed
- Tracked using Git LFS
- Provided as lightweight versions

---

## 8. How to Reproduce Key Results

### Antenna S11
Open CST project → Run frequency sweep → Export S-parameters.

### Rectifier Efficiency
Run ADS Harmonic Balance → Measure DC output vs RF input power.

### PMM Startup
Run LTspice transient simulation.

---

## 9. Outputs Used in the Thesis

The results generated here are used to produce:

- S11 comparison plots  
- Radiation patterns  
- Efficiency curves  
- Output voltage graphs  

Processed plots are stored in:

```
results/plots/
```

---

## 10. Simulation Philosophy

These simulations were designed to:

✔ Bridge EM and circuit domains  
✔ Validate design before fabrication  
✔ Minimize mismatch losses  
✔ Optimize RF-to-DC conversion  

This ensures the experimental prototype closely follows theoretical predictions.
