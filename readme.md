# RF Energy Harvesting Module for IoT Applications

![](https://img.shields.io/badge/Field-RF%20Energy%20Harvesting-blue)
![](https://img.shields.io/badge/Band-2.4GHz%20ISM-green)
![](https://img.shields.io/badge/Type-Academic%20Research-orange)
![](https://img.shields.io/badge/Status-Experimental-success)

![cover](figures/system_block.png)

- [RF Energy Harvesting Module for IoT Applications](#rf-energy-harvesting-module-for-iot-applications)
  * [1. Introduction](#1-introduction)
  * [2. Dependency](#2-dependency)
    + [2.1. Software Tools](#21-software-tools)
    + [2.2. Hardware & Measurement Equipment](#22-hardware--measurement-equipment)
  * [3. What's in this repository](#3-whats-in-this-repository)
    + [3.1. Antenna Design](#31-antenna-design)
    + [3.2. Matching Network](#32-matching-network)
    + [3.3. RF-DC Rectifier](#33-rf-dc-rectifier)
    + [3.4. Power Management Module](#34-power-management-module)
    + [3.5. Measurements](#35-measurements)
    + [3.6. Data Processing Scripts](#36-data-processing-scripts)
  * [4. How to Reproduce Results](#4-how-to-reproduce-results)
    + [4.1. Run Simulations](#41-run-simulations)
    + [4.2. Process Measurement Data](#42-process-measurement-data)
  * [5. Q&A](#5-qa)
  * [License](#license)

---

## 1. Introduction

Ambient radiofrequency signals emitted by WiFi routers, cellular networks, and wireless devices represent a constant yet underutilized energy source. This project explores the **capture, conversion, and management** of that energy to power low-consumption electronic systems.

This repository documents the complete development of an **RF Energy Harvesting Module** operating at **2.4 GHz**, including:

- 📡 High-gain antenna  
- 🔧 Impedance matching network  
- 🔋 Schottky-diode RF-to-DC rectifier  
- ⚙️ Power management stage  

The objective is to demonstrate the feasibility of **battery-less or extended-lifetime IoT systems** powered by ambient electromagnetic energy.

---

## 2. Dependency

### 2.1. Software Tools

- CST Studio Suite (Electromagnetic simulation)
- ADS (Microwave circuit design)
- LTspice (Rectifier and PMM simulation)
- Python ≥ 3.8 (Data processing)
  - NumPy
  - Pandas
  - Matplotlib

### 2.2. Hardware & Measurement Equipment

- NanoVNA / Vector Network Analyzer
- Oscilloscope
- RF Signal Source (or ambient WiFi signals)
- FR4 PCB prototypes
- Schottky diodes (e.g., SMS7630)
- Supercapacitors / storage elements

---

## 3. What's in this repository

### 3.1. Antenna Design
Contains EM models, radiation patterns, and reflection coefficient (S11) results for the harvesting antenna.

- `simulations/cst/`  
- `results/plots/`  

### 3.2. Matching Network
Impedance transformation networks designed to maximize power transfer to the rectifier.

- `simulations/ads/matching/`  

### 3.3. RF-DC Rectifier
Nonlinear diode-based rectifier designs, including efficiency optimization across power levels.

- `simulations/ads/rectifier/`  
- `simulations/ltspice/rectifier/`  

### 3.4. Power Management Module
Boost conversion and energy storage to provide stable DC output for low-power loads.

- `simulations/ltspice/pmm/`  
- `docs/notes/`  

### 3.5. Measurements
Experimental measurement data and plots, including S-parameters and output voltage behavior.

- `data/raw/`  
- `data/processed/`  
- `results/plots/`  

### 3.6. Data Processing Scripts
Python scripts used to process data and generate publication-ready plots.

- `code/python/`

---

## 4. How to Reproduce Results

### 4.1. Run Simulations

- Antenna (CST): open projects in `simulations/cst/`
- Matching + Rectifier (ADS): open projects in `simulations/ads/`
- PMM (LTspice): open schematics in `simulations/ltspice/`

> Note: Some simulation projects may be compressed or stored using Git LFS due to file size.

### 4.2. Process Measurement Data

Example (Python):

```bash
cd code/python
python plot_s11.py
