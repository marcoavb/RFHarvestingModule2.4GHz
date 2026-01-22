# RFHarvestingModule2.4GHz

Módulo de recolección de energía RF (RF Energy Harvesting) enfocado en la banda de 2.4 GHz (Wi‑Fi / Bluetooth). Este repositorio contiene el diseño, documentación y recursos (esquemáticos, PCB, lista de materiales) para un módulo capaz de convertir energía RF en energía DC utilizable para dispositivos de baja potencia.

## Descripción
El módulo captura señales RF en 2.4 GHz, las rectifica y almacena la energía en un elemento de almacenamiento (supercondensador o batería pequeña). Está pensado para experimentación, prototipado y demostraciones de recolección de energía con emisores en la banda ISM (p. ej. Wi‑Fi, Bluetooth, transmisores RF, generadores de señal).

## Características principales
- Antena de entrada optimizada para 2.4 GHz
- Red de adaptación de impedancia para máximo acople
- Rectificador de RF (topología de diodo o rectificador de banda ancha)
- Filtro y regulación básica a la salida
- Puntos de prueba para medición de tensión y corriente
- Conector para almacenamiento (supercap / batería / condensador)
- PCB y esquemático incluidos (si aplica)

## Casos de uso
- Demostraciones de recolección de energía en aulas y ferias
- Prototipos de sensores pasivos o micro‑dispositivos energizados por RF
- Estudios de eficiencia y optimización de antenas/matching networks

## Contenido del repositorio
- /hardware — esquemáticos y archivos de PCB (EAGLE/ KiCad / Altium)
- /bom — lista de materiales (BOM) con referencias y alternativas
- /docs — notas de diseño, cálculos y mediciones
- /examples — ejemplos de montaje y pruebas
- README.md — este archivo

(Ajusta las rutas anteriores si en tu repo la estructura difiere.)

## Componentes recomendados (ejemplos)
- Rectificador RF: HSMS‑285x, SMS7630 (o topología Schottky de baja caída)
- Mezcla de condensadores y inductores para matching (SMD)
- Diodos Schottky para la etapa de rectificación
- Supercondensador o batería recargable pequeña para almacenamiento
- Regulador/convertidor DC‑DC de baja potencia (opcional) para obtener salida estable
- Conector para antena (u.FL) o antena PCB integrada

> Nota: Revisa la disponibilidad y footprints en la BOM incluida.

## Construcción e instalación
1. Revisa el esquema y la BOM en /hardware y /bom.
2. Fabrica la PCB usando los archivos en /hardware (Gerbers o proyecto).
3. Suelda componentes SMD/TH según la lista de materiales.
4. Conecta una antena compatible con 2.4 GHz (o usa conector u.FL).
5. Conecta el elemento de almacenamiento (supercap o batería) en la salida indicada.

## Pruebas y verificación
- Medir la tensión en el punto de salida sin carga y con carga de prueba (resistencia o dispositivo de baja potencia).
- Fuente de prueba recomendada:
  - Generador de señal RF (2.4 GHz) con control de potencia.
  - Transmisor Wi‑Fi o un paquete Bluetooth cercano para pruebas reales.
- Instrumentos útiles:
  - Analizador de espectro o receptor SDR para verificar la señal de entrada.
  - Multímetro/osciloscopio para medir la salida DC.
- Observaciones: la potencia disponible desde fuentes RF ambientales es muy baja; para obtener tensiones útiles probablemente necesites una fuente RF cerca y/o un transmisor dedicado.

## Pinout / Conexiones
- ANT: entrada de RF (conector u.FL o pad de antena)
- GND: referencia de tierra
- VOUT: salida DC rectificada (conectar a almacenamiento o carga)
- TEST+: punto de medida en la salida
- TEST-RF: punto de medida en la RF antes del rectificador (si disponible)

(Ajusta los nombres de pines según el esquemático real del proyecto.)

## Seguridad y regulaciones
- No exceder límites legales de transmisión RF en tu país.
- Evita transmitir señales de alta potencia sin las autorizaciones necesarias.
- Ten en cuenta que la recolección de energía a partir de transmisiones de terceros puede estar sujeta a regulación local.

## Contribuciones
Contribuciones bienvenidas:
- Mejoras en matching network y eficiencia
- Variantes de antena
- Integración con PMICs de baja potencia
- Resultados de medidas y protocolos de prueba

Por favor abre issues o pull requests proponiendo cambios. Incluye:
- Descripción del cambio
- Archivos modificados (esquemático, PCB, BOM)
- Resultados de pruebas y mediciones (si aplica)

## Licencia
Este proyecto está bajo licencia MIT. Cambia la licencia si prefieres otra.

## Créditos & Contacto
Autor: marcoavb  
Email / GitHub: @marcoavb

Si quieres, puedo:
- Añadir imágenes, diagramas o links a esquemáticos concretos
- Generar un template de BOM detallada
- Crear badges (build, license, etc.)
- Preparar un CONTRIBUTING.md y plantillas para issues/PRs
