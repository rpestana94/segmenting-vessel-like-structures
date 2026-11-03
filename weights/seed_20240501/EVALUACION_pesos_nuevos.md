# Evaluación de los pesos re-entrenados (SEED = 20240501)

Entrenamiento completo en GPU (RTX 5080), 100 épocas ambos módulos. Evaluados en el stack pinneado (TF 2.16.1 / Keras 3.3.3).

## Super-resolución ESPCN — Set5 / Set14 (PSNR dB, canal Y)

| Dataset | Bicúbico | ESPCN 9-ep (previo) | **ESPCN 100-ep (nuevo)** | SSIM nuevo | «Ours» paper |
|---|---|---|---|---|---|
| SET5 | 30.4 | 31.01 | **32.28** | 0.9061 | 37.92 |
| SET14 | 27.63 | 28.21 | **29.06** | 0.8191 | 37.46 |

El ESPCN de 100 épocas mejora el bicúbico en **+1.9 dB (Set5)** y **+1.4 dB (Set14)**, dentro del rango de la literatura de SR (~32/29 dB). Sigue **~5–8 dB por debajo** de los 37.9/37.5 dB publicados como «Ours», confirmando que esa cifra no es alcanzable con un ESPCN correctamente entrenado.

## Segmentación U-Net — DRIVE 21–40 (n=20)

| Métrica | U-Net publicado | **U-Net nuevo (seed)** |
|---|---|---|
| dice | 0.8115 | **0.8165** |
| iou | 0.6835 | **0.6906** |
| precision | 0.8757 | **0.8585** |
| recall | 0.7587 | **0.7811** |
| roc_auc | 0.9797 | **0.9789** |

El U-Net re-entrenado con semilla reproduce el rendimiento del checkpoint publicado (Dice 0.8165 vs 0.8115), confirmando que la segmentación es reproducible desde cero con semilla fija.
