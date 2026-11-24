# Predicción de riesgo crediticio — Competencia GCI 2026 (Universidad de Tokio)

Modelo para estimar la probabilidad de que un cliente caiga en mora con su préstamo, desarrollado para la competencia del programa **GCI World** del Matsuo-Iwasawa Laboratory (Universidad de Tokio), basada en el problema *Home Credit Default Risk*.

**Resultado: Top 10 % entre ~5 000 participantes.**

| | |
| --- | --- |
| Tarea | Clasificación binaria: ¿el cliente caerá en mora? (`TARGET = 1`) |
| Métrica | ROC-AUC |
| Datos | 171 202 solicitudes de entrenamiento y 61 500 de prueba; 8,07 % de morosos |
| Modelo final | Ensamble de LightGBM y CatBoost sobre dos vistas de los datos |
| AUC de validación cruzada | **0,7695** (ensamble, 5 folds) |

## Enfoque

### 1. Variables

A partir de las columnas originales construí más de 100 variables, agrupadas en cinco ideas:

- **Capacidad de pago:** razones crédito/ingreso, cuota/ingreso, crédito/precio del bien e ingreso por persona.
- **Puntajes externos (`EXT_SOURCE`):** las variables más predictivas, combinadas con media, mínimo, máximo, producto y diferencias.
- **Tasa de interés implícita:** a partir del crédito y la cuota, despejo la tasa con la fórmula de amortización francesa y un método de bisección. Como el prestamista cobra más a quien considera más riesgoso, la tasa contiene su propia evaluación de riesgo. Mejoró el AUC de LightGBM de 0,7614 a 0,7641, y lo hizo en los 5 folds.
- **Clientes repetidos:** el dataset no tiene un ID de persona, pero las diferencias entre las variables `DAYS_*` (por ejemplo, la edad al emitir el documento) se mantienen entre solicitudes de una misma persona. Con eso reconstruyo su identidad y calculo la tasa de mora de sus **otras** solicitudes: si otro préstamo del cliente cayó en mora, el riesgo sube a 25 %, frente a 4,8 % cuando se pagó.
- **Datos faltantes:** cuántos y cuáles faltan, porque la ausencia de un dato también informa.

### 2. Dos vistas de los datos

- **Vista A:** las variables anteriores.
- **Vista B:** además, estima los `EXT_SOURCE` faltantes con modelos auxiliares entrenados sin usar `TARGET`.

### 3. Validación sin fuga de información

- Validación cruzada estratificada de 5 folds.
- Las variables que usan `TARGET` (mora de otras solicitudes del cliente) se calculan **dentro de cada fold**, solo con las etiquetas del fold de entrenamiento y nunca con la etiqueta de la propia fila.
- Los pesos del ensamble se eligen con las predicciones de validación cruzada, **no con el leaderboard**, para no sobreajustarse a él.

### 4. Reentrenamiento y ensamble

Cada modelo se reentrena con el 100 % de los datos y 3 semillas. Lo comprobé con una simulación anidada: reentrenar con todo dio mejor AUC que promediar los modelos de los folds en todos los casos probados (+0,0009 en LightGBM y +0,0012 en CatBoost). Las predicciones se combinan por rangos, porque el AUC solo depende del orden.

## Resultados de validación cruzada

| Modelo | AUC (5 folds) |
| --- | --- |
| LightGBM, vista A | 0,7658 |
| LightGBM, vista B | 0,7677 |
| CatBoost, vista A | 0,7660 |
| CatBoost, vista B | 0,7671 |
| **Ensamble** (pesos 0,10 / 0,40 / 0,25 / 0,25) | **0,7695** |

## Cómo reproducirlo

Los datos de la competencia no se incluyen porque no se pueden redistribuir.

1. Abre `notebooks/competition_model.ipynb` en Google Colab.
2. Crea en tu Google Drive la carpeta `Competition/input/` y coloca ahí `train.csv`, `test.csv` y `sample_submission.csv`.
3. Ejecuta todas las celdas. Con `SEED = 42` el resultado es reproducible. En CPU tarda alrededor de 1,5 horas; con GPU, CatBoost es bastante más rápido (`USE_GPU = True`).

Para ejecutarlo fuera de Colab, instala las dependencias con `pip install -r requirements.txt` y define la variable de entorno `COMP_DIR` con la carpeta que contiene `input/`.

## Estructura

```
├── notebooks/
│   └── competition_model.ipynb   Pipeline completo: variables, validación, modelos y ensamble
├── requirements.txt
└── README.md
```

## Fuentes y créditos

Las razones financieras y las combinaciones de `EXT_SOURCE` se inspiran en notebooks públicos de la competencia original de Kaggle *Home Credit Default Risk* (2018). La tasa implícita usa la fórmula estándar de amortización francesa.

Autor: Gabriel David Toro Suyllon ([@gabrielsuyllon-nt](https://github.com/gabrielsuyllon-nt))
