
# Potato Soil Nitrogen and Tuber Growth Prediction

This repository contains an encoder-only Transformer model designed to predict daily soil nitrogen dynamics and potato tuber growth. The approach addresses the challenge of nitrogen management in sandy soils by combining broad system knowledge with field realism.

## Modeling Pipeline
The pipeline consists of a base model pretrained on 4.7 million DSSAT-generated scenarios (incorporating weather, soil, and management data) and a fine-tuned model adapted using ground-truth field observations. 

## Getting Started
You can explore the model predictions and compare the Base, Fine-Tuned (FT), and DSSAT models by running the provided Jupyter Notebook:
`inference.ipynb`

## Soil Model Results (NRMSE)

| FARM-YEAR | NRMSE-Base | NRMSE-FT | NRMSE-DSSAT | Improvement over Base (%) | Improvement over DSSAT (%) |
| --- | --- | --- | --- | --- | --- |
| F1-2014 | 0.14 | **0.04** | 0.05 | 72.27 | 26.32 |
| F1-2012 | 0.07 | **0.07** | 0.11 | 6.20 | 36.78 |
| F2-2011 | 0.08 | **0.04** | 0.09 | 46.49 | 53.98 |
| **Average** |  |  | | **41.65** | **39.02** |

## Tuber Model Results (NRMSE)

| FARM-YEAR | NRMSE-Base | NRMSE-FT | NRMSE-DSSAT | Improvement over Base (%) | Improvement over DSSAT (%) |
| --- | --- | --- | --- | --- | --- |
| F1-2014 | 0.11 | **0.09** | 0.35 | 21.02 | 75.33 |
| F1-2012 | 0.22 | **0.04** | 0.17 | 79.84 | 74.13 |
| F2-2011 | 0.28 | **0.02** | 0.26 | 94.68 | 94.25 |
| **Average** |  |  |  | **65.18** | **81.24** |

