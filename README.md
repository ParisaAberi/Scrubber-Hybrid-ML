# Two-Stage Physics-Informed Hybrid ML for Scrubber Pressure Drop Prediction

## Overview

This project presents a two-stage physics-informed hybrid machine learning framework for predicting pressure drop in a packed-bed scrubber.

The main idea is to combine a physics-based model with machine learning models. The physics model estimates the pressure drop based on process conditions and scrubber fouling. Machine learning models are then used to learn the remaining prediction error and correct the physics-based results.

The framework includes model training, validation, model selection, and final testing to compare the performance of different approaches.

## Methodology

The framework consists of the following steps:

1. **Process simulation:** Generate process conditions over a 600-hour simulation period.
2. **Physics-based modeling:** Calculate pressure drop based on gas properties, deposition, porosity changes, the Ergun equation, and demister pressure drop.
3. **Feature engineering:** Prepare baseline and physics-informed input features.
4. **Machine learning:** Train and compare LSBoost, regression tree ensembles, and neural network models.
5. **Hybrid modeling:** Predict the residual error of the physics-based model and use it to correct its predictions.
6. **Model validation and selection:** Compare the models using R², RMSE, and MAE, select the best hybrid model based on validation R², and evaluate the models on a separate test set.

## Implementation

The framework is implemented in MATLAB. The main script includes the simulation, model training and evaluation procedures, and custom supporting functions.

The simulation generates 2,000 time points. After preprocessing and feature construction, the dataset contains 1,999 samples, divided into:

- Training set: 1,199 samples
- Validation set: 399 samples
- Test set: 401 samples

## Data

The current version generates synthetic process data internally and does not require an external dataset.

The pressure-drop target is constructed from the physics-based pressure-drop calculation, an additional concentration-dependent term, and synthetic sensor noise.

Therefore, the reported results represent model performance on simulated data, not validation against measured industrial data.

## Results

### Validation Performance

| Model                     |          R² |       RMSE |        MAE |
| ------------------------- | ----------: | ---------: | ---------: |
| Physics Only              |     0.93065 |     51.085 |     42.902 |
| Baseline LSBoost          |     0.97562 |     30.290 |     24.233 |
| Baseline Random Forest    |     0.97632 |     29.854 |     23.958 |
| Baseline Neural Network   |     0.95631 |     40.548 |     31.978 |
| Hybrid LSBoost            |     0.97646 |     29.763 |     23.988 |
| Hybrid Random Forest      |     0.97673 |     29.592 |     23.820 |
| **Hybrid Neural Network** | **0.97765** | **28.999** | **23.360** |

### Test Performance

| Model                     |          R² |       RMSE |        MAE |
| ------------------------- | ----------: | ---------: | ---------: |
| Physics Only              |     0.92596 |     52.001 |     43.083 |
| Baseline LSBoost          |     0.97069 |     32.718 |     25.708 |
| Baseline Random Forest    |     0.97065 |     32.741 |     25.766 |
| Baseline Neural Network   |     0.95212 |     41.818 |     32.161 |
| Hybrid LSBoost            |     0.97045 |     32.851 |     25.729 |
| Hybrid Random Forest      |     0.97190 |     32.034 |     25.181 |
| **Hybrid Neural Network** | **0.97307** | **31.361** | **24.345** |

The Hybrid Neural Network achieved the highest validation R² of 0.97765 and the highest test R² of 0.97307 among the evaluated models.

The results show that machine learning can improve pressure-drop predictions compared with the physics-only model under the simulated conditions. However, the improvement from hybrid modeling varies across the different algorithms.

## How to Run

1. Download or clone this repository.
2. Open MATLAB and set the repository folder as the current working directory.
3. Run `Srubber_ML.m`.

The script generates the process data, trains and evaluates the models, displays diagnostic plots, and saves the output files.

## Output Files

After execution, the script generates:

- `Model_Validation_Performance.xlsx` — model performance on the validation set.
- `Model_Test_Performance.xlsx` — model performance on the test set.
- `Validated_HybridML_Workspace.mat` — saved MATLAB workspace variables.

Diagnostic plots are displayed during execution but are not included as repository files in this version.

## Limitations and Future Work

This version is a simulation-based implementation. The generated data are not measured industrial data, and the reported performance should be interpreted accordingly.

Future work may include validation using real process data, time-based evaluation, further development of the physics-based model, and additional diagnostic visualizations.
