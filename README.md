# Two-Stage Physics-Informed Hybrid ML for Scrubber Pressure Drop Prediction

## Overview

This project presents a two-stage physics-informed hybrid machine learning framework for predicting pressure drop in a packed-bed scrubber.

The main idea is to combine a physics-based model with machine learning models. The physics-based model estimates pressure drop based on process conditions and scrubber fouling. Machine learning models are then used to learn the remaining prediction error and correct the physics-based results.

The framework includes model training, validation, model selection, and final testing to compare the performance of different approaches.

## What Does Physics-Informed Hybrid ML Mean?

Physics-informed hybrid machine learning combines physical modeling with data-driven prediction. Instead of relying entirely on machine learning or using only a physics-based model, this approach uses the strengths of both.

In this project, the framework has three main components:

- **Physics-based model:** Estimates scrubber pressure drop using process conditions, gas properties, deposition, changes in bed porosity, and pressure-drop correlations.
- **Machine learning model:** Learns patterns in the relationship between process inputs and the prediction errors that remain after the physics-based calculation.
- **Hybrid model:** Combines the physics-based prediction with the machine learning correction to obtain the final pressure-drop estimate.

The hybrid prediction is calculated as follows:

`DeltaP_hybrid = DeltaP_physics + DeltaP_ML_correction`

where:

- `DeltaP_hybrid` is the final predicted pressure drop.
- `DeltaP_physics` is the pressure drop estimated by the physics-based model.
- `DeltaP_ML_correction` is the residual error predicted by the machine learning model.

The machine learning model is trained to predict the difference between the target pressure drop and the physics-based prediction. This correction is then added to the physics-based estimate.

In this implementation, the term *physics-informed* refers to using the physics-based prediction as the foundation of the hybrid model. The physical equations are not directly imposed as constraints on the neural network training process.

## Methodology

The framework consists of the following steps:

1. **Process simulation:** Generate process conditions over a 600-hour simulation period.
2. **Physics-based modeling:** Calculate pressure drop based on gas properties, deposition, porosity changes, the Ergun equation, and demister pressure drop.
3. **Feature engineering:** Prepare baseline and physics-informed input features.
4. **Machine learning:** Train and compare LSBoost, a custom regression tree ensemble, and neural network models.
5. **Hybrid modeling:** Predict the residual error of the physics-based model and use it to correct its predictions.
6. **Model validation and selection:** Compare model performance using R², RMSE, and MAE, select the best hybrid model based on validation R², and evaluate the models on a separate test set.

## Implementation

The framework is implemented in MATLAB. The main script includes the simulation, model training and evaluation procedures, and custom supporting functions.

The simulation generates 2,000 time points. After preprocessing and feature construction, the dataset contains 1,999 samples, divided into:

- Training set: 1,199 samples
- Validation set: 399 samples
- Test set: 401 samples

The models are evaluated using three performance metrics:

- **R²:** Measures how well the predictions explain the variation in the target values.
- **RMSE:** Measures the square root of the average squared prediction errors.
- **MAE:** Measures the average absolute prediction error.

Lower RMSE and MAE values indicate smaller prediction errors, while a higher R² generally indicates better predictive performance.

## Data

The current version generates synthetic process data internally and does not require an external dataset.

The simulated process conditions include inlet temperature, inlet concentration, and cooling temperature. The physics-based model estimates scrubber pressure drop using process properties and dynamic changes associated with deposition and bed porosity.

The pressure-drop target is constructed from the physics-based pressure-drop calculation, an additional concentration-dependent term, and synthetic sensor noise.

Therefore, the reported results represent model performance on simulated data, not validation against measured industrial data.

## Results

### Validation Performance

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Physics Only | 0.93065 | 51.085 | 42.902 |
| Baseline LSBoost | 0.97562 | 30.290 | 24.233 |
| Baseline Random Forest | 0.97632 | 29.854 | 23.958 |
| Baseline Neural Network | 0.95631 | 40.548 | 31.978 |
| Hybrid LSBoost | 0.97646 | 29.763 | 23.988 |
| Hybrid Random Forest | 0.97673 | 29.592 | 23.820 |
| **Hybrid Neural Network** | **0.97765** | **28.999** | **23.360** |

### Test Performance

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Physics Only | 0.92596 | 52.001 | 43.083 |
| Baseline LSBoost | 0.97069 | 32.718 | 25.708 |
| Baseline Random Forest | 0.97065 | 32.741 | 25.766 |
| Baseline Neural Network | 0.95212 | 41.818 | 32.161 |
| Hybrid LSBoost | 0.97045 | 32.851 | 25.729 |
| Hybrid Random Forest | 0.97190 | 32.034 | 25.181 |
| **Hybrid Neural Network** | **0.97307** | **31.361** | **24.345** |

### Discussion

The Hybrid Neural Network achieved the highest validation R² of 0.97765 and the highest test R² of 0.97307 among the evaluated models.

Compared with the physics-only model, the machine learning models achieved lower prediction errors on the simulated dataset. However, the improvement from hybrid modeling varied across the different algorithms. For example, Hybrid Random Forest performed better than its baseline counterpart on the test set, while Hybrid LSBoost performed slightly worse than Baseline LSBoost.

These results suggest that residual-based hybrid modeling can improve pressure-drop prediction under the simulated conditions, but its effectiveness depends on the selected machine learning method.

## How to Run

1. Download or clone this repository.
2. Open MATLAB and set the repository folder as the current working directory.
3. Run `Srubber_ML.m`.

The script generates the process data, trains and evaluates the models, displays diagnostic plots, and saves the output files.

No external dataset or separate custom function files are required for the current implementation.

## Output Files

After execution, the script generates:

- `Model_Validation_Performance.xlsx` — model performance on the validation set.
- `Model_Test_Performance.xlsx` — model performance on the test set.
- `Validated_HybridML_Workspace.mat` — saved MATLAB workspace variables.

Diagnostic plots are displayed during execution but are not included as repository files in this version.

## Limitations and Future Work

This version is a simulation-based implementation. The generated data are not measured industrial data, and the reported performance should be interpreted accordingly.

Further work may include:

- Validation using measured industrial process data.
- Evaluation using time-based data splitting.
- Further development of the physics-based model.
- Investigation of the effects of operating conditions and scrubber fouling on pressure-drop prediction.
- Addition of diagnostic plots and visual comparisons between the models.
