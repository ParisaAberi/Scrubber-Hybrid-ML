%% ============================================================
% TWO-STAGE PHYSICS-INFORMED HYBRID ML FRAMEWORK
% WITH VALIDATION & MODEL SELECTION STAGE
%
% TOOLBOX-FREE VERSION (No Statistics Toolbox Required)
% ============================================================

clear;
clc;
close all;

rng(42);

fprintf('\n===============================================================\n');
fprintf(' TWO-STAGE PHYSICS-INFORMED HYBRID ML FRAMEWORK WITH VALIDATION\n');
fprintf('===============================================================\n\n');

%% ============================================================
% 1. SIMULATION SETTINGS
% ============================================================
fprintf('1. Initializing simulation settings...\n');
num_samples = 2000;
t_hours = linspace(0,600,num_samples)';
dt = t_hours(2)-t_hours(1);

%% ============================================================
% 2. PROCESS INPUTS
% ============================================================
fprintf('2. Generating realistic process conditions...\n');
t = t_hours;

T_in = 145 + 12*sin(2*pi*t/24) + 5*sin(2*pi*t/168) + 2.5*randn(num_samples,1);
T_in = min(max(T_in,90),190);

production_cycle = 0.5 + 0.5*sin(2*pi*(t-20)/120);
load_variation = 0.20*sin(2*pi*t/72);
random_variation = 0.04*randn(num_samples,1);
C_in = 1200 + 2800*production_cycle .* (1 + load_variation + random_variation);
C_in = max(C_in,300);

T_cool = 50 + 3*sin(2*pi*t/48) + 1.0*randn(num_samples,1);
T_cool = min(max(T_cool,40),65);

%% ============================================================
% 3. PHYSICAL MODEL PARAMETERS
% ============================================================
H_bed = 1.2; D_bed = 1.6;
cross_section_area = pi*D_bed^2/4;
V_bed = cross_section_area*H_bed;
dp = 0.003; eps_clean = 0.75; eps_min = 0.35;
R_air = 287.05; cp_air = 1005;
rho_ref = 1.225; mu_ref = 1.81e-5; T_ref_K = 288.15;
rho_plasticizer = 980; deposition_efficiency = 0.35; removal_rate = 0.0008;
T_cond_start = 82; T_cond_width = 8; T_sat_ref = 82; C_sat_ref = 2200; saturation_slope = 0.045;
UA_eff = 1800; Airflow_reference_CFM = 5000; Q_reference = Airflow_reference_CFM*0.000471947;
Fan_shutoff_pressure = 1800; Fan_reference_pressure = 550;
K_fan = (Fan_shutoff_pressure - Fan_reference_pressure) / Q_reference^2;
P_fixed = 300; P_demister_clean = 120; Q_demister_ref = Q_reference;
Ergun_constant_viscous = 150; Ergun_constant_inertial = 1.75;

%% ============================================================
% 4 & 5. PREALLOCATE & INITIAL CONDITIONS
% ============================================================
eps = zeros(num_samples,1); FoulingIndex = zeros(num_samples,1);
Airflow_CFM = zeros(num_samples,1); Airflow_m3_s = zeros(num_samples,1);
GasVelocity = zeros(num_samples,1); ReynoldsNumber = zeros(num_samples,1);
GasDensity = zeros(num_samples,1); GasViscosity = zeros(num_samples,1);
T_gas_out = zeros(num_samples,1); HeatTransferEffectiveness = zeros(num_samples,1);
CondensationFraction = zeros(num_samples,1); C_saturation = zeros(num_samples,1);
C_condensable = zeros(num_samples,1); m_dot_condensed = zeros(num_samples,1);
m_dot_deposited = zeros(num_samples,1); m_dot_removed = zeros(num_samples,1);
DepositedMass = zeros(num_samples,1); DepositedVolume = zeros(num_samples,1);
ViscousTerm = zeros(num_samples,1); InertialTerm = zeros(num_samples,1);
DeltaP_Ergun = zeros(num_samples,1); DeltaP_Demister = zeros(num_samples,1);
DeltaP_System = zeros(num_samples,1); DeltaP_Fan = zeros(num_samples,1);
DeltaP_Physics = zeros(num_samples,1); DeltaP_Scrubber = zeros(num_samples,1);
FanOperatingPointResidual = zeros(num_samples,1);

eps(1) = eps_clean; FoulingIndex(1) = 0; DepositedMass(1) = 0; DepositedVolume(1) = 0;
T_gas_mean_K_1 = ((T_in(1) + T_cool(1))/2) + 273.15;
GasDensity(1) = 101325/(R_air*T_gas_mean_K_1);
GasViscosity(1) = mu_ref*(T_gas_mean_K_1/T_ref_K)^0.7;

eps_current = eps(1); Q_low = 0.02; Q_high = Q_reference*1.40;
for iter = 1:60
    Q_test = 0.5*(Q_low + Q_high);
    v_test = Q_test/cross_section_area;
    viscous_test = Ergun_constant_viscous * ((1-eps_current)^2/eps_current^3) * (GasViscosity(1)*v_test/dp^2) * H_bed;
    inertial_test = Ergun_constant_inertial * ((1-eps_current)/eps_current^3) * (GasDensity(1)*v_test^2/dp) * H_bed;
    DeltaP_system_test = P_fixed + P_demister_clean * (Q_test/Q_demister_ref)^2 + viscous_test + inertial_test;
    DeltaP_fan_test = Fan_shutoff_pressure - K_fan*Q_test^2;
    if (DeltaP_fan_test - DeltaP_system_test) > 0, Q_low = Q_test; else, Q_high = Q_test; end
end
Q_current = 0.5*(Q_low + Q_high);
Airflow_m3_s(1) = Q_current; Airflow_CFM(1) = Q_current/0.000471947;
GasVelocity(1) = Q_current/cross_section_area;
ReynoldsNumber(1) = GasDensity(1) * GasVelocity(1) * dp / GasViscosity(1);

m_dot_air_1 = GasDensity(1)*Q_current;
NTU_effective_1 = UA_eff/(m_dot_air_1*cp_air);
HeatTransferEffectiveness(1) = 1-exp(-NTU_effective_1);
T_gas_out(1) = T_in(1) - HeatTransferEffectiveness(1)*(T_in(1)-T_cool(1));
C_saturation(1) = max(C_sat_ref * exp(saturation_slope *(T_gas_out(1)-T_sat_ref)), 50);
C_condensable(1) = max(0,C_in(1)-C_saturation(1));
CondensationFraction(1) = min(max(C_condensable(1)/max(C_in(1),1), 0),1);

ViscousTerm(1) = Ergun_constant_viscous * ((1-eps(1))^2/eps(1)^3) * (GasViscosity(1)*GasVelocity(1)/dp^2) * H_bed;
InertialTerm(1) = Ergun_constant_inertial * ((1-eps(1))/eps(1)^3) * (GasDensity(1)*GasVelocity(1)^2/dp) * H_bed;
DeltaP_Ergun(1) = ViscousTerm(1) + InertialTerm(1);
DeltaP_Demister(1) = P_demister_clean * (Q_current/Q_demister_ref)^2;
DeltaP_System(1) = P_fixed + DeltaP_Demister(1) + DeltaP_Ergun(1);
DeltaP_Fan(1) = Fan_shutoff_pressure - K_fan*Q_current^2;
FanOperatingPointResidual(1) = DeltaP_Fan(1) - DeltaP_System(1);
DeltaP_Physics(1) = DeltaP_System(1);
DeltaP_Scrubber(1) = DeltaP_Physics(1) + 0.015*C_in(1);

%% ============================================================
% 6. DYNAMIC PHYSICS SIMULATION
% ============================================================
fprintf('6. Running dynamic physics model...\n');
for i = 2:num_samples
    T_gas_mean_K = ((T_in(i-1)+T_cool(i-1))/2) + 273.15;
    GasDensity(i) = 101325 / (R_air*T_gas_mean_K);
    GasViscosity(i) = mu_ref * (T_gas_mean_K/T_ref_K)^0.7;
    eps_current = eps(i-1); Q_low = 0.02; Q_high = Q_reference*1.40;
    for iter = 1:60
        Q_test = 0.5*(Q_low+Q_high);
        v_test = Q_test/cross_section_area;
        viscous_test = Ergun_constant_viscous * ((1-eps_current)^2 / eps_current^3) * (GasViscosity(i)*v_test/dp^2) * H_bed;
        inertial_test = Ergun_constant_inertial * ((1-eps_current)/eps_current^3) * (GasDensity(i)*v_test^2/dp) * H_bed;
        residual = (Fan_shutoff_pressure - K_fan*Q_test^2) - (P_fixed + P_demister_clean * (Q_test/Q_demister_ref)^2 + viscous_test + inertial_test);
        if residual > 0, Q_low = Q_test; else, Q_high = Q_test; end
    end
    Q_current = 0.5*(Q_low+Q_high);
    Airflow_m3_s(i) = Q_current; Airflow_CFM(i) = Q_current/0.000471947;
    GasVelocity(i) = Q_current/cross_section_area;
    ReynoldsNumber(i) = GasDensity(i) * GasVelocity(i) * dp / GasViscosity(i);

    m_dot_air = GasDensity(i)*Q_current;
    NTU_effective = UA_eff/(m_dot_air*cp_air);
    HeatTransferEffectiveness(i) = 1-exp(-NTU_effective);
    T_gas_out(i) = T_in(i-1) - HeatTransferEffectiveness(i)*(T_in(i-1)-T_cool(i-1));
    C_saturation(i) = max(C_sat_ref * exp(saturation_slope *(T_gas_out(i)-T_sat_ref)), 50);
    excess_concentration = max(0, C_in(i-1)-C_saturation(i));
    CondensationFraction(i) = min(max(excess_concentration/max(C_in(i-1),1),0),1);
    C_condensable(i) = excess_concentration;

    m_dot_condensed(i) = C_condensable(i) * Q_current / 1e6 * 3600;
    m_dot_deposited(i) = deposition_efficiency * m_dot_condensed(i);
    m_dot_removed(i) = removal_rate * DepositedMass(i-1);
    DepositedMass(i) = max(DepositedMass(i-1) + (m_dot_deposited(i) - m_dot_removed(i))*dt, 0);
    DepositedVolume(i) = DepositedMass(i) / rho_plasticizer;

    fouling_loading = DepositedMass(i) / (rho_plasticizer*V_bed);
    eps(i) = min(max(eps_min + (eps_clean-eps_min) * exp(-2.5*fouling_loading), eps_min), eps_clean);
    FoulingIndex(i) = min(max((eps_clean-eps(i)) / (eps_clean-eps_min), 0), 1);

    v = GasVelocity(i);
    ViscousTerm(i) = Ergun_constant_viscous * ((1-eps(i))^2 / eps(i)^3) * (GasViscosity(i)*v/dp^2) * H_bed;
    InertialTerm(i) = Ergun_constant_inertial * ((1-eps(i)) / eps(i)^3) * (GasDensity(i)*v^2/dp) * H_bed;
    DeltaP_Ergun(i) = ViscousTerm(i) + InertialTerm(i);
    DeltaP_Demister(i) = P_demister_clean * (Q_current/Q_demister_ref)^2;
    DeltaP_System(i) = P_fixed + DeltaP_Demister(i) + DeltaP_Ergun(i);
    DeltaP_Fan(i) = Fan_shutoff_pressure - K_fan*Q_current^2;
    FanOperatingPointResidual(i) = DeltaP_Fan(i) - DeltaP_System(i);
    DeltaP_Physics(i) = DeltaP_System(i);
    DeltaP_Scrubber(i) = DeltaP_Physics(i) + 0.015*C_in(i);
end

%% ============================================================
% 7 TO 13. NOISE, DIAGNOSTICS & FEATURE MATRIX
% ============================================================
sensor_noise_level = 0.025;
DeltaP_Scrubber_noisy = DeltaP_Scrubber .* (1 + sensor_noise_level*randn(num_samples,1));
PhysicsResidual = DeltaP_Scrubber_noisy - DeltaP_Physics;

OperatingHours = t_hours;
dataset = table(OperatingHours, T_in, C_in, T_cool, T_gas_out, C_saturation, ...
    C_condensable, CondensationFraction, eps, FoulingIndex, DepositedMass, ...
    Airflow_CFM, GasVelocity, GasDensity, GasViscosity, ReynoldsNumber, ...
    DeltaP_Ergun, DeltaP_Demister, DeltaP_System, DeltaP_Fan, ...
    FanOperatingPointResidual, DeltaP_Physics, PhysicsResidual, DeltaP_Scrubber_noisy, ...
    'VariableNames', {'OperatingHours','InletTemp_C','Concentration_mg_m3','CoolingTemp_C', ...
    'GasOutletTemp_C','SaturationConcentration_mg_m3','CondensableConcentration_mg_m3', ...
    'CondensationFraction','Porosity','FoulingIndex','DepositedMass_kg','Airflow_CFM', ...
    'GasVelocity_m_s','GasDensity_kg_m3','GasViscosity_Pa_s','ReynoldsNumber', ...
    'DeltaP_Ergun_Pa','DeltaP_Demister_Pa','DeltaP_System_Pa','DeltaP_Fan_Pa', ...
    'FanOperatingPointResidual_Pa','DeltaP_Physics_Pa','PhysicsResidual_Pa','DeltaP_Scrubber_Pa'});

dataset.Lag_Airflow = [NaN; dataset.Airflow_CFM(1:end-1)];
dataset.Lag_InletTemp = [NaN; dataset.InletTemp_C(1:end-1)];
dataset.Lag_Concentration = [NaN; dataset.Concentration_mg_m3(1:end-1)];
dataset.Lag_DeltaP_Physics = [NaN; dataset.DeltaP_Physics_Pa(1:end-1)];
dataset(1,:) = [];
data = dataset;

Y = data.DeltaP_Scrubber_Pa;
Y_physics = data.DeltaP_Physics_Pa;
Y_correction = Y - Y_physics;

baseVars = {'OperatingHours','InletTemp_C','Concentration_mg_m3','CoolingTemp_C','Airflow_CFM','Lag_Airflow','Lag_InletTemp','Lag_Concentration'};
physVars = {'OperatingHours','InletTemp_C','Concentration_mg_m3','CoolingTemp_C','Airflow_CFM','Lag_Airflow','Lag_InletTemp','Lag_Concentration','DeltaP_Physics_Pa','FoulingIndex','GasVelocity_m_s','ReynoldsNumber','Lag_DeltaP_Physics'};

X_baseline = data{:,baseVars};
X_physics = data{:,physVars};

validRows = all(isfinite(X_baseline),2) & all(isfinite(X_physics),2) & isfinite(Y) & isfinite(Y_correction);
X_baseline = X_baseline(validRows,:);
X_physics = X_physics(validRows,:);
Y = Y(validRows);
Y_physics = Y_physics(validRows);
Y_correction = Y_correction(validRows);
data = data(validRows,:);

%% ============================================================
% 16. THREE-WAY SPLIT: TRAIN (60%), VALIDATION (20%), TEST (20%)
% ============================================================
fprintf('16. Creating Train / Validation / Test split...\n');
N = size(X_baseline,1);
rng(42);
randomOrder = randperm(N);

nTrain = floor(0.60 * N);
nVal   = floor(0.20 * N);

trainIdx = false(N,1);
valIdx   = false(N,1);
testIdx  = false(N,1);

trainIdx(randomOrder(1:nTrain)) = true;
valIdx(randomOrder(nTrain+1 : nTrain+nVal)) = true;
testIdx(randomOrder(nTrain+nVal+1 : end)) = true;

% Split Features
Xb_train = X_baseline(trainIdx,:); Xb_val = X_baseline(valIdx,:); Xb_test = X_baseline(testIdx,:);
Xp_train = X_physics(trainIdx,:);  Xp_val = X_physics(valIdx,:);  Xp_test = X_physics(testIdx,:);

% Split Targets
Y_train = Y(trainIdx);             Y_val = Y(valIdx);             Y_test = Y(testIdx);
Ycorr_train = Y_correction(trainIdx); Ycorr_val = Y_correction(valIdx); Ycorr_test = Y_correction(testIdx);
Yphysics_val = Y_physics(valIdx);  Yphysics_test = Y_physics(testIdx);

fprintf('Training samples   : %d\n', sum(trainIdx));
fprintf('Validation samples : %d\n', sum(valIdx));
fprintf('Testing samples    : %d\n\n', sum(testIdx));

%% ============================================================
% 17. MODEL TRAINING & VALIDATION STAGE
% ============================================================
fprintf('===============================================================\n');
fprintf(' STAGE 1: TRAINING & VALIDATION\n');
fprintf('===============================================================\n');

% -------------------------------------------------------------
% 17.1 Physics-Only Model
% -------------------------------------------------------------
Ypred_phys_val = Yphysics_val;
[R2_phys_val, RMSE_phys_val, MAE_phys_val] = calc_metrics(Y_val, Ypred_phys_val);

% -------------------------------------------------------------
% 17.2 Baseline LSBoost
% -------------------------------------------------------------
boostParams.numTrees = 100; boostParams.learnRate = 0.08;
boostParams.maxDepth = 3;   boostParams.minLeaf = 5; boostParams.numCandidateSplits = 10;

model_bLSBoost = custom_boost_train(Xb_train, Y_train, boostParams);
Ypred_bLSBoost_val = custom_boost_predict(model_bLSBoost, Xb_val);
[R2_bLSBoost_val, RMSE_bLSBoost_val, MAE_bLSBoost_val] = calc_metrics(Y_val, Ypred_bLSBoost_val);

% -------------------------------------------------------------
% 17.3 Baseline Random Forest
% -------------------------------------------------------------
rfParams.numTrees = 80; rfParams.maxDepth = 6; rfParams.minLeaf = 5;
rfParams.numCandidateSplits = 10; rfParams.featureFraction = 0.70;

model_bRF = custom_rf_train(Xb_train, Y_train, rfParams);
Ypred_bRF_val = custom_rf_predict(model_bRF, Xb_val);
[R2_bRF_val, RMSE_bRF_val, MAE_bRF_val] = calc_metrics(Y_val, Ypred_bRF_val);

% -------------------------------------------------------------
% 17.4 Baseline Neural Network
% -------------------------------------------------------------
mu_Xb = mean(Xb_train,1); sigma_Xb = std(Xb_train,0,1); sigma_Xb(sigma_Xb==0) = 1;
Xb_train_norm = (Xb_train - mu_Xb) ./ sigma_Xb;
Xb_val_norm   = (Xb_val - mu_Xb) ./ sigma_Xb;
Xb_test_norm  = (Xb_test - mu_Xb) ./ sigma_Xb;

nnParams.hidden1 = 32; nnParams.hidden2 = 16; nnParams.learningRate = 0.001;
nnParams.epochs = 800; nnParams.lambda = 0.001; nnParams.verbose = 0;

model_bNN = custom_nn_train(Xb_train_norm, Y_train, nnParams);
Ypred_bNN_val = custom_nn_predict(model_bNN, Xb_val_norm);
[R2_bNN_val, RMSE_bNN_val, MAE_bNN_val] = calc_metrics(Y_val, Ypred_bNN_val);

% -------------------------------------------------------------
% 17.5 Hybrid LSBoost
% -------------------------------------------------------------
model_hLSBoost = custom_boost_train(Xp_train, Ycorr_train, boostParams);
Ycorr_hLSBoost_val = custom_boost_predict(model_hLSBoost, Xp_val);
Ypred_hLSBoost_val = Yphysics_val + Ycorr_hLSBoost_val;
[R2_hLSBoost_val, RMSE_hLSBoost_val, MAE_hLSBoost_val] = calc_metrics(Y_val, Ypred_hLSBoost_val);

% -------------------------------------------------------------
% 17.6 Hybrid Random Forest
% -------------------------------------------------------------
model_hRF = custom_rf_train(Xp_train, Ycorr_train, rfParams);
Ycorr_hRF_val = custom_rf_predict(model_hRF, Xp_val);
Ypred_hRF_val = Yphysics_val + Ycorr_hRF_val;
[R2_hRF_val, RMSE_hRF_val, MAE_hRF_val] = calc_metrics(Y_val, Ypred_hRF_val);

% -------------------------------------------------------------
% 17.7 Hybrid Neural Network
% -------------------------------------------------------------
mu_Xp = mean(Xp_train,1); sigma_Xp = std(Xp_train,0,1); sigma_Xp(sigma_Xp==0) = 1;
Xp_train_norm = (Xp_train - mu_Xp) ./ sigma_Xp;
Xp_val_norm   = (Xp_val - mu_Xp) ./ sigma_Xp;
Xp_test_norm  = (Xp_test - mu_Xp) ./ sigma_Xp;

model_hNN = custom_nn_train(Xp_train_norm, Ycorr_train, nnParams);
Ycorr_hNN_val = custom_nn_predict(model_hNN, Xp_val_norm);
Ypred_hNN_val = Yphysics_val + Ycorr_hNN_val;
[R2_hNN_val, RMSE_hNN_val, MAE_hNN_val] = calc_metrics(Y_val, Ypred_hNN_val);

% Display Validation Table
Val_Models = {'Physics Only'; 'Baseline LSBoost'; 'Baseline Random Forest'; 'Baseline Neural Network'; ...
              'Hybrid LSBoost'; 'Hybrid Random Forest'; 'Hybrid Neural Network'};
Val_R2   = [R2_phys_val; R2_bLSBoost_val; R2_bRF_val; R2_bNN_val; R2_hLSBoost_val; R2_hRF_val; R2_hNN_val];
Val_RMSE = [RMSE_phys_val; RMSE_bLSBoost_val; RMSE_bRF_val; RMSE_bNN_val; RMSE_hLSBoost_val; RMSE_hRF_val; RMSE_hNN_val];
Val_MAE  = [MAE_phys_val; MAE_bLSBoost_val; MAE_bRF_val; MAE_bNN_val; MAE_hLSBoost_val; MAE_hRF_val; MAE_hNN_val];

ValResultsTable = table(Val_Models, Val_R2, Val_RMSE, Val_MAE, 'VariableNames', {'Model','R2_Val','RMSE_Val','MAE_Val'});
fprintf('\n--- VALIDATION RESULTS ---\n');
disp(ValResultsTable);

%% ============================================================
% 18. AUTOMATED BEST MODEL SELECTION BASED ON VALIDATION
% ============================================================
hybrid_Val_R2 = [R2_hLSBoost_val, R2_hRF_val, R2_hNN_val];
hybridNames   = {'Hybrid LSBoost', 'Hybrid Random Forest', 'Hybrid Neural Network'};
[bestValR2, bestIdx] = max(hybrid_Val_R2);

fprintf('===============================================================\n');
fprintf(' BEST MODEL SELECTION\n');
fprintf(' Selected Model based on Validation R2: %s (R2_Val = %.6f)\n', hybridNames{bestIdx}, bestValR2);
fprintf('===============================================================\n\n');

%% ============================================================
% 19. FINAL TESTING STAGE (UNSEEN DATA EVALUATION)
% ============================================================
fprintf('===============================================================\n');
fprintf(' STAGE 2: FINAL TEST EVALUATION (UNSEEN TEST DATA)\n');
fprintf('===============================================================\n');

% Physics Only Test
Ypred_phys_test = Yphysics_test;
[R2_phys_test, RMSE_phys_test, MAE_phys_test] = calc_metrics(Y_test, Ypred_phys_test);

% Baseline ML Test
Ypred_bLSBoost_test = custom_boost_predict(model_bLSBoost, Xb_test);
[R2_bLSBoost_test, RMSE_bLSBoost_test, MAE_bLSBoost_test] = calc_metrics(Y_test, Ypred_bLSBoost_test);

Ypred_bRF_test = custom_rf_predict(model_bRF, Xb_test);
[R2_bRF_test, RMSE_bRF_test, MAE_bRF_test] = calc_metrics(Y_test, Ypred_bRF_test);

Ypred_bNN_test = custom_nn_predict(model_bNN, Xb_test_norm);
[R2_bNN_test, RMSE_bNN_test, MAE_bNN_test] = calc_metrics(Y_test, Ypred_bNN_test);

% Hybrid ML Test
Ypred_hLSBoost_test = Yphysics_test + custom_boost_predict(model_hLSBoost, Xp_test);
[R2_hLSBoost_test, RMSE_hLSBoost_test, MAE_hLSBoost_test] = calc_metrics(Y_test, Ypred_hLSBoost_test);

Ypred_hRF_test = Yphysics_test + custom_rf_predict(model_hRF, Xp_test);
[R2_hRF_test, RMSE_hRF_test, MAE_hRF_test] = calc_metrics(Y_test, Ypred_hRF_test);

Ypred_hNN_test = Yphysics_test + custom_nn_predict(model_hNN, Xp_test_norm);
[R2_hNN_test, RMSE_hNN_test, MAE_hNN_test] = calc_metrics(Y_test, Ypred_hNN_test);

% Final Test Results Table
Test_R2   = [R2_phys_test; R2_bLSBoost_test; R2_bRF_test; R2_bNN_test; R2_hLSBoost_test; R2_hRF_test; R2_hNN_test];
Test_RMSE = [RMSE_phys_test; RMSE_bLSBoost_test; RMSE_bRF_test; RMSE_bNN_test; RMSE_hLSBoost_test; RMSE_hRF_test; RMSE_hNN_test];
Test_MAE  = [MAE_phys_test; MAE_bLSBoost_test; MAE_bRF_test; MAE_bNN_test; MAE_hLSBoost_test; MAE_hRF_test; MAE_hNN_test];

TestResultsTable = table(Val_Models, Test_R2, Test_RMSE, Test_MAE, 'VariableNames', {'Model','R2_Test','RMSE_Test','MAE_Test'});
fprintf('\n--- FINAL TEST RESULTS ---\n');
disp(TestResultsTable);

%% ============================================================
% 20. PLOTS AND EXPORT
% ============================================================
figure;
bar([Val_R2, Test_R2]);
set(gca, 'XTickLabel', Val_Models, 'XTickLabelRotation', 30);
ylabel('R^2 Score');
legend({'Validation Set', 'Test Set'}, 'Location', 'best');
title('Model Performance: Validation vs Test Sets');
grid on;

figure;
plot(Y_test, 'k', 'LineWidth', 1.3); hold on;
plot(Ypred_hLSBoost_test, 'r--', 'LineWidth', 1.1);
plot(Ypred_bLSBoost_test, 'b-.', 'LineWidth', 1.1);
plot(Yphysics_test, 'g:', 'LineWidth', 1.2);
xlabel('Test Sample Index'); ylabel('\DeltaP (Pa)');
title('Test Set Predictions vs Actual Measured \DeltaP');
legend({'Measured', 'Hybrid LSBoost', 'Baseline LSBoost', 'Physics Only'}, 'Location', 'best');
grid on;

% Save files
writetable(ValResultsTable, 'Model_Validation_Performance.xlsx');
writetable(TestResultsTable, 'Model_Test_Performance.xlsx');
save('Validated_HybridML_Workspace.mat');

fprintf('\nValidation, Model Selection, and Testing workflow executed successfully.\n');
%% ============================================================
% 21. ADVANCED VISUALIZATION & DIAGNOSTICS SUITE
% ============================================================
fprintf('21. Generating comprehensive diagnostic & comparison plots...\n');

set(groot, 'DefaultAxesFontName', 'Segoe UI');
set(groot, 'DefaultAxesFontSize', 10);

%% Figure 1: Validation vs Test Metrics Comparison (Bar Chart Matrix)
figure('Name', 'Model Performance Metrics', 'Position', [100, 100, 900, 500]);
subplot(1,3,1);
bar([Val_R2, Test_R2]);
title('R^2 Score'); ylabel('R^2'); grid on;
set(gca, 'XTickLabel', Val_Models, 'XTickLabelRotation', 45);

subplot(1,3,2);
bar([Val_RMSE, Test_RMSE]);
title('RMSE (Pa)'); ylabel('Error (Pa)'); grid on;
set(gca, 'XTickLabel', Val_Models, 'XTickLabelRotation', 45);

subplot(1,3,3);
bar([Val_MAE, Test_MAE]);
title('MAE (Pa)'); ylabel('Error (Pa)'); grid on;
set(gca, 'XTickLabel', Val_Models, 'XTickLabelRotation', 45);
legend({'Validation Set', 'Test Set'}, 'Position', [0.42 0.02 0.2 0.04], 'Orientation', 'horizontal');

%% Figure 2: Parity Plots (Actual vs Predicted) on Test Set
figure('Name', 'Parity Plots - Test Set', 'Position', [150, 150, 1000, 450]);

subplot(1,3,1);
scatter(Y_test, Yphysics_test, 15, 'g', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
plot([min(Y_test) max(Y_test)], [min(Y_test) max(Y_test)], 'k--', 'LineWidth', 1.5);
title(sprintf('Physics-Only (R^2 = %.3f)', R2_phys_test));
xlabel('Measured \DeltaP (Pa)'); ylabel('Predicted \DeltaP (Pa)'); grid on; axis square;

subplot(1,3,2);
scatter(Y_test, Ypred_bLSBoost_test, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
plot([min(Y_test) max(Y_test)], [min(Y_test) max(Y_test)], 'k--', 'LineWidth', 1.5);
title(sprintf('Baseline Neural Network (R^2 = %.3f)', R2_bNN_test));
xlabel('Measured \DeltaP (Pa)'); ylabel('Predicted \DeltaP (Pa)'); grid on; axis square;

subplot(1,3,3);
scatter(Y_test, Ypred_hLSBoost_test, 15, 'r', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
plot([min(Y_test) max(Y_test)], [min(Y_test) max(Y_test)], 'k--', 'LineWidth', 1.5);
title(sprintf('Hybrid Neural Network (R^2 = %.3f)', R2_hNN_test));
xlabel('Measured \DeltaP (Pa)'); ylabel('Predicted \DeltaP (Pa)'); grid on; axis square;

%% Figure 3: Residual Analysis & Error Distribution
res_phys = Y_test - Yphysics_test;
res_bLSBoost = Y_test - Ypred_bLSBoost_test;
res_hLSBoost = Y_test - Ypred_hLSBoost_test;

figure('Name', 'Residual Diagnostics', 'Position', [200, 200, 950, 450]);

subplot(1,2,1);
plot(res_phys, 'g', 'LineWidth', 0.8); hold on;
plot(res_bLSBoost, 'b', 'LineWidth', 0.8);
plot(res_hLSBoost, 'r', 'LineWidth', 1.0);
title('Prediction Residuals on Test Set');
xlabel('Test Sample Index'); ylabel('Residual Error (Pa)');
legend({'Physics Error', 'Baseline LSBoost Error', 'Hybrid LSBoost Error'}, 'Location', 'best'); grid on;

subplot(1,2,2);
histogram(res_phys, 30, 'FaceColor', 'g', 'FaceAlpha', 0.4); hold on;
histogram(res_bLSBoost, 30, 'FaceColor', 'b', 'FaceAlpha', 0.4);
histogram(res_hLSBoost, 30, 'FaceColor', 'r', 'FaceAlpha', 0.6);
title('Error Histogram Comparison');
xlabel('Residual Error (Pa)'); ylabel('Frequency');
legend({'Physics', 'Baseline LSBoost', 'Hybrid LSBoost'}, 'Location', 'best'); grid on;

%% Figure 4: Physical Dynamics & Fouling Tracking (Time-Series)
figure('Name', 'System Physical Variables Dynamic Response', 'Position', [250, 250, 900, 600]);

subplot(3,1,1);
plot(data.OperatingHours, data.FoulingIndex, 'k-', 'LineWidth', 1.5);
title('Bed Fouling Index Progress Over Time');
ylabel('Fouling Index (-)'); grid on;

subplot(3,1,2);
plot(data.OperatingHours, data.Porosity, 'm-', 'LineWidth', 1.5);
title('Bed Porosity (\epsilon) Decay Due to Plasticizer Deposition');
ylabel('Porosity (\epsilon)'); grid on;

subplot(3,1,3);
plot(data.OperatingHours, data.DeltaP_Ergun_Pa, 'c-', 'LineWidth', 1.2); hold on;
plot(data.OperatingHours, data.DeltaP_Demister_Pa, 'm--', 'LineWidth', 1.2);
plot(data.OperatingHours, data.DeltaP_Scrubber_Pa, 'k-', 'LineWidth', 1.5);
title('Scrubber Pressure Drop Component Breakdown');
xlabel('Operating Hours (h)'); ylabel('\DeltaP (Pa)');
legend({'Packed Bed (Ergun)', 'Demister Pad', 'Total Measured \DeltaP'}, 'Location', 'northwest'); grid on;

%% Figure 5: Feature Importance Architecture (Hybrid Model)
figure('Name', 'Feature Importance', 'Position', [300, 300, 750, 450]);
barh(model_hLSBoost.featureImportance, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'YTick', 1:length(physVars), 'YTickLabel', physVars);
xlabel('Normalized Importance Score');
title('Feature Importance Breakdown (Hybrid LSBoost Model)');
grid on;

fprintf('All diagnostic plots rendered successfully.\n');

%% ============================================================
% LOCAL FUNCTIONS (UNCHANGED NUMERICAL ENGINES)
% ============================================================

function [R2,RMSE,MAE] = calc_metrics(Ytrue,Ypred)
    Ytrue = Ytrue(:); Ypred = Ypred(:);
    residuals = Ytrue-Ypred;
    SSE = sum(residuals.^2);
    SST = sum((Ytrue-mean(Ytrue)).^2);
    if SST == 0, R2 = 0; else, R2 = 1-SSE/SST; end
    RMSE = sqrt(mean(residuals.^2));
    MAE = mean(abs(residuals));
end

function model = custom_boost_train(X,Y,params)
    [N,P] = size(X);
    numTrees = params.numTrees; learningRate = params.learnRate;
    prediction = mean(Y)*ones(N,1);
    trees = cell(numTrees,1); importance = zeros(1,P); treeWeights = zeros(numTrees,1);
    for b = 1:numTrees
        residual = Y-prediction;
        tree = train_regression_tree(X, residual, params.maxDepth, params.minLeaf, params.numCandidateSplits, 1:P);
        treePrediction = predict_regression_tree(tree,X);
        prediction = prediction + learningRate*treePrediction;
        trees{b} = tree; treeWeights(b) = learningRate;
        importance = importance + tree.importance;
    end
    if sum(importance)>0, importance = importance/sum(importance); end
    model.basePrediction = mean(Y); model.trees = trees; model.treeWeights = treeWeights; model.featureImportance = importance;
end

function Ypred = custom_boost_predict(model,X)
    N = size(X,1); Ypred = model.basePrediction*ones(N,1);
    for b = 1:length(model.trees)
        Ypred = Ypred + model.treeWeights(b)*predict_regression_tree(model.trees{b},X);
    end
end

function model = custom_rf_train(X,Y,params)
    [N,P] = size(X); numTrees = params.numTrees;
    trees = cell(numTrees,1); importance = zeros(1,P);
    for b = 1:numTrees
        bootstrapIdx = randi(N,N,1);
        numFeatures = max(1,round(params.featureFraction*P));
        featurePool = randperm(P,numFeatures);
        tree = train_regression_tree(X(bootstrapIdx,:), Y(bootstrapIdx), params.maxDepth, params.minLeaf, params.numCandidateSplits, featurePool);
        trees{b} = tree; importance = importance + tree.importance;
    end
    if sum(importance)>0, importance = importance/sum(importance); end
    model.trees = trees; model.featureImportance = importance;
end

function Ypred = custom_rf_predict(model,X)
    N = size(X,1); numTrees = length(model.trees); allPred = zeros(N,numTrees);
    for b = 1:numTrees, allPred(:,b) = predict_regression_tree(model.trees{b},X); end
    Ypred = mean(allPred,2);
end

function tree = train_regression_tree(X,Y,maxDepth,minLeaf,numCandidateSplits,featurePool)
    [node,importance] = build_tree_node(X,Y,0,maxDepth,minLeaf,numCandidateSplits,featurePool);
    tree = node; tree.importance = importance;
end

function [node,importance] = build_tree_node(X,Y,depth,maxDepth,minLeaf,numCandidateSplits,featurePool)
    [N,P] = size(X); importance = zeros(1,P);
    node.isLeaf = true; node.prediction = mean(Y); node.feature = 0; node.threshold = 0; node.left = []; node.right = [];
    if N <= 2*minLeaf || depth >= maxDepth || var(Y) < 1e-12, return; end
    bestSSE = inf; bestFeature = 0; bestThreshold = 0; parentSSE = sum((Y-mean(Y)).^2);
    featureOrder = featurePool; if isempty(featureOrder), featureOrder = 1:P; end
    featureOrder = featureOrder(randperm(length(featureOrder)));
    for f = featureOrder
        x = X(:,f); finiteMask = isfinite(x);
        if sum(finiteMask) < 2*minLeaf, continue; end
        xFinite = x(finiteMask); xmin = min(xFinite); xmax = max(xFinite);
        if xmin == xmax, continue; end
        q = linspace(0.10,0.90,numCandidateSplits); thresholds = zeros(length(q),1);
        sortedX = sort(xFinite);
        for k = 1:length(q), thresholds(k) = sortedX(max(1,min(length(sortedX), round(q(k)*length(sortedX))))); end
        thresholds = unique(thresholds);
        for s = 1:length(thresholds)
            threshold = thresholds(s); leftIdx = x <= threshold; rightIdx = x > threshold;
            if sum(leftIdx) < minLeaf || sum(rightIdx) < minLeaf, continue; end
            SSEleft = sum((Y(leftIdx)-mean(Y(leftIdx))).^2); SSEright = sum((Y(rightIdx)-mean(Y(rightIdx))).^2);
            if (SSEleft+SSEright) < bestSSE, bestSSE = SSEleft+SSEright; bestFeature = f; bestThreshold = threshold; end
        end
    end
    if bestFeature == 0 || (parentSSE-bestSSE) <= 1e-12, return; end
    leftIdx = X(:,bestFeature) <= bestThreshold; rightIdx = X(:,bestFeature) > bestThreshold;
    if sum(leftIdx)<minLeaf || sum(rightIdx)<minLeaf, return; end
    node.isLeaf = false; node.feature = bestFeature; node.threshold = bestThreshold;
    importance(bestFeature) = importance(bestFeature)+(parentSSE-bestSSE);
    [leftNode,leftImp] = build_tree_node(X(leftIdx,:), Y(leftIdx), depth+1, maxDepth, minLeaf, numCandidateSplits, featurePool);
    [rightNode,rightImp] = build_tree_node(X(rightIdx,:), Y(rightIdx), depth+1, maxDepth, minLeaf, numCandidateSplits, featurePool);
    node.left = leftNode; node.right = rightNode;
    importance = importance + leftImp + rightImp;
end

function Ypred = predict_regression_tree(tree,X)
    N = size(X,1); Ypred = zeros(N,1);
    for i = 1:N, Ypred(i) = predict_single_tree_node(tree,X(i,:)); end
end

function value = predict_single_tree_node(node,x)
    if node.isLeaf, value = node.prediction; return; end
    if x(node.feature) <= node.threshold, value = predict_single_tree_node(node.left,x);
    else, value = predict_single_tree_node(node.right,x); end
end

function model = custom_nn_train(X,Y,params)
    [N,P] = size(X); H1 = params.hidden1; H2 = params.hidden2; lr = params.learningRate; epochs = params.epochs; lambda = params.lambda;
    yMean = mean(Y); yStd = std(Y); if yStd == 0, yStd = 1; end
    Yn = (Y-yMean)/yStd;
    W1 = randn(P,H1)*sqrt(2/(P+H1)); b1 = zeros(1,H1);
    W2 = randn(H1,H2)*sqrt(2/(H1+H2)); b2 = zeros(1,H2);
    W3 = randn(H2,1)*sqrt(2/(H2+1)); b3 = 0;
    for epoch = 1:epochs
        Z1 = X*W1 + b1; A1 = tanh(Z1);
        Z2 = A1*W2 + b2; A2 = tanh(Z2);
        Yhat = A2*W3 + b3;
        error = Yhat-Yn;
        dY = (2/N)*error;
        dW3 = A2'*dY + lambda*W3; db3 = sum(dY,1);
        dA2 = dY*W3'; dZ2 = dA2.*(1-A2.^2);
        dW2 = A1'*dZ2 + lambda*W2; db2 = sum(dZ2,1);
        dA1 = dZ2*W2'; dZ1 = dA1.*(1-A1.^2);
        dW1 = X'*dZ1 + lambda*W1; db1 = sum(dZ1,1);
        W3 = W3-lr*dW3; b3 = b3-lr*db3;
        W2 = W2-lr*dW2; b2 = b2-lr*db2;
        W1 = W1-lr*dW1; b1 = b1-lr*db1;
    end
    model.W1 = W1; model.b1 = b1; model.W2 = W2; model.b2 = b2; model.W3 = W3; model.b3 = b3;
    model.yMean = yMean; model.yStd = yStd;
end

function Ypred = custom_nn_predict(model,X)
    Z1 = X*model.W1 + model.b1; A1 = tanh(Z1);
    Z2 = A1*model.W2 + model.b2; A2 = tanh(Z2);
    Yn = A2*model.W3 + model.b3;
    Ypred = Yn*model.yStd + model.yMean;
end