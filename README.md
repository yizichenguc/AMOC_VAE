This is repository for paper: 'Long-Term Slowdown in Atlantic Medirional Overturning Circulation'

Please ignore all '.ipynb_checkpoints' files.

* Folder 'input_data' includes all input data needed for 'pre_process.ipynb', 'model.ipynb' and 'MS_plots.ipynb'.
* File 'pre_process.ipynb' includes codes for data pre-processing steps.
* File 'model.ipynb' includes codes for training/testing data generation process, model building details and model inference.
* File 'MS_plots.ipynb' includes codes for plots generation that are shown in paper.
* Folder 'plots' includes all generated plots shown in paper.
* Folder 'hyper_tuning' includes hyper parameter tuning results shown in 'model.ipynb'.
* Folder 'model_output' will not be shown on git repository as the data are large and no need to include it.

#### Data Preprocessing

The file 'pre_process.ipynb' includes codes for data pre-processing steps. Input data files are located at `input_data` folder: `input_data/AMOC267_MAX_PD_monthly.nc` (CESM present-day run data for AMOC), `input_data/AMOC267_MAX_ENSEMBLE_monthly.nc` (CESM 28 model runs data for AMOC), `input_data/AMOC_index/PD_r1.txt`(CESM present-day run data for AMOC Index), `input_data/AMOC_index/ENSEMBLE_r1.txt` (CESM 28 model runs data for AMOC Index), `input_data/AMOC_model.mat` (RAPID AMOC Observation), `input_data/AMOC_index/obs_r1.txt` (NOAA AMOC Index Observation):

1. Pre-processing for AMOC CESM ensemble data: de-trend seasonality with monthly mean of present-day run data.
2. Pre-processing for AMOC Index CESM ensemble data: de-trend seasonality with monthly mean of present-day run data.
3. Pre-processing for AMOC Observation data: de-trend seasonality with self monthly mean.
4. Pre-processing for AMOC index Observation data: de-trend seasonality with self monthly mean.

The pre-processed data `dtdata_AMOC_Ensemble.csv`, `dtdata_AMOCindex_Ensemble.csv`, `dtdata_AMOC_Obs.csv`, `dtdata_AMOCindex_Obs.csv` are saved under folder `input_data/AMOC_140_years/`. The data will be used for model training and inference purpose.

#### DVAE Model Training

The file 'model.ipynb' includes codes for data generation process, model building/training and model inference.

1. The pre-processed data from `input_data/AMOC_140_years/` will be used to generate 6000 time segments data with 140 years length for model training purpose. 
2. The hyper-parameter tuning results, includes trained tensorflow model (file with suffix .hdf5) under different $\beta$ and generated hyper-parameter tuning plots (training history plots for each $\beta$, boxplots for MAE and Empirical Coverage) are saved under folder `hyper_tuning/AMOC_140_years/`. The grid results for hyper-parameter tuning are saved under folder `model_output` which is not shown on git repository as the data are large and no need to include it. For tensorflow seeds, original seeds = 1, and folder `new_seed_0` incldues results for seed = 10, folder `new_seed_1` incldues results for seed = 4.
3. After choosing best hyper-parameter, the model training process results (.hdf5 and training history plots) using training + validation data set are saved under folder `hyper_tuning/AMOC_140_years/`.
4. The model inference results are saved under folder `model_output/AMOC_140_years/beta_0.1`: selected inference results from testing data cases (`AMOC_input_test_0.npy` for large missing-gap AMOC data, `AMOCindex_input_test_0.npy` for corresponding AMOC Index data, `target_test_0.npy` for target AMOC long-term trend, `test_pred_0.npy` for 1000 trails prediction results at certain testing case), overall model inference performance (`emp_prob.npy` for empirical probability, `test_pred_mean.npy` for mean of 1000 prediction trails, `test_bias.npy` for diviation of mean from target, `test_pred_lower_q.npy` `test_pred_upper_q.npy` for lower 0.025 and upper 0.975 quantile of 1000 prediction results.), 1000 prediction trails for AMOC observation data (`Obs_pred.npy`).

### Plots

The file 'MS_plots.ipynb' includes codes for plots generation that are shown in paper.

1. The input data used are from folder `model_output/AMOC_140_years/beta_0.1`.
2. For plots of future century AMOC prediction, the input data, 12 CMIP6 ensemble data are from folder `input_data/AMOC_21C/`.
3. The generated plots are saved to folder `plots/AMOC_140_years/beta_0.1`.