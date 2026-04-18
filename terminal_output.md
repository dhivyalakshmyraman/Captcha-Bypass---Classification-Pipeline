# Terminal Output - Pipeline Execution

```text
[SKIP] fp_pipeline.joblib already exists
[SKIP] net_pipeline.joblib already exists
[SKIP] wb_pipeline.joblib already exists
[SKIP] meta_model.joblib already exists
[SKIP] GAN generator for ks already exists
[SKIP] GAN generator for ms already exists
[SKIP] GAN generator for fp already exists
[SKIP] GAN generator for wb already exists

============================================================
  STEP: Adversarial augmentation for: ks
============================================================
--- Adversarial Augment for ks ---
Generated 4080, Oracle accepted 0 (conf > 0.6)
[SKIP] adversarial parquet for ms already exists

============================================================
  STEP: Adversarial augmentation for: fp
============================================================
--- Adversarial Augment for fp ---
Generated 1200, Oracle accepted 0 (conf > 0.6)
[SKIP] adversarial parquet for net already exists
[SKIP] adversarial parquet for wb already exists

============================================================
  STEP: Adversarial Retraining
============================================================
[SKIP] No adversarial data for ks
--- Retraining ms with Augmentation ---
Baseline AUC: 0.9562
.\venv\Scripts\python.exe : D:\Captcha Bypass Project\venv\Lib\site-packages\mlflow\tracking\_tracking_service\utils.py:184: FutureWarning: The filesystem tracking backend (e.g., './mlruns') is deprecated as of February 2026. Consider transitioning to a database backend (e.g., 'sqlite:///mlflow.db') to take advantage of the latest MLflow features. See https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store for migration guidance.
At line:1 char:1
+ .\venv\Scripts\python.exe -u run_all.py 2>&1 | Tee-Object -FilePath " ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (D:\Captcha Bypa...ation guidance.:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
  return FileStore(store_uri, store_uri)
Loaded existing features: 92304 rows
ROC-AUC: 0.9562
F1: 0.8645
Confusion Matrix:
[[10320  1617]
 [  326  6198]]
2026/04/05 04:39:20 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/05 04:39:20 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
Test set saved: D:\Captcha Bypass Project\data\processed\ms_test.parquet (18461 rows)
Model logged to MLflow run: e6e325207abd457fae7987ec0d7fc66a
New AUC on original test set: 0.9562
Registering new model! (AUC delta: 0.0000)
[SKIP] No adversarial data for fp
--- Retraining net with Augmentation ---
Baseline AUC: 0.8965
Loaded existing features: 29230 rows
Training IsolationForest on 12851 BENIGN rows
ROC-AUC (attack detection): 0.8965
2026/04/05 04:39:52 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/05 04:39:52 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
Test set saved: D:\Captcha Bypass Project\data\processed\net_test.parquet (5846 rows)
Model logged to MLflow run: f4e928427f234bdc88a0a51a648940ec
New AUC on original test set: 0.8965
Registering new model! (AUC delta: 0.0000)
--- Retraining wb with Augmentation ---
Baseline AUC: 1.0000
Loaded existing features: 152 rows
D:\Captcha Bypass Project\venv\Lib\site-packages\xgboost\training.py:200: UserWarning: [04:39:58] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:782: Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
D:\Captcha Bypass Project\venv\Lib\site-packages\xgboost\training.py:200: UserWarning: [04:39:58] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:782: Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
D:\Captcha Bypass Project\venv\Lib\site-packages\xgboost\training.py:200: UserWarning: [04:39:58] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:782: Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
D:\Captcha Bypass Project\venv\Lib\site-packages\xgboost\training.py:200: UserWarning: [04:39:58] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:782: Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
D:\Captcha Bypass Project\venv\Lib\site-packages\xgboost\training.py:200: UserWarning: [04:39:59] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:782: Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
ROC-AUC: 1.0000
F1: 1.0000
Confusion Matrix:
[[21  0]
 [ 0 10]]
2026/04/05 04:39:59 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/05 04:39:59 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
Test set saved: D:\Captcha Bypass Project\data\processed\wb_test.parquet (31 rows)
Model logged to MLflow run: 5191d089721d4b4ca55cbb441617b577
New AUC on original test set: 1.0000
Registering new model! (AUC delta: 0.0000)

============================================================
  STEP: Retraining Meta-Learner (post-GAN augmentation)
============================================================
Meta dataset: 31 rows, 5 features
Label distribution: {np.int64(0): np.int64(21), np.int64(1): np.int64(10)}
ROC-AUC: 1.0000
F1: 1.0000
Log Loss: 0.1081
Confusion Matrix:
[[5 0]
 [0 2]]
Model coefficients: {'p_ks': np.float64(1.2071340114979197), 'p_ms': np.float64(0.7362819344990116), 'p_fp': np.float64(1.2071340078840846), 'p_net': np.float64(-0.428245578835399), 'p_wb': np.float64(1.1980430528242927)}
2026/04/05 04:40:10 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/05 04:40:10 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
Meta model saved: D:\Captcha Bypass Project\data\processed\meta_model.joblib
Model logged to MLflow run: 247134a8ba794e93add64b3b54a75665

============================================================
  STEP: Running Deepchecks Validation
============================================================
INFO:src.validation.run_deepchecks:Checking ks data integrity...
D:\Captcha Bypass Project\venv\Lib\site-packages\deepchecks\core\serialization\dataframe\html.py:16: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import parse_version
WARNING:src.validation.run_deepchecks:Deepchecks failed ('max_error' is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get valid options.), falling back to pandas report
INFO:src.validation.run_deepchecks:Pandas report saved: D:\Captcha Bypass Project\reports\ks_data_validation.html
INFO:src.validation.run_deepchecks:Checking ms data integrity...
WARNING:src.validation.run_deepchecks:Deepchecks failed ('max_error' is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get valid options.), falling back to pandas report
INFO:src.validation.run_deepchecks:Pandas report saved: D:\Captcha Bypass Project\reports\ms_data_validation.html
INFO:src.validation.run_deepchecks:Checking fp data integrity...
WARNING:src.validation.run_deepchecks:Deepchecks failed ('max_error' is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get valid options.), falling back to pandas report
INFO:src.validation.run_deepchecks:Pandas report saved: D:\Captcha Bypass Project\reports\fp_data_validation.html
INFO:src.validation.run_deepchecks:Checking net data integrity...
WARNING:src.validation.run_deepchecks:Deepchecks failed ('max_error' is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get valid options.), falling back to pandas report
INFO:src.validation.run_deepchecks:Pandas report saved: D:\Captcha Bypass Project\reports\net_data_validation.html
INFO:src.validation.run_deepchecks:Checking wb data integrity...
WARNING:src.validation.run_deepchecks:Deepchecks failed ('max_error' is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get valid options.), falling back to pandas report
INFO:src.validation.run_deepchecks:Pandas report saved: D:\Captcha Bypass Project\reports\wb_data_validation.html

============================================================
  ALL PIPELINE STAGES COMPLETE
============================================================

Processed files in: D:\Captcha Bypass Project\data\processed
Reports in:         D:\Captcha Bypass Project\reports
MLflow tracking:    D:\Captcha Bypass Project\mlflow
```
