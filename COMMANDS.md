## Copy YooChoose/Retailrocket datasets from unzipped `SA2C_code.zip`

```bash
SA2C_code_unzip_dir=/raid/data_share/antonchernov/transformer_benchmark_rl/data/raw/SA2C_code
rsync -a "${SA2C_code_unzip_dir}/Kaggle/data/" "Kaggle/data/"
rsync -a "${SA2C_code_unzip_dir}/RC15/data/" "RC15/data/"
```

## Run Torch (local .venv + uv) — SASRec only

```bash
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
```

## Plot test results (clicks + purchase ndcg@10)

- Writes combined plots (rectools + torch) under `results/plots/{dataset_name}/test_results.png` and, for `persrec_tc5_*`, `results/plots/{dataset_name}/{eval_scheme}/test_results.png`.

```bash
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
python scripts/plot_test_results.py  --max-metric-value 0.5

# examples
# (omit --dataset to plot all datasets found under logs/)
python scripts/plot_test_results.py --dataset retailrocket --max-metric-value 0.5
python scripts/plot_test_results.py --max-metric-value 1.0 0.6 0.3
python scripts/plot_test_results.py --script SA2C_SASRec_rectools --dataset persrec_tc5_2025-08-21 --eval-scheme bert4rec_eval
```

## retailrocket
```bash
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/default.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/baseline.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/purchase_only.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/baseline.yml --smoke-cpu
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/baseline.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_pointwise_critic.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_from_pretrained.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_from_pretrained_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_from_pretrained.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_from_pretrained_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_pointwise_critic_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/purchase_only.yaml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/purchase_only_ndcg.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/baseline.yml --smoke-cpu --max_steps 64
```

## yoochoose
```bash
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/default.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/baseline.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/purchase_only.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/baseline.yml --smoke-cpu
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/baseline.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_pointwise_critic.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_pointwise_critic_mlp.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_from_pretrained.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_from_pretrained_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_from_pretrained.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_from_pretrained_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_pointwise_critic_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_pointwise_critic_mlp_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/purchase_only.yaml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/purchase_only_ndcg.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/baseline.yml --smoke-cpu --max_steps 64
```

## Optuna gridsearch (rectools)

```bash
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/baseline_gridsearch.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/baseline_gridsearch.yml
```

## Optuna gridsearch (rectools) — persrec_tc5 (bert4rec_eval)

- Use `limit_chunks_pct` (float in (0, 1]) to load only the first N parquet chunks for persrec_tc5 and cache derived artifacts under:
  - `data/persrec_tc5_<calc_date>/limit_chunks=<n_chunks>/`

```bash
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline_gridsearch.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default_gridsearch.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default_auto_warmup_subset_10pct.yml
```

## persrec_tc5 (BERT4Rec parquet format) — rectools

- Expects parquet at `data/persrec_tc5_<calc_date>/dataset_train.parquet/` (directory of parquet part-files).
- If missing, downloads from `<dataset.hdfs_working_prefix>/training/dataset_train.parquet` (tries `hdfs dfs -get`, then `hadoop fs -get`).

```bash
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/sa2c_eval/baseline.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/sa2c_eval/baseline.yml --sanity
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline.yml --sanity
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline_approx_hparams.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default.yml --sanity
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default_auto_warmup.yml
python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default_auto_warmup.yml --sanity
```

- Regular artifacts (created if missing):
  - `data/persrec_tc5_2025-08-21/built_vocabulary.pkl`
  - `data/persrec_tc5_2025-08-21/data_splits.npz`
  - `data/persrec_tc5_2025-08-21/data_statis.df`
  - `data/persrec_tc5_2025-08-21/pop_dict.txt`
- Sanity artifacts (created if missing; does not touch regular artifacts):
  - `data/persrec_tc5_2025-08-21/built_vocabulary_sanity.npz`
  - `data/persrec_tc5_2025-08-21/data_splits_sanity.npz`
  - `data/persrec_tc5_2025-08-21/data_statis_sanity.df`
  - `data/persrec_tc5_2025-08-21/pop_dict_sanity.txt`
- BERT4Rec-style LOO split artifacts (created if missing):
  - `data/persrec_tc5_2025-08-21/bert4rec_eval/dataset_splits.npz`
  - `data/persrec_tc5_2025-08-21/bert4rec_eval/dataset_splits_sanity.npz`

## Install conda envs (torch / tf)

```bash
conda env create -f dependencies/environment_torch.yml
conda env create -f dependencies/environment_tf.yml
```

## Run SA2C scripts using installed envs
```bash
conda activate sa2c_code_tf
cd Kaggle && python SA2C.py --data data
cd RC15 && python SA2C.py --data data
```