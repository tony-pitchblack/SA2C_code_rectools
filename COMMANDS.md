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

## Sampled softmax (CE) with 1/4 vocab

- Add `ce_n_negatives: 0.25` to your rectools YAML config (top-level). This sets the number of sampled negatives to \(\lfloor 0.25 * item\_num \rfloor\).
- Implementation detail: for fractional `ce_n_negatives`, training uses a matmul-based “shared negatives” construction (no per-row `[N,C,H]` embedding gather).
- `ce_n_negatives: null` means full softmax (scores full vocab).
- `num_val_negative_samples` (optional) controls eval candidate pool size for `bert4rec_loo` only.

### persrec_tc5_2025-08-21 (bert4rec_eval)

```bash
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/vocab-pct=0.25_chunks-pct=0.1.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/vocab-pct=0.25_approx_hparams_chunks-pct=0.1.yml

# DDP example (2 GPUs)
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --module SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/vocab-pct=0.25_chunks-pct=0.1.yml
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --module SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/vocab-pct=0.25_approx_hparams_chunks-pct=0.1.yml
```

## Plot test results (clicks + purchase ndcg@10)

- Writes combined plots (rectools + torch) under `results/plots/{dataset_name}/test_results.png` and, for `persrec_tc5_*`, `results/plots/{dataset_name}/{eval_scheme}/test_results.png`.

```bash
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
CUDA_VISIBLE_DEVICES=3 python scripts/plot_test_results.py  --max-metric-value 0.5

# examples
# (omit --dataset to plot all datasets found under logs/)
CUDA_VISIBLE_DEVICES=3 python scripts/plot_test_results.py --dataset retailrocket --max-metric-value 0.5
CUDA_VISIBLE_DEVICES=3 python scripts/plot_test_results.py --max-metric-value 1.0 0.6 0.3
CUDA_VISIBLE_DEVICES=3 python scripts/plot_test_results.py --script SA2C_SASRec_rectools --dataset persrec_tc5_2025-08-21 --eval-scheme bert4rec_eval
```

## retailrocket
```bash
CUDA_VISIBLE_DEVICES=3 python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/default.yml
CUDA_VISIBLE_DEVICES=3 python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/baseline.yml
CUDA_VISIBLE_DEVICES=3 python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/baseline.yml --smoke-cpu
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/baseline.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_pointwise_critic.yml
CUDA_VISIBLE_DEVICES=4 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/crr.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_from_pretrained.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_from_pretrained_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_from_pretrained.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_from_pretrained_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_pointwise_critic_auto_warmup.yml

# purchase-only
CUDA_VISIBLE_DEVICES=3 python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_ndcg_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/baseline_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/default_ndcg_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/baseline.yml --smoke-cpu --max_steps 64
```

## yoochoose
```bash
CUDA_VISIBLE_DEVICES=3 python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/default.yml
CUDA_VISIBLE_DEVICES=3 python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/baseline.yml
CUDA_VISIBLE_DEVICES=3 python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/baseline.yml --smoke-cpu
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/baseline.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_pointwise_critic.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_pointwise_critic_mlp.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/crr.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_from_pretrained.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_from_pretrained_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_from_pretrained.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_from_pretrained_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_pointwise_critic_auto_warmup.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_pointwise_critic_mlp_auto_warmup.yml

# purchase-only
CUDA_VISIBLE_DEVICES=3 python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_ndcg_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_auto_warmup_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/baseline_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/default_ndcg_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/baseline.yml --smoke-cpu --max_steps 64
```

## Optuna gridsearch (rectools)

```bash
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/baseline_gridsearch.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/yoochoose/baseline_gridsearch_purchase_only.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/baseline_gridsearch.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/retailrocket/baseline_gridsearch_purchase_only.yml
```

## Optuna gridsearch (rectools) — persrec_tc5 (bert4rec_eval)

- Use `limit_chunks_pct` (float in (0, 1]) to load only the first N parquet chunks for persrec_tc5 and cache derived artifacts under:
  - `data/persrec_tc5_<calc_date>/limit_chunks=<n_chunks>/`

```bash
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline_gridsearch_chunks-pct=0.1.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default_gridsearch_chunks-pct=0.1.yml
```

## persrec_tc5 (BERT4Rec parquet format) — rectools

- Expects parquet at `data/persrec_tc5_<calc_date>/dataset_train.parquet/` (directory of parquet part-files).
- If missing, downloads from `<dataset.hdfs_working_prefix>/training/dataset_train.parquet` (tries `hdfs dfs -get`, then `hadoop fs -get`).
- Add `--sanity` to any command to create/use `_sanity` artifacts without touching regular artifacts.

## PLU distribution diagnostics (persrec_tc5 parquet)

```bash
source .venv/bin/activate
python scripts/plu_distribution.py --local-working-prefix /raid/data_share/antonchernov/bert4recv1/data/tc5/
```

```bash
source .venv/bin/activate
python scripts/plu_distribution.py --local-working-prefix ./data/persrec_tc5_2025-08-21
```

```bash
source .venv/bin/activate
python scripts/plu_debug.py --root  ./data/persrec_tc5_2025-08-21/limit_chunks\=20/
```

```bash
# sa2c_eval/
source .venv/bin/activate
uv pip install -r dependencies/requirements_torch.txt
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/sa2c_eval/baseline.yml

# bert4rec_eval/
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/crr.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline_approx_hparams.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline_chunks-pct=0.1.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default_chunks-pct=0.1.yml
CUDA_VISIBLE_DEVICES=3 python -m SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default_auto_warmup.yml
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --module SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/default_auto_warmup_chunks-pct=0.1.yml

# DDP example (3 GPUs)
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --standalone --nproc_per_node=3 --module SA2C_SASRec_rectools --config conf/SA2C_SASRec_rectools/persrec_tc5_2025-08-21/bert4rec_eval/baseline_approx_hparams_chunks-pct=0.1.yml
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
cd Kaggle && CUDA_VISIBLE_DEVICES=3 python SA2C.py --data data
cd RC15 && CUDA_VISIBLE_DEVICES=3 python SA2C.py --data data
```