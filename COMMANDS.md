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

## retailrocket
```bash
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/default.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/baseline.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/purchase_only.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/retailrocket/baseline.yml --smoke-cpu
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/retailrocket/default.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/retailrocket/baseline.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/retailrocket/default_auto_warmup.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/retailrocket/sampled_loss_auto_warmup.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/retailrocket/purchase_only.yaml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/retailrocket/purchase_only_ndcg.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/retailrocket/baseline.yml --smoke-cpu --max_steps 64
```

## yoochoose
```bash
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/default.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/baseline.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/purchase_only.yml
python SA2C_SASRec_torch.py --config conf/SA2C_SASRec_torch/yoochoose/baseline.yml --smoke-cpu
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/yoochoose/default.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/yoochoose/baseline.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/yoochoose/default_auto_warmup.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/yoochoose/sampled_loss_auto_warmup.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/yoochoose/purchase_only.yaml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/yoochoose/purchase_only_ndcg.yml
python SA2C_SASRec_rectools.py --config conf/SA2C_SASRec_rectools/yoochoose/baseline.yml --smoke-cpu --max_steps 64
```

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