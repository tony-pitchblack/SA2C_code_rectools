## Copy YooChoose/Retailrocket datasets from unzipped `SA2C_code.zip`
SA2C_code_unzip_dir=/raid/data_share/antonchernov/transformer_benchmark_rl/data/raw/SA2C_code
rsync -a "${SA2C_code_unzip_dir}/Kaggle/data/" "Kaggle/data/"
rsync -a "${SA2C_code_unzip_dir}/RC15/data/" "RC15/data/"

## Run Torch (uv + local .venv) — SASRec only
conda activate sa2c_code_torch

python Kaggle/SA2C_SASRec_torch.py \
  --model SASRec \
  --data Kaggle/data \
  --batch_size 10500 \ # ~29GB GPU
  --num_workers 4 \
  --device_id 2

## Install conda envs (torch / tf)
conda env create -f dependencies/environment_torch.yml
conda env create -f dependencies/environment_tf.yml

## Run SA2C scripts using installed envs
conda activate sa2c_code_tf
cd Kaggle && python SA2C.py --data data
cd RC15 && python SA2C.py --data data