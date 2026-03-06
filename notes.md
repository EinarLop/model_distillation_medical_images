### Create Environment
conda env create -f environment.yml
### Activate Environment
conda activate model_distillation_medical_images
### Update Environment (With Env Active)
conda env update --file environment.yml
### Install Manually
conda install pytorch torchvision cpuonly -c pytorch

### Uninstall 
conda uninstall pytorch torchvision cpuonly

### M1
conda install pytorch torchvision -c pytorch


### Cluster
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
### Kaggle (Inline Env)
chmod 600 ./.kaggle/kaggle.json
KAGGLE_CONFIG_DIR=$(pwd)/.kaggle ./scripts/chexpert_kaggle_download.sh 
conda install "mkl<2024.1" "intel-openmp<2024.1" -c conda-forge -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121