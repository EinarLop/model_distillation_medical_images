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

### Cluster (Match CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c