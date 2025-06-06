# Install dependencies for Custom Voice Wakeup Framework (CUVOW)

# 1. install conda env
conda create -n cuvow python=3.10
conda activate cuvow
# or you want to use a specific path for your conda environment
# conda create --prefix {path to your conda env} python=3.10
# conda activate {path to your conda env}

# 2. install conda dependencies
conda install -y -c conda-forge pynini==2.1.5 sox tqdm requests

# 3. install pip requirements
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --timeout 300 --retries 10

