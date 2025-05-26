# Install dependencies for bcresnet
# install conda env
conda create --prefix /data2/hrq/miniconda3/envs/bcresnet python=3.10
conda activate /data2/hrq/miniconda3/envs/bcresnet
conda install -y -c conda-forge pynini==2.1.5 sox tqdm requests
# install pip requirements
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --timeout 300 --retries 10

