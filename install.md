# 安装说明

conda create -n dinet python=3.7

conda activate dinet

pip install -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

conda install pytorch=1.13.0 torchvision=0.14.0 torchaudio=0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

conda install ffmpeg -c pytorch -c nvidia

