conda create -yn polar-vl pytorch torchvision torchaudio pytorch-cuda=12.1 numpy=1.26.4 pandas matplotlib -c pytorch -c nvidia
conda run -n polar-vl conda install pip
conda run -n polar-vl pip install ftfy regex tqdm
conda run -n polar-vl pip install git+https://github.com/openai/CLIP.git