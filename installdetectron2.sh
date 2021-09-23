wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3.sh
bash ~/miniconda3.sh -b -p $HOME/miniconda3

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

conda create -n detectron2 python=3.8 -y
conda activate detectron2

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
yes | python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html


cp /usr/local/lib/python3.8/site-packages/asap.pth /home/user/miniconda3/envs/detectron2/lib/python3.8/site-packages/
cp /usr/local/lib/python3.8/site-packages/whole-slide-data.pth /home/user/miniconda3/envs/detectron2/lib/python3.8/site-packages/
pip install shapely
pip install opencv-python

# install extensions on server
# select conda env
# install ikernel jupyter (vscode will ask)

## add to jupyte notebook envs
# ipython kernel install --name "detectron2" --user

# run jupyter notebook
# forward port 