
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3.sh
bash ~/miniconda3.sh -b -p $HOME/miniconda3

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

conda create -n detectron python=3.8 -y
conda activate detectron

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y
yes | python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

cp /usr/local/lib/python3.8/site-packages/asap.pth /home/user/miniconda3/envs/detectron2/lib/python3.8/site-packages/
pip install shapely
pip install opencv-python
pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main
#todo intall wholeslidedetectron2

conda install -c anaconda ipykernel -y
python -m ipykernel install --name=detectron
