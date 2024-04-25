# This script is used to initialize runpod.io server
# It assumes we have a Pod deployed with 2GPUs
git clone -b https://github.com/mytnguyen26/METCS777-GenAIForTheme.git
export PYTHONPATH="$PYTHONPATH:/workspace/METCS777-GenAIForTheme/"
cd METCS777-GenAIForTheme
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
gdown --folder https://drive.google.com/drive/u/0/folders/1UF9OXiONIDlI3dQfC8FQbDsJEUluOrG4
export CUDA_VISIBLE_DEVICES=0,1
