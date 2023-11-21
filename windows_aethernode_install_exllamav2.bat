@echo off
REM Change directory to the script's location
cd %~dp0

REM Create a virtual environment named 'venv'
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install requirements from requirements.txt
pip install -r requirements.txt

REM The following line assumes you have CUDA 11.8 installed, change it if you have a different version
REM Please visit https://pytorch.org/get-started/locally/ to find the correct install command for your CUDA version
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

REM Install specific versions of transformers and optimum
pip install transformers>=4.32.0 optimum>=1.12.0

pip install exllamav2==0.0.8

echo Setup is complete.
pause
