echo "Creating environment"
pyenv install 3.9.5
pyenv virtualenv 3.9.5 "cityscapes"

echo "Activating environment."
pyenv activate "cityscapes"

echo "Installing requierements"
pip install -r requirements.txt

echo "Activated!"
