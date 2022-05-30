# Thesis
Code for thesis. 

# Workflow
To ensure integrity of the python environment, we should use a virtual python environment. Everytime you fetch from origin, please do the following:
1. `python3 -m venv .env` to create a new virtual environment called `.env`. 
2. `source .env/bin/activate` to activate the virtual python environment. 
3. `which python` and `which pip` and check that the `python` and `pip` commands point to the virtual environment and not your system python. 
4. `python -m pip install -r requirements.txt` to install the new list of required packages. 
5. If you need to exit the virtual environment, use `source deactivate`.