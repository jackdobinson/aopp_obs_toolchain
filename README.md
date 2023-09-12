# aopp_obs_toolchain


## Development Environment Setup ##

This section details how to set up a development environment for this project.

Placeholders:
	`<REPO_DIR>`
		The top-level directory of this repository. I.e. the directory this file is in.


### setting up the virtual environment ###

* Download and install python 3.11.5 from the [python website](https://www.python.org/downloads/release/python-3115/) or any other location.

* Create a virtual environment in this directory using the following command:
	`python3.11 -m venv <VENV_DIR>`
	
	Where `<VENV_DIR>` is the directory to store the virtual environment in. I suggest `.venv_3.10.11`, and will assume this name in the rest of this document.

* Run the command `cd <REPO_DIR>; echo "${PWD}/src" > <VENV_DIR>/lib/python3.11/site-packages/aopp_obs_toolchain.pth`. This will create a ".pth" file that tells python where to look for the package's source files.

* **If** you have modified the environment variable PYTHONPATH, and those changes will interfere with development, run the command `echo -e"alias python="python -E"\nalias python3="python3 -E"\nalias python3.11="python3.11 -E" >> <VENV_DIR>/bin/activate`. This will ensure that the PYTHONPATH environment variable is ignored for the virtual environment.

* Activate the virtual environment via the command: `<VENV_DIR>/bin/activate`. I will assume the virual environment is active from now on.

* Run the command `pip install -r <REPO_DIR>/requirements.txt` to install required supporting packages.
