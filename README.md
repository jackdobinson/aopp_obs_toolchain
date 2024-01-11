# aopp_obs_toolchain


## Development Environment Setup ##

This section details how to set up a development environment for this project.

Placeholders:

	`<REPO_DIR>`
		The top-level directory of this repository. I.e. the directory this file is in.

	`<PYTHON_VERSION>`
		The version of python used in development, currently this is `3.11.5`.
	
	`<VENV_DIR>`
		The directory of the python virtual environment for the project. Default name is `.venv_<PYTHON_VERSION>`.


### setting up the virtual environment ###

* Download and install python 3.11.5 from the [python website](https://www.python.org/downloads/release/python-3115/) or any other location.
  - Note: 3.11.5 is the python version used in development, newer and/or older versions may work as well.

* Create a virtual environment in this directory using the following command:
	`python3.11 -m venv <VENV_DIR>`
	
	Where `<VENV_DIR>` is the directory to store the virtual environment in. I suggest `.venv_<PYTHON_VERSION>`, and will assume this name in the rest of this document.

* Run the command `cd <REPO_DIR>; echo "${PWD}/src" > <VENV_DIR>/lib/python3.11/site-packages/aopp_obs_toolchain.pth`. This will create a ".pth" file that tells python where to look for the package's source files.

* **If** you have modified the environment variable PYTHONPATH, and those changes will interfere with development, run the command `echo -e "alias python=\"python -E"\nalias python3=\"python3 -E"\nalias python3.11=\"python3.11 -E\"" >> <VENV_DIR>/bin/activate`. This will ensure that the PYTHONPATH environment variable is ignored for the virtual environment.

* Activate the virtual environment via the command: `source <VENV_DIR>/bin/activate`. I will assume the virual environment is active from now on.

* Run the command `pip install -r <REPO_DIR>/requirements.txt` to install required supporting packages.


### VSCode Setup ###


#### If using WSL (Windows Subsystem for Linux) ####

##### Getting Plots To Display Correctly #####

* Download an X11 server for windows [VcXsrv](https://sourceforge.net/projects/vcxsrv/) is a well supported one, and the one assumed for the rest of this document.

* Launch VcXsrv, on the "Extra Settings" page, tick the "Disable Access Control" checkbox (last checkbox).

* To make WSL find the X11 server, enter the following command:

  - **IF** using WSL 1: `export DISPLAY=${DISPLAY:-localhost:0.0}`

  - **IF** using WSL 2: `export DISPLAY=${DISPLAY:-$(grep -oP "(?<=nameserver ).+" /etc/resolv.conf):0.0}`

  - Add the command to your `~/.bashrc` file (or equivalent) so you don't have to do it every time via,

    + **IF** using WSL 1: `echo 'export DISPLAY=${DISPLAY:-localhost:0.0}' >> ~/.bashrc`

    + **IF** using WSL 2: `echo 'export DISPLAY=${DISPLAY:-$(grep -oP "(?<=nameserver ).+" /etc/resolv.conf):0.0}' >> ~/.bashrc`

  NOTE:
    The `${DISPLAY:-<WORD>}` construct is called "Parameter Expansion", it returns the expansion of `<WORD>` ONLY IF `$DISPLAY` is unset or null. See [this page on parameter expansion](https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html) for more information.

* Test the setup using the command `python3 -c 'import matplotlib.pyplot as plt; plt.plot([i for i in range(-50,51)],[i**2 for i in range(-50,51)]); plt.show()'`, if a plot shows up then everything worked and plotting commands should work nicely in WSL.

* If anything went wrong, see the first answer to [this question on stack overflow](https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2)


#### If Using Linux (Including WSL) ####

##### Getting the virtual environment to activate automatically #####

Often the way VSCode activates virtual environments is not the "normal" way, so things like `*.pth` files are not treated correctly, to get around this problem don't let VSCode activate virtual environments for itself, and add a check to your `~/.bashrc` file to look for a virtual environment.

* In VSCode, go to settings > Extensions > Python, and uncheck the following options: "Terminal : Activate Env In Current Terminal", and "Terminal : Activate Environment".

* In VSCode, go to settings > terminal > integrated > env : linux (easiest to search for "terminal integrated env:linux" in settings). Click "Edit in settings.json". Add the line `"VSCODE_WORKSPACE_DIR" : "${workspaceFolder}"` to the `"terminal.integrated.env.linux"` entry so it looks like the following:

```
    "terminal.integrated.env.linux": {
        "VSCODE_WORKSPACE_DIR" : "${workspaceFolder}"
    }
```

  This adds the environment variable "VSCODE_WORKSPACE_DIR" to every terminal that VSCode opens, and sets it to the current **workspace** (i.e. top level) folder.

* Run the following command to get your `~/.bashrc` file to load up the virtual environment properly when the terminal opens:
```
cat >> ~/.bashrc <<- "END_OF_FILE"
# Only executed when VSCode has opened the terminal
if [ "${TERM_PROGRAM}" == "vscode" ]; then
    
    # Check to make sure we have a workspace directory
    if [ "${VSCODE_WORKSPACE_DIR:-'UNSET_OR_NULL'}" == 'UNSET_OR_NULL' ]; then
        echo "ERROR: $$TERM_PROGRAM == \"vscode\", but $$VSCODE_WORKSPACE_DIR is unset or null"
    else
        # Count the number of python virtual environments
        shopt -s nullglob 
        VENV_DIRS=(.venv*)
        shopt -u nullglob
    
        # If we only have one virtual environment, activate it, otherwise print out activation commands
        if [ ${#VENV_DIRS[@]} == 1 ]; then
            source ${VENV_DIRS[0]}/bin/activate
        elif [ ${#VENV_DIRS[@]} -ge 2 ]; then
            echo "Multiple python virtual environments found in \"${VSCODE_WORKSPACE_DIR}\""
            echo "activate one of them with:"
            
            for VENV_DIR in ${VENV_DIRS[@]}; do
                echo -e "\tsource ${VENV_DIR}/bin/activate"
            done
        fi
    fi
fi
END_OF_FILE
```


## Running Tests ##

The tests are in the directory `<REPO_DIR>/tests`, there is a package `<REPO_DIR>/scientest` which is a testing tool. The module `<REPO_DIR>/scientest/run.py` will search for tests and run them one by one. It tries to ensure that tests do not have side-effects. 

Folders are searched if:
	* They **do not**  begin with double underscores (`__`).

Files are searched if:
	* They have `test` in their name.
	* They end with `.py`.
	* They **do not** begin with double underscores (`__`).


### Steps to run tests ###

* Open a terminal window.

* Ensure you are in the top-level repository directory via `cd <REPO_DIR>`

* Activate the virtual environment with `source <VENV_DIR>/bin/activate`

* Run the tests (includes test discovery) via `python3 -m scientest.run ./tests`.

