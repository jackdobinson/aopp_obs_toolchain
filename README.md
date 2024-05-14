# aopp_obs_toolchain #

A set of tools for dealing with observational data from CCD and IFU instruments.

## aopp_deconv_tool ##

Deconvolution algorithms, PSF fitting, image (and psf) filtering, and other data operations to support deconvolution.


## Development Environment Setup ##

This section details how to set up a development environment for this project.

### placeholder definitions ###

Placeholders:

	`<REPO_DIR>`
		The top-level directory of this repository. I.e. the directory this file is in.

	`<X.Y.Z>`
		The version of python used in development, currently this is `3.12.2`.
	
	`<VENV_DIR>`
		The directory of the python virtual environment for the project. Default name is `.venv_<PYTHON_VERSION>`.


### setting up the virtual environment ###

* Created using python 3.12.2, I'll refer to the version number as `<X.Y.Z>` in code, where X is the major release, Y is the minor release, and Z is the build number.

  - NOTE: Sometimes only some of the version numbers are used e.g. `<X.Y>` for just the major and minor number, or `<XYZ>` when the numbers are concatenated together 
    without the dots inbetween them (i.e. `3122` instead of `3.12.2`).

  - I.e. the python executable for some version would be `python<X.Y.Z>`

* Download and install python 3.12.2 from the [python website](https://www.python.org/downloads/release/python-3122/) or any other location.
  - Note: 3.12.2 is the python version used in development, newer and/or older versions may work as well.

* Create a virtual environment in this directory using the following command:

  - `python<x.y> -m venv <VENV_DIR>`

  - Where `<VENV_DIR>` is the directory to store the virtual environment in. I suggest `.venv_<X.Y.Z>`, and will assume this name in the rest of this document.

* To ensure that the virtual environment can find the development files, we will create a ".pth" file inside the virtual environment that tells python
  where to look for modules. Replace `<REPO_DIR>`, `<VENV_DIR>`, `<X>`, `<Y>` in the following command:

  - `cd <REPO_DIR>; echo "${PWD}/src" > <VENV_DIR>/lib/python<X>.<Y>/site-packages/aopp_obs_toolchain.pth`

* **If** you have modified the environment variable PYTHONPATH, and those changes will interfere with development, you want to ensure the PYTHONPATH environment
  variable is ignored for the virtual environment. The following command will set up aliases to the python executables in the virtual environment, with command-line
  arguments set so they will ignore PYTHONPATH.
  
  - `echo -e "alias python=\"python -E\"\nalias python3=\"python3 -E\"\nalias python<X>.<Y>=\"python<X>.<Y> -E\"" >> <VENV_DIR>/bin/activate`

* Activate the virtual environment via the command: 

  - `source <VENV_DIR>/bin/activate`. I will assume the virual environment is active from now on.

* Install required supporting packages with the command 

  - `pip install -r <REPO_DIR>/requirements.txt`

#### default installation as a single command ####

Use the following command to install the desired version of python and do the steps above.

* Alter the "PYTHON_INSTALL_DIRECTORY" variable to install python in a different place

* Alter the "REPO_DIR" variable to put the repository in a different place

* Alter the "VENV_PREFIX" variable to have a different prefix for the virtual environment

* NOTE: use [CTRL]+[SHIFT]+[V] when pasting into the terminal to avoid strangeness.

```
PYTHON_VERSION=(3 12 2) && \
PYTHON_INSTALL_DIRECTORY="${HOME:?}/python/python_versions" && \
REPO_DIR="${HOME:?}/repos/aopp_obs_toolchain" && \
VENV_PREFIX=".venv" && \
echo "" && \
echo "PYTHON_MAJOR_VERSION=${PYTHON_VERSION[0]:?ERROR: PYTHON_VERSION must have 3 entries (e.g. '(3 12 2)')}" && \
echo "PYTHON_MINOR_VERSION=${PYTHON_VERSION[1]:?ERROR: PYTHON_VERSION must have 3 entries (e.g. '(3 12 2)')}" && \
echo "PYTHON_BUILD_VERSION=${PYTHON_VERSION[2]:?ERROR: PYTHON_VERSION must have 3 entries (e.g. '(3 12 2)')}" && \
echo "PYTHON_INSTALL_DIRECTORY=${PYTHON_INSTALL_DIRECTORY:?ERROR: PYTHON_INSTALL_DIRECTORY must be set}" && \
echo "REPO_DIR=${REPO_DIR:?ERROR: REPO_DIR must be set}" && \
echo "VENV_PREFIX=${VENV_PREFIX:?ERROR: VENV_PREFIX must be set}" && \
echo "" && \
PYTHON_VERSION_STR="${PYTHON_VERSION[0]}.${PYTHON_VERSION[1]}.${PYTHON_VERSION[2]}" && \
echo "${PYTHON_VERSION_STR}" && \
VENV_DIR="${REPO_DIR:?}/${VENV_PREFIX:?}_${PYTHON_VERSION_STR:?}" && \
DOWNLOAD_URL="https://www.python.org/ftp/python/${PYTHON_VERSION_STR:?}/Python-${PYTHON_VERSION_STR:?}.tgz" && \
PYTHON_VERSION_INSTALL_DIR="${PYTHON_INSTALL_DIRECTORY:?}/python${PYTHON_VERSION_STR:?}" && \
echo "PYTHON_VERSION_INSTALL_DIR=${PYTHON_VERSION_INSTALL_DIR:?}" && \
mkdir -p ${PYTHON_VERSION_INSTALL_DIR:?} && \
PYTHON_VERSION_SOURCE_DIR="${PYTHON_VERSION_INSTALL_DIR:?}/Python-${PYTHON_VERSION_STR:?}" && \
echo "PYTHON_VERSION_SOURCE_DIR=${PYTHON_VERSION_SOURCE_DIR:?}" && \
PYTHON_VERSION_DOWNLOAD_FILE="${PYTHON_VERSION_SOURCE_DIR:?}.tgz" && \
echo "PYTHON_VERSION_DOWNLOAD_FILE=${PYTHON_VERSION_DOWNLOAD_FILE:?}" && \
echo "Downloading python source from ${PYTHON_VERSION_DOWNLOAD_FILE:?}..." && \
curl ${DOWNLOAD_URL:?} --output ${PYTHON_VERSION_DOWNLOAD_FILE:?} && \
echo "Python source downloaded." && \
echo "Installing dependencies..." && \
sudo apt-get install -y \
    curl \
    gcc \
    libbz2-dev \
    libev-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncurses-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    make \
    tk-dev \
    wget \
    zlib1g-dev && \
echo "Dependencies installed." && \
echo "Extracting source file ${PYTHON_VERSION_DOWNLOAD_FILE:?}" && \
tar -xvzf ${PYTHON_VERSION_DOWNLOAD_FILE:?} -C ${PYTHON_VERSION_INSTALL_DIR:?} && \
echo "source file extacted." && \
echo "Moving into source-code directory. ${PYTHON_VERSION_SOURCE_DIR:?}..." && \
cd ${PYTHON_VERSION_SOURCE_DIR:?} && \
echo "Configuring python installation..." && \
./configure \
    --prefix=${PYTHON_VERSION_INSTALL_DIR:?} \
	--enable-optimizations \
	--enable-ipv6 \
     && \
echo "Configuration done." && \
echo "Running makefile..." && \
make && \
echo "Makefile complete." && \
echo "Performing installation..." && \
make install && \
echo "Installation complete" && \
echo "Making virtual environment at ${VENV_DIR:?}" && \
${PYTHON_VERSION_INSTALL_DIR:?}/bin/python3 -m venv ${VENV_DIR:?} && \
echo "Virtual environment created" && \
PYTHON_MAJOR_MINOR_STR="${PYTHON_VERSION[0]:?}.${PYTHON_VERSION[1]:?}" && \
echo "PYTHON_MAJOR_MINOR_STR=${PYTHON_MAJOR_MINOR_STR:?}" && \
REPO_SOURCE_DIR=$(readlink -f "${REPO_DIR:?}/src") && \
echo "REPO_SOURCE_DIR=${REPO_SOURCE_DIR:?}" && \
echo "Creating '.pth' file for virtual environment..." && \
echo "${REPO_SOURCE_DIR:?}" > ${VENV_DIR:?}/lib/python${PYTHON_MAJOR_MINOR_STR:?}/site-packages/aopp_obs_toolchain.pth && \
echo "'.pth' file created" && \
echo "Adding aliases to avoid PYTHONPATH..." && \
echo -e "alias python=\"python -E\"\nalias python3=\"python3 -E\"\nalias python${PYTHON_MAJOR_MINOR_STR:?}=\"python${PYTHON_MAJOR_MINOR_STR:?} -E\"" >> ${VENV_DIR:?}/bin/activate && \
echo "Aliases added." && \
echo "Activating virtual environment..." && \
source ${VENV_DIR:?}/bin/activate && \
echo "Virtual environment activated" && \
echo "Installing required packages" && \
pip install -r ${REPO_DIR:?}/requirements.txt && \
echo "Required packages installed"
```





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
        else
          echo "ERROR: No virtual environments (matching glob with \".venv*\") found in \"${VSCODE_WORKSPACE_DIR}\""
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

### Test Output ###

* Some logging output will appear at the top, this is recognisable as each logging line has a prefix '<TIME> <FILE>:<LINE> "<FUNCTION>" <LEVEL>: '
  
  - <TIME> is the system time the log is written
  
  - <FILE> is the python file (not including folders) the log is comming from
  
  - <LINE> is the line of the file the log is coming from
  
  - <FUNCTION> is the function in the file the log is coming from
  
  - <LEVEL> is the level of the log, by default there are 5 log levels. In order of severity they are: "DEBUG", "INFO", "WARN", "ERROR", "CRIT".

* There is a "Discovery Summary" section that details all of the tests found by `scientest`. The format is:

  - The summary starts with a line that looks like "================== Discovery Summary ======================"
  
  - Following entries have a module on one line, then the tests found in that module on subsequent lines with a hanging indent e.g.
    ```
    module "lucy_richardson_test" contains tests:
        test_call_altered_instantiated_parameters
        test_on_example_data
        test_on_example_data_with_plotting_hooks
        test_runs_for_basic_data
    module "clean_modified_test" contains tests:
        test_clean_modified_call_altered_instantiated_parameters
        test_clean_modified_on_example_data
        test_clean_modified_on_example_data_with_plotting_hooks
    ```
    
* Next there is the "Running Tests" section, it's default is to output the results live, so it may take a while to complete. 
  The format is:

  - Starts with a line that looks like "====================== Running Tests ========================="
  
  - Each line details the test module and the test function on the LHS in the format "module::function", the staus 
    (Passed, Failed, Skipped, etc.) on the RHS, if a test is skipped the following line contains some right-justified text 
    explaining why the test was skipped. Tests should all pass or be skipped with a reason. E.g.
    ```
    lucy_richardson_test::test_call_altered_instantiated_parameters -------------------------------------------------- Passed
    lucy_richardson_test::test_on_example_data ----------------------------------------------------------------------- Passed
    lucy_richardson_test::test_on_example_data_with_plotting_hooks -------------------------------------------------- Skipped
                         broken: displays animated plots that are incompatible with intercepting matplotlib's "show" function
    lucy_richardson_test::test_runs_for_basic_data ------------------------------------------------------------------- Passed
    ```

## Building the Package ##

Run the command `python -m build` from the `<REPO_DIR>` directory. The `<REPO_DIR>/dist` and `<REPO_DIR>/aopp_deconv_tool.egg-info` folders
should be created. These contain the built package files. 

NOTE: When rebuilding, you may want to use `rm ./dist/*; python -m build` instead, as otherwise the
previous build information will stick around.

## Uploading the Package to Pypi ##

From the `<REPO_DIR>` directory, run **ONE** of the following commands:

* `python3 -m twine upload --repository testpypi -u <USERNAME> -p <PASSWORD> dist/*` to upload to the **TEST** python package index

* `python3 -m twine upload -u <USERNAME> -p <PASSWORD> dist/*` to upload to the **REAL** python package index

Use `__token__` for `<USERNAME>`, and an API Token value for `<PASSWORD>` (including the `pypi-` prefix). See 
[this guide for uploading to the package index](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives) 
for more information.

Verify the package uploaded correctly by going to `https://test.pypi.org/project/aopp-deconv-tool/`.

## Test the package uploaded correctly ##

Create a NEW virtual environment to test the package in. If you use the one in this repository it will just
use the defaults we have set up for development.

If using the **TEST** python package index, run the following command:

* `pip cache purge && pip install --index-url https://test.pypi.org/simple/ --no-deps aopp-deconv-tool`

* NOTE: This will NOT install the dependencies (as the **TEST** python package index probably does not have them).
  Therefore, install them yourself using `pip install -r <REPO_DIR>/requirements/deconv.txt`.

If using the **REAL** python package index, run the following command:

* `pip install aopp-deconv-tool`

### run the examples using the newly installed package ###

In whatever test directory you want, run the examples with:

* `python <REPO_DIR>/examples/psf_model_example.py`, will output to `./ultranest_logs`

* `python <REPO_DIR>/examples/amateur_data_analysis.py`, will output to `<REPO_DIR>/example_data/amateur_data/set_0/output`

* `python <REPO_DIR>/examples/amateur_data_deconv_comparison.py`, will output to `<REPO_DIR>/example_data/amateur_data/set_0/comparison_deconv`

These should all complete and write files to their outputs just as if you ran them in the development environment. You might want
to delete the contents of the output directories before running so you can see the new files being created.
