# Getting Started #

## Installation ##

### Python Installation and Virtual Environment Setup <a id="python-installation-and-virtual-environment-setup"></a>  ###

As Python is used by many operating systems as part of its tool-chain it's a good idea to avoid
fiddling with the "system" installation (so you don't unintentionally overwrite packages or 
python versions and make it incompatible with your OS). Therefore the recommended way to use Python
is via a *virtual environment*. A *virtual environment* is isolated from your OS's Python installation.
It has its own packages, its own `pip` (stands for "pip install packages", recursive acronyms...),
and its own versions of the other bits it needs. 

This package was developed using Python 3.12.2. Therefore, you need a Python 3.12.2 (or later)
installation, and ideally a *virtual environment* to run it in.

#### Installing Python <a id="installing-python"></a>  ####

It is recommended that you **do not add the installed python to your path**. If you do, the operating system may/will find the installed version
before the operating system's expected version. And as our new installation almost certainly doesn't have the packages the operating system requires,
and may be an incompatible version, annoying things can happen. Instead, install Python in an obvious place that you can access easily. 

Suggested installation locations:

* Windows : `C:\Python\Python3.12`

* Unix/Linux/Mac: `${HOME}/python/python3.12`

Installation instructions for [windows, mac](#windows-back-slash-mac-installation-instructions), [unix and linux](#unix-back-slash-linux-installation-instructions) are slightly
different, so please refer to the appropriate section below.

Once installed, if using the suggested installation location, the actual Python interpreter executable will be at one of the following
locations or an equivalent relative location if not using the suggested install location:

* Windows: `C:\Python\Python3.12\bin\python3.exe`

* Unix/Linux/Mac: `${HOME}/python/python3.12/bin/python3`

* NOTE: if using *Anaconda*, you will have a `conda` command that manages the installation location of Python for you.

NOTE: I will assume a linux installation in this guide, so the executable will be at `${HOME}/python/python3.12/bin/python3`
in all code snippets. Alter this appropriately if using windows or a non-suggested installation location.

##### Windows/Mac Installation Instructions <a id="windows/mac-installation-instructions"></a> #####

* Download and run an installer from [the official Python site](https://www.python.org/downloads/).


##### Unix/Linux Installation Instructions <a id="unix/linux-installation-instructions"></a> #####

* **IF** you have `sudo` access [see this snippet for a sudo access]({{site.baseurl}}/resources#sudo-access-test), try one of the following:

  - Install the desired version of Python via the Package Manager included in your operating system

  - Build and [install python from source](https://docs.python.org/3/using/unix.html).

	+ NOTE: Building from source can be a little fiddly, but there are [online tools to help with building from source](https://www.build-python-from-source.com/).
	  There is also a [python installation script on the resources page]({{site.baseurl}}/resources#linux-python-installation) that will fetch the python 
	  source code, install it, and create a virtual environment.

* **OTHERWISE**, if you don't have `sudo` access, [anaconda python](https://docs.anaconda.com/free/miniconda/index.html#quick-command-line-install)
  is probably the easiest way as it does not require `sudo`. I recommend the `miniconda` version (linked above), as the main version installs many
  packages you may not need. You can always install other packages later.

  - NOTE: The main problem is that installing dependencies requires `sudo` access and, while there 
	[are ways around `sudo`](https://askubuntu.com/questions/339/how-can-i-install-a-package-without-root-access), 
	they are fiddly and annoying to use as you can quickly end up in [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell).





#### Creating and Activating a Virtual Environment <a id="creating-and-activating-a-virtual-environment"></a> ####

A virtual environment isolates the packages you are using for a project from your normal environment and other virtual environments.
Generally they are created in a directory which we will call `<VENV_DIR>`, and then activated and deactivated as required. NOTE:
*anaconda python* has slightly different commands for managing virtual environments, and uses **names** of virtual environments instead
of **directories**, however the concept and the idea of activating and deactivating them remains the same dispite the slightly different
technical details.

NOTE: For the rest of this guide, *python* refers to a manual Python installation and *anaconda python* to Python provided by Anaconda.
NOTE: I will assume python version `3.12.2` for the rest of this guide but this will also work for different versions as long as the
	  version number is changed appropriately.

##### Check Python Installation <a id="check-python-installation"></a> #####

With [Python installed](#installing-python), make sure you have the correct version via `${HOME}/python/python3.12/bin/python3 --version`. The command should print `Python 3.12.2`, or whichever version you expect.

##### Check Anaconda Python Installation <a id="check-anaconda-python-installation"></a> #####

If using *anaconda python* check that everything is installed correctly by using the command `conda --version`. This should print
a string like `conda X.Y.Z`, where X,Y,Z are the version number of anaconda.

##### Creating a Python Virtual Environment <a id="creating-a-python-virtual-environment"></a> #####

To create a virtual environment use the command `${HOME}/python/python3.12/bin/python3 -m venv <VENV_DIR>`, where `<VENV_DIR>` is the directory
you want the virtual environment to be in. E.g., `${HOME}/python/python3.12/bin/python3 -m venv .venv_3.12.2` will create the virtual
environment in the directory `.venv_3.12.2` in the current folder (NOTE: the `.` infront of the directory
name will make it hidden by default).

##### Creating an Anaconda Python Virtual Environment <a id="creating-an-anaconca-python-virtual-environment"></a> #####

*Anaconda Python* manages many of the background details for you. Use the command `conda create -n <VENV_NAME> python=3.12.2`, where
`<VENV_NAME>` is the name of the virtual environment to create. E.g., `conda create -n venv_3.12.2 python=3.12.2`


##### Activating and Deactivating a Python Virtual Environment <a id="activating-and-deactivating-a-python-virtual-environment"></a> #####

The process of activating the virtual environment varies depending on the terminal shell you are using.
On the command line, use one of the following commands:

* cmd.exe (Windows): `<VENV_DIR>\Scripts\activate.bat` 

* PowerShell (Windows, maybe Linux): `<VENV_DIR>/bin/Activate.ps1`

* bash\|zsh (Linux, Mac): `source <VENV_DIR>/bin/activate`

* fish (Linux, Mac): `source <VENV_DIR>/bin/activate.fish`

* csh\|tcsh (Linux, Mac): `source <VENV_DIR>/bin/activate.csh`

Once activated, your command line prompt should change to have something like `(.venv_3.12.2)` infront of it.

To check everything is working, enter the following commands (NOTE: the full path is not required as we are now using the virtual environment):

* `python --version`
  - Should output the version you expect, e.g., `Python 3.12.2`

* `python -c 'import sys; print(sys.prefix != sys.base_prefix)'`
  - Should output `True` if you are in a virtual environment or `False` if you are not.

To deactivate the environment, use the command `deactivate`. Your prompt should return to normal.

##### Activating and Deactivating an Anaconda Python Virtual Environment <a id="activating-and-deactivating-an-anaconda-python-virtual-environment"></a> #####

*Anaconda python* has a simpler way of activating a virtual environment. Use the command `conda activate <VENV_NAME>`, your prompt
should change to have something like `(<VENV_NAME>)` infront of it. Use `python --version` to check that the activated environment
contains the expected python version.

To deactivate the environment, use the command `conda deactivate`. Your prompt should return to normal.



### Installing the Package via Pip <a id="installing-the-package-via-pip"></a> ###

NOTE: If using *anaconda python* you **may** be able to use `conda install` instead of `pip` but I have not tested this. Conda [should behave well](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages) when using packages installed via `pip`.

Once you have [installed Python 3.12.2 or higher](#installing-python), [created a virtual environment](#creating-and-activating-a-virtual-environment), and [activated
the virtual environment](#activating-and-deactivating-a-python-virtual-environment). Use the following command to install the package:

* `python -m pip install --upgrade pip`
  - This updates pip to its latest version

* `python -m pip install aopp_deconv_tool`
  - This actually installs the package.

NOTE: We are using `python -m pip` instead of just `pip` incase the `pip` command does not point to the virtual environment's `pip` executable. You can run
`pip --version` to see which version of python it is for and where the pip executable is located if you want. As explanation, `python -m pip` means "use python
to run its pip module", whereas `pip` means "look on my path for the first executable called 'pip' and run it". Usually they are the same, but not always.


To update the package to it's newest version use:

* `python -m pip install --upgrade aopp_deconv_tool`

## Quirks of the System ##


### FITS Specifier <a id="fits-specifier"></a> ###

When operating on a FITS file there are often multiple extensions, the axes ordering is unknown, and you may only need a subset of the data. Therefore, where possible scripts accept a *fits specifier* instead of a bare path. The format is a follows:

```
path_to_fits_file{ext}[slice0,slice1,...,sliceM]{axes_type_1:(ax11,ax12,...,ax1L),...,axes_type_N:(axN1,axN2,...,axNL)}
```
`path_to_fits_file` : str
: (required) A string that represents  the path to the FITS file on the file system

`ext` : str \| int
: (optional) A string or integer that specifies the extension of the FITS file to operate upon

`sliceM` : slice
: (optional) A slice (in python slice syntax) that chooses a sub-set of the of the data in the FITS file extension.

`axes_type_N` : str
: (optional) A string that tells a script how the axes in the FITS file correspond to physical quantities. These can often be automatically found.

`(axN1, ..., axNL)` : (int, ..., int)
: (optional) A tuple (in python tuple syntax) of integers that denotes the axes (in C ordering, **not** FORTRAN ordering) that correspond to the physical quantities described by `axes_type_N`. These can often be found automatically, and even when not can be specified without the corresponding `axes_type_N` much of the time).

In words, a *fits specifier* consists of: a string that describes which FITS file to load; the extension (i.e., backplane) name or number to use enclosed in curly brackets; the slices (i.e., sub-regions) that should be operated upon in [python slice syntax]({{site.baseurl}}/resources#python-slice-syntax); and the data axes to operate on as a [tuple]({{site.baseurl}}/resources#python-tuple-syntax) or as a [python dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) with strings as keys and [tuples]({{site.baseurl}}/resources#python-tuple-syntax) as values.

See the resources page for a [quick introduction to the FITS format]({{site.baseurl}}/resources#fits-file-format-information) for a description of why this information is needed, and why [axis numbers are different between FITS and Python]({{site.baseurl}}/resources#fits-data-order).


## Running the First Example ##