## Code Snippets ##

Some useful code snippets are presented below.

### Bash ###

#### Sudo Access Test <a id="sudo-access-test"></a>  ####

Enter the following commands at the command line:

* `ls`
* `sudo ls` 

If, after entering your password, you see the same output for both commands, you have `sudo` access. Otherwise, you do not.


### Python ###

#### Location of package source files <a id="location-of-package-source-files"></a> #### 

To find the location of the package's files, run the following command:

* `python -c 'import site; print(site.getsitepackages())'`

This will output the *site packages* directory for the python executable. The package's
files will be in the `aopp_deconv_tool` subdirectory.

#### Getting documentation from within python <a id="getting-documentation-from-within-python"></a> ####

Python's command line, often called the "Read-Evaluate-Print-Loop (REPL)", has a built-in help system. 

To get the help information for a class, function, or object use the following code. Note, `>>>` denotes the
python REPL (i.e., the command line you get when you type the `python` command), and `$` denotes the shell 
command-line. 

This example is for the built-in `os` module, but should work with any python object.

```
$ python
... # prints information about the python version etc. here
>>> import os
>>> help(os)
... # prints out the docstring of the 'os' module
```
