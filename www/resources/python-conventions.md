## Python Conventions ##


### Command-line Script Help Message Syntax <a id="overview-of-help-message-syntax"></a> ###

Python scripts use the [argparse](https://docs.python.org/3/library/argparse.html) standard library module to parse command-line arguments. This generates help messages that follow a fairly standard syntax. Unfortunately there is no accepted standard, but [some style guides are available](https://docs.oracle.com/cd/E19455-01/806-2914/6jc3mhd5q/index.html).

The help message consists of the following sections:
* A "usage" line
* A quick description of the script, possibly with an example invocation.
* Positional and Keyword argument descriptions
* Any extra information that would be useful to the user.

#### The Usage Line ####

Contains a short description of the invocation syntax. 

* Starts with `usage: <script_name>` where `<script_name>` is the name of the script being invoked. 
* Then keyword arguments are listed, followed by positional arguments (sometimes these are reversed). 
  - There are two types of keyword arguments, *short* and *long*, usually to save space only the *short* name is in the usage line.
	+ *short* denoted by a single dash followed by a single letter, e.g., `-h`.
	+ *long* denoted by two dashes followed by a string, e.g., `--help`.
* Optional arguments are denoted by square brackets `[]`.
* A set of choices of arguments that are mutually exclusive are separated by a pipe `|`
* A grouping of arguments (e.g., for required arguments that are mutually exclusive) is done with round brackets `()`
* A keyword argument that requires a value will have one of the following after it:
  - An uppercase name that denotes the kind of value e.g., `NAME` 
  - A data type that denotes the value e.g., `float` or `int` or `string`
  - A set of choices of literal values (one of which must be chosen) is denoted by curly brackets `{}`, e.g., `{1,2,3}` or `{choice1,choice2}`.

Example: `usage: deconvolve.py [-h] [-o OUTPUT_PATH] [--plot] [--deconv_method {clean_modified,lucy_richardson}] obs_fits_spec psf_fits_spec`

* The script file is `deconvolve.py`
* The `-h` `-o` `--plot` `--deconv_method` keyword arguments are optional
* The `--deconv_method` argument accepts one of the values `clean_modified` or `lucy_richardson`
* The keyword arguments are `obs_fits_spec` and `psf_fits_spec`

Example: `usage: spectral_rebin.py [-h] [-o OUTPUT_PATH] [--rebin_operation {sum,mean,mean_err}] [--rebin_preset {spex} | --rebin_params float float] fits_spec`

* The script file is `spectral_rebin.py`
* The all of the keyword arguments `-h` `-o` `--rebin_operation` `--rebin_preset` `-rebin_params` are optional
* The single positional argument is `fits_spec` 
* The `--rebin_operation` argument has a choice of values `sum`,`mean`,`mean_err`
* The `--rebin_preset` and `--rebin_params` arguments are mutually exclusive, only one of them can be passed
* The `--rebin_preset` argument only accepts one value `spex`
* The `--rebin_params` argument accepts two values that should be floats


#### The Quick Description ####

The goal of the quick description is to summarise the intent of the program/script, and possibly give some guidance on how to invoke it.

#### Argument Descriptions ####

Usually these are grouped into two sections `positional arguments` and `options` (personally I would call them keyword arguments, as they can sometimes be required) but they follow the same syntax.

* The argument name, or names if it has more than one.
* Any parameters to the argument (for keyword arguments).
* A description of the argument, and ideally the default value of the argument.

Example:

```
--rebin_operation {sum,mean,mean_err}
						Operation to perform when binning.
```
* Argument is a keyword argument (starts with `--`)
* Argument name is 'rebin_operation'
* Accepts one of `sum`,`mean`,`mean_err`
* Description is `Operation to perform when binning.`

A full argument description looks like this:
```
positional arguments:
  obs_fits_spec         The observation's (i.e., science target) FITS SPECIFIER, see the end of the help message for more information
  psf_fits_spec         The psf's (i.e., calibration target) FITS SPECIFIER, see the end of the help message for more information

options:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
						Output fits file path. By default is same as the `fits_spec` path with "_deconv" appended to the filename (default: None)
  --plot                If present will show progress plots of the deconvolution (default: False)
  --deconv_method {clean_modified,lucy_richardson}
						Which method to use for deconvolution. For more information, pass the deconvolution method and the "--info" argument. (default: clean_modified)
```

#### Extra Information ####

Information listed at the end is usually clarification about the formatting of string-based arguments and/or any other information that would be required to use the script/program. For example, in the above fill argument description example the FITS SPECIFIER format information is added to the end as extra information.


### Language Syntax Conventions ###

For those unfamiliar, a quick guide to important bits of python syntax that the command-line interface relies upon. For more information, see [the offical Python tutorial](https://docs.python.org/3/tutorial/index.html).

#### Python tuple syntax <a id="python-tuple-syntax"></a> ####

Tuples are ordered collections of hetrogeneous items. They are denoted by separating each element with a comma and enclosing the whole thing in round brackets. Tuples can be nested.

Examples:
* `(1,2,3)`
* `('cat', 7, 'dog', 9)`
* `(5.55, ('x', 'y', 0), 888, 'abc', 'def')`

#### Python slice syntax <a id="python-slice-syntax"></a> ####

When specifying subsets of datacubes, it is useful to be able to select a N-square (i.e., square, cube, tesseract) 
region to operate upon to reduce data volume and therefore processing time. [Python and numpy's slicing syntax](https://www.w3schools.com/python/numpy/numpy_array_slicing.asp)
is a nice way to represent these operations. A quick explanation of the syntax follows.

##### Important Points #####

* Python arrays are zero-indexed. I.e., the first element of an array has the index `0`.

* Negative indices count backwards from the end of the array. If you have `N` entries in an array, `-1` becomes `N-1`, so the slice 
  `0:-1` selects the whole array except the last element.

* Python slices have the format `start:stop:step` when they are used to index arrays via square brackets, e.g., `a[5:25:2]`
  returns a slice of the object `a`. Slices can also be defined by the `slice` object via `slice(start,stop,step)`, e.g.
  `a[slice(5,25,2)]` returns the same slice of object `a` as the last example.

* The `start`, `stop`, and `step` parameters generate indices of an array by iteratively adding `step` to `start` until the
  value is equal to or greater than `stop`. I.e., `selected_indices = start + step * i` where `i = {0, 1, ..., N-1}`, 
  `N = CEILING[(stop - start) / step]`.
  
* Mathematically, a slice specifies a [half-open interval](https://en.wikipedia.org/wiki/Interval_(mathematics)),
  the `step` of a slice selects every `step` entry of that interval. I.e., they include their start point but not their end point, 
  and only select every `step` elements of the interval. E.g., `5:25:2` selects the elements at the indices {5,7,9,11,13,15,17,19,21,23}

* By default `start` is zero, `stop` is the number of elements in the array, and `step` is one.

* Only the first colon in a slice is required to define it. The slice `:` is equivalent to the slice `0:N:1`, where `N` is the number
  of elements in the object being sliced. E.g., `a[:]` selects all the elements of `a`.

* Negative parameters to `start` and `stop` work the same way as negative indices. Negative values to `step` reverse the default values
  of `start` and `stop`, but otherwise work in the same way as positive values.
  
* A slice never extends beyond the beginning or end of an array. I.e., even though negative numbers are valid parameters, a slice that
  would result in an index *before* its `start` will be empty (i.e., select no elements of the array). E.g., If we have an array with 5
  entries, the slice `1:3:-1` **does not** select {1, 0, 4}, it is empty.

##### Details #####

Let `a` be a 1 dimensional array, such that `a = np.array([10,11,15,16,19,20])`, selecting an element of the array
is done via square brackets `a[1]` is the 1^th element, and as python is 0-indexed is equal to 11 for our example array.

Slicing is also done via square brackets, instead of a number (that would select and element), we pass a slice. Slices are
defined via the format `<start>:<stop>:<step>`. Where `<start>` is the first index to include in the slice, `<stop>` is the
first index to **not** include in the slice, and `<step>` is what to add to the previously included index to get the next
included index.

E.g., 
* `2:5:1` includes the indices 2,3,4 in the slice. So `a[2:5:1]` would select 15,16,19.
* `0:4:2` includes the indices 0,2 in the slice. So `a[0:4:2]` would select 10,15.

Everything in a slice is optional except the first colon. The defaults of everything are as follows:
* `<start>` defaults to 0
* `<stop>` defaults to the last + 1 index in the array. As python supports negative indexing (-ve indices "wrap around", with 
  -1 being the index **after** the largest index) this is often called the -1^th index, or that `<stop>` defaults to -1.
* `<step>` defaults to 1

Therefore, the slice `:` selects all of the array, and the slice `::-1` selects all of the array, but reverses the ordering.

When dealing with N-dimensional arrays, indexing accepts a tuple. 
E.g., for a 2-dimensional array `b=np.array([[10,11,15],[16,19,20],[33,35,36]])`, 
* `b[1,2]` is equal to 20
* `b[2,1]` is equal to 35

Similarly, slicing an N-dimensional array uses tuples of slices. E.g.,
```python
>>> b
array([[10, 11, 15],
	   [16, 19, 20],
	   [33, 35, 36]])
>>> b[::-1,:]
array([[33, 35, 36],
	   [16, 19, 20],
	   [10, 11, 15]])
>>> b[1:2,::-1]
array([[20, 19, 16]])
```

Slices and indices can be mixed, so you can slice one dimension and select an index from another. E.g.,
```python
>>> b[:1, 2]
array([15])
>>> b[::-1, 0]
array([33, 16, 10])
>>> b[0,::-1]
array([15, 11, 10])
```

There is a `slice` object in Python that can be used to programmatically create slices, its prototype is `slice(start,stop,step)`,
but only `stop` is required, and if `stop=None` the slice will continue until the end of the array dimension. Slice objects
are almost interchangeable with the slice syntax. E.g.,
```python
>>> s = slice(2)
>>> b[s,0]
array([10, 16])
>>> b[:2,0]
array([10, 16])
```

