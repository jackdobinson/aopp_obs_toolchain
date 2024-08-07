
## FITS File Format Information <a id="fits-file-format-information"></a> ##

Documentation for the Flexible Image Transport System (FITS) file format is hosted at [NASA's Goddard Space Flight centre](https://fits.gsfc.nasa.gov/fits_standard.html), please refer to that as the authoritative source of information. What follows is a brief description of the format to aid understanding, see below for a [schematic of a fits file](#fits-file-schematic).

A FITS file consists of one or more *header data units" (HDUs). An HDU contains header and (optionally) data information. The first HDU in a file is the "primary" HDU, and others are "extension" HDUs. The primary HDU always holds image data, extension HDUs can hold other types of data (not just images, but tables and anything else specified by the standard). An HDU always has a number which describes it's order in the FITS file, and can optionally have a name. Naming an HDU is always a good idea as it helps users navigate the file. NOTE: The terms "extension", "HDU", and "backplane" are used fairly interchangeably to mean HDU

Within each HDU there is header-data and (optionally) binary-data. The header-data consists of keys and values stored as restricted [ASCII](https://en.wikipedia.org/wiki/ASCII) strings of 80 characters in total. I.e., the whole key+value string must be 80 characters, they can be padded with spaces on the right. Practically, you can have as many header key-value entries as you have memory for. There are some reserved keys that define how the binary data of the HDU is to be interpreted. NOTE: Keys can only consist of uppercase latin letters, underscores, dashes, and numerals. The binary-data of an HDU is stored bin-endian, and intended to be read as a byte stream. The header-data describes how to read the binary-data, the most common data is image data and tabular data.

Fits image HDUs (and the primary HDU) define the image data via the following header keywords.

BITPIX
: The magnitude is the number of bits in each pixel value. Negative values are floats, positive values are integers.

NAXIS
: The number of axes the image data has, from 0->999 (inclusive).

NAXISn
: The number of elements along axis "n" of the image.

Relating an axis to a coordinate system is done via more keywords that define a world coordinate system (WCS), that maps integer pixel indices to floating-point coordinates in, for example, time, sky position, spectral frequency, etc. The specifications for this are suggestions rather than rules, and non-conforming FITS files are not too hard to find. As details are what the spec is for, here is a high-level overview. Pixel indices are linearly transformed into "intermediate pixel coordinates", which are rescaled to physical units as "intermediate world coordinates", which are then projected/offset/have some (possibly non-linear) function applied to get them to "world coordinates". The CTYPE keyword for an axis describes what kind of axis it is, i.e., sky-position, spectral frequency, time, etc.

Therefore, when using a FITS file it is important to specify which HDU (extension) to use, which axes of an image correspond to what physical coordinates, and sometimes what subset of the binary-data we want to operate upon.

#### Fits File Schematic <a id="fits-file-schematic"></a> ####
```
FITS FILE
|- Primary HDU
|  |- header data
|  |- binary data (optional)
|- Extension HDU (optional)
|  |- header data
|  |- binary data (optional)
|- Extension HDU (optional)
|  |- header data
|  |- binary data (optional)
.
.
.
```

### FITS Data Order <a id="fits-data-order"></a> ###

FITS files use the FORTRAN convention of **column-major ordering**, whereas Python uses **row-major ordering** (sometimes called "C" ordering). For example, if we have an N-dimensional matrix, then we can specify a number in that matrix by its indices (e_1, e_2, e_3, ..., e_N). FITS files store data so that the **left** most index changes the fastest, i.e., in memory the data is stored {a_11, a_21, a_31, ..., a_M1, a_M2, ..., a_ML}. However, Python stores its data where the **right** most index changes the fastest, i.e., data is stored in memory as {a_11, a_12, a_13, ..., a_1L, a_2L, a_3L, ..., a_NL}. Also, just to make things more difficult FITS (and Fortran) start indices at 1 whereas Python (and C) start indices at 0. The upshot of all of this is that if you have data in a FITS file, the axis numbers are related via the equation 

N - f_i = p_i

where N is the number of dimensions, f_i is the FITS/FOTRAN axis number, p_i is the Python/C axis number.

The upshot is that even though both FITS and Python label the axes of an array from left-to-right, (the "leftmost" axis being 0 for python, 1 for FITS), the ordering of the data in memory means that when reading a FITS array in Python, the axes are reversed.

Example:
```

Let `x` be a 3 by 4 matrix, it has two axes.
The FOTRAN convention is they are labelled 1 and 2,
the C convention is that they are labelled 0 and 1.
To make it obvious when we are talking about axes numbers
in c or fortran I will use (f1, f2) for fortran, and
(c0, c1) for c.

	/ a b c d \
x = | e f g h |
	\ i j k l /

Assume `x` is stored as a 2 dimensional array.

In the FORTRAN convention, the matrix is stored in memory as
{a e i b f j c g k d h l}, i.e., the COLUMNS vary the fastest

|offset from start | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|
|------------------|--|--|--|--|--|--|--|--|--|--|--|--|
|value             | a| e| i| b| f| j| c| g| k| d| h| l|

Therefore, the memory offset of an element from the start of
memory is 
m_i = (row-1) + number_of_rows * (column-1)

In FOTRAN, the left most index varies the fastest, and indices
start from 1 so to extract a single number from x we index it via
x[row, column]
e.g., x[2,3] is an offset of (2-1)+3*(3-1) = 7, which selects 'g'.

In the C convention, the matrix is stored in memory as
{a b c d e f g h i j k l}, i.e., the ROWS vary the fastest

|offset from start | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|
|------------------|--|--|--|--|--|--|--|--|--|--|--|--|
|value             | a| b| c| d| e| f| g| h| i| j| k| l|

Therefore, the memory offset of an element from the start of
memory is 
m_i = number_of_columns * (row) + column

In C, the right most index varies the fastest, and indices
start at 0 so to extract a single number from x we index it via
x[row, column]
E.g., x[1,2] is an offset of 4*1+2 = 6, which selects 'g' also.

Wait, these are the same (except the offset of 1)?! That is because
we just looked at NATIVE data ordering. I.e., when FOTRAN and C have
data they have written themselves.

What if we get C to read a FORTRAN written array?

The data in memory is stored as
{a e i b f j c g k d h l}

If we index this the same way we did before, using x[row, column]
we will have a problem.
E.g., x[1,2] is an offset of 4*1+2 = 6, which selects 'c', not 'g'!

This is happening because C assumes the axis that varies the fastest
is the RIGHT-MOST axis. But this data is written so the fastest
varing axis is the LEFT-MOST axis. So we should swap them around
and use x[column, row].
E.g., 
For C; m_i = number_of_columns * (row) + column. Therefore,
x[2,1] is an offset of 4*2+1 = 9, but that selects 'd', not 'g'!?

This fails because we forgot to swap around the formula for the
memory offset. The better way of writing the memory offset formula
is

m_i = number_of_entries_in_fastest_varying_axis * (slowest_varying_axis_index) 
	  + fastest_varying_axis_index

And we know that for the data being written this way, the fastest
varying axis has 3 entries not 4.

E.g., x[2,1] is an offset of 3*2+1 = 7, which that selects 'g', success!

What we have actually done is just make sure that we read the data
the way it was written, in C the fastest varying axis is the RIGHT-MOST
axis when indexing, so we have to reverse the indices AND the lengths
of the axes. Therefore, data written in FORTRAN with axes (f1, f2, ... fN) and
axis lengths (K, L, ..., M), should be read in C with axes ordered as
(c0 = fN, c1 = fN-1, ..., cN-2 = f2, cN-1 = f1) and lengths (M, ..., L, K).

I.e., The fastest varying axis should always go at the correct position
LEFT in FORTRAN and RIGHT in C.

Confusion happens because:

* FORTRAN and C order axes from left-to-right
  - FORTRAN starts at 1
  - C starts at 0

* The **data order** is different between FOTRAN and C
  - FORTRAN writes data by varying the left-most axis the fastest
  - C writes data by varying the right-most axis the fastest
  
* FORTRAN and C index an N-dimensional **native data order** array the
  same way e.g., x[row, column]
  - Axes numbers are based on the **order in the written expression**
  - FORTRAN numbers these axes as row=1, column=2
  - C numbers these axes as row=0, column=1

* When reading data, it should always be read how it is ordered **in memory**
  for native data order this is as above, but for **non native data order**
  the axes order is reversed as it **requires the axis speeds to be the same**.
  - C should read data as x[slowest, fastest]
  - FORTRAN should read data as x[fastest, slowest]
  - So x[row, column] written by FORTRAN is read as x[column, row] by C
	FORTRAN numbers them row=1, column=2; C numbers them column=0, row=1.
  - In FOTRAN **low** axis numbers are fastest
  - In C **high** axis numbers are fastest


NOTE: axis numbers are always from the left hand side.
--------------------------------------
|FORTRAN     | In memory     |C axis |
|axis number | varying speed |number |
|------------|---------------|-------|
|     N      |   slow        |   0   |
|    N-1     |   slower      |   1   |
|    ...     |   ...         |  ...  |
|     2      |   fast        |  N-2  |
|     1      |   fastest     |  N-1  |
--------------------------------------

```





