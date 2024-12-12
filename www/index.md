
<!--
{% 
	include message.html 
		level="note" 
		message="This site is currently a work in progress. The various sections of the site are being filled in and are in various stages of completion." 
%}
-->


# AOPP Obs Toolchain #

*AOPP Obs Toolchain* is a set of tools, examples, and tutorials used to process observational data that can be used as a resource by the professional and amateur community. Currently only *AOPP Deconv Tool* (a Python package designed to apply the Modified CLEAN algorithm to observations in the FITS file format) exists but we hope to include other tools/packages as we develop them for our own use, and we welcome contributions from others.

## Web Application ##

There is a [web-application version]({{site.baseurl}}/webapp/index.html) of the code that is usable from within your browser. It accepts and returns images in TIFF format.

## Hosting ##

The toolchain is hosted on [github](https://github.com/jackdobinson/aopp_obs_toolchain), with the documentation available on this site, and data for the examples are hosted on [Zenodo](https://zenodo.org/records/13384454).

Please report any bugs via the [github issues](https://github.com/jackdobinson/aopp_obs_toolchain/issues).

## Toolchain Structure ##

AOPP Obs Toolchain
: Top level collection of tools and packages.
  
  [AOPP Deconv Tool](https://pypi.org/project/aopp-deconv-tool/)
  : Apply the Modified CLEAN algorithm to observations in the FITS file format

