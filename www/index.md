
{% 
	include message.html 
		level="note" 
		message="This site is currently a work in progress. The various sections of the site are being filled in and are in various stages of completion. Please see the <a href='USAGE.md'>usage</a> for a big markdown file containing a large amount of documentation for the *aopp_obs_toolchain* repository." 
%}


# AOPP Obs Toolchain #

*AOPP Obs Toolchain* is a set of tools, examples, and tutorials used to process observational data that can be used as a resource by the professional and amateur community. Currently only *AOPP Deconv Tool* (a Python package designed to apply the Modified CLEAN algorithm to observations in the FITS file format) exists but we hope to include other tools/packages.

The toolchain is hosted on [github](https://github.com/jackdobinson/aopp_obs_toolchain), with the documentation available on this site, and data for the examples are hosted on [zenodo]({{ site.baseurl }}/assets/md/link-not-present.md).


## Toolchain Structure ##

* AOPP Obs Toolchain
  : Top level collection of tools and packages.
  
  - AOPP Deconv Tool
    : Apply the Modified CLEAN algorithm to observations in the FITS file format

