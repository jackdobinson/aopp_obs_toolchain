# Examples #

The data for examples in this section are hosted on {% include stub-link.html text="zenodo" %}.

{% capture includeContent %}
{% include_relative ./jupyter/example_1.html %}
{% endcapture %}
{% include concertina.html heading="Deconvolution of a singe wavelength FITS file" content=includeContent %}

<!--
[Deconvolution of a single wavelength image](./jupyter/example_1.html)
: Starting with a science observation and a standard star. The science observation is deconvolved, various common problems are encountered and resolved.
-->