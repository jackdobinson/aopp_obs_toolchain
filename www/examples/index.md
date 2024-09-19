---
examples_to_embed:
  - name: Downloading Example Datasets
    versions:
      - name: Python
        url: examples/example_0/jupyter/example.html
      - name: Bash
        url: examples/example_0/bash/example.html
  - name: Deconvolution of a single wavelength
    versions:
      - name: Python
        url: examples/example_1/jupyter/example.html
      - name: Bash
        url: examples/example_1/bash/example.html
  - name: Deconvolution of a TIFF file
    versions:
      - name: Python
        url: examples/example_2/jupyter/example.html
---


# Examples #

The data for examples in this section are hosted on [Zenodo](https://zenodo.org/records/13384454).

The number of examples will be expanded as time goes on as more use cases arise. If there is an example not present that would be helpful to you, or you wish to contribute an example, please do so via the [github issues](https://github.com/jackdobinson/aopp_obs_toolchain/issues) for the project.

{% for example in page.examples_to_embed %}
{% capture example_summary %}
<h2>{{example.name}}</h2>
{% endcapture %}
{% include concertina-page.html heading=example_summary tabs=example.versions %}
{% endfor %}