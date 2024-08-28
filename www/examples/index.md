---
examples_to_embed:
  - name: Downloading Example Datasets
    versions:
      - name: Bash
        url: examples/example_0/example.html
  - name: Deconvolution of a single wavelength
    versions:
      - name: Python
        url: examples/example_1/jupyter/example.html
      - name: Bash
        url: examples/example_1/bash/example.html
---


# Examples #

The data for examples in this section are hosted on {% include stub-link.html text="zenodo" %}.

{% for example in page.examples_to_embed %}
{% capture example_summary %}
<h2>{{example.name}}</h2>
{% endcapture %}
{% include concertina-page.html heading=example_summary tabs=example.versions %}
{% endfor %}