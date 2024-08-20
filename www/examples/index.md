---
examples_to_embed:
  - name: Deconvolution of a single wavelength
    versions:
      - name: Python
        url: examples/jupyter/example_1.html
      - name: Bash
        url: examples/bash/example_1.md
---


# Examples #

The data for examples in this section are hosted on {% include stub-link.html text="zenodo" %}.

{% for example in page.examples_to_embed %}
{% capture example_summary %}
<h2>{{example.name}}</h2>
{% endcapture %}
{% include concertina-page.html heading=example_summary tabs=example.versions %}
{% endfor %}