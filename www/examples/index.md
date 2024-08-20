---
examples_to_embed:
  - name: Deconvolution of a single wavelength
    versions:
      - name: Python
        url: /examples/jupyter/example_1.html
      - name: Bash
        url: /examples/bash/example_1.html
---


# Examples #

The data for examples in this section are hosted on {% include stub-link.html text="zenodo" %}.

{% for example in page.examples_to_embed %}
	{{example.name}}
	{{example.versions}}
	{% include concertina-page.html heading="<h2>{{example.name}}</h2>" tabs=example.versions %}
{% endfor %}