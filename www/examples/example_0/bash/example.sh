# Zenodo download example

: << ---MD
---
layout: bare
---

## Downloading Example Datasets ##

Example datasets are hosted on [zenodo]() at the following URLs:

Small Dataset
: https://zenodo.org/records/13384454/files/aopp_deconv_tool_example_datasets_small.tar?download=1

Use the \`curl\` and \`tar\` commands to download and extract the dataset.
---MD

#:begin{CELL}
EXAMPLE_DATA_DIR="./example_data"
URL="https://zenodo.org/records/13384454/files/aopp_deconv_tool_example_datasets_small.tar?download=1"

# Create a folder for the dataset
mkdir -p ${EXAMPLE_DATA_DIR}

# Download the dataset
curl -o ${EXAMPLE_DATA_DIR}/small_dataset.tar ${URL}

# Extract the dataset
cd ${EXAMPLE_DATA_DIR}
tar -xf ./small_dataset.tar

# Remove the downloaded archive as we have extracted its contents
rm -f ./small_dataset.tar
cd ..

# Print the files extracted from the example dataset
echo "Example dataset:"
ls ${EXAMPLE_DATA_DIR}

#:HIDE
rm -rf ${EXAMPLE_DATA_DIR}

#:end{CELL}