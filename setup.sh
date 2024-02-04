#!/bin/bash

# Install requirements
conda env create -f requirements.yaml

# Create basic folders
mkdir sfx
mkdir generated
mkdir dectalk

# Install dectalk
cd dectalk
wget https://github.com/dectalk/dectalk/releases/download/2023-10-30/ubuntu-latest.tar.gz && tar -xf ubuntu-latest.tar.gz