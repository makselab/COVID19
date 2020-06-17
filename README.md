# Project COVID19 K-core tracker

Source code for the paper titiled:
Superspreading k-cores at the center of pandemic persiste

The code consist of two parts:

## 1. Generating close contact network from raw mobile tracing data. 

This part of code also serve as the back end calculation of our actual pipeline. It requires the deployment of elasticsearch.

Please refer to the repository at staging folder: https://github.com/shaojunluo/mobile_covid19_tracker and follow the instructions.

## 2. Analysis code on top of close contact network. 

Please refer the `document.pdf` for detail information about the usage.
