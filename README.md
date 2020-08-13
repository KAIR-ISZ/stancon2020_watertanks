In this repository we present the data and code used for our presentation at StanCon 2020 titled "Process fault detection using Stan"
by Jerzy Baranowski, Waldemar Bauer, Rafał Mularczyk, Bartłomiej Gondek from AGH University of Science & Technology, Kraków, Poland

Folder Data has its own readme and can be easily used.


The src folder contains the scripts necessary to recreate the analyzes presented at the Stancon2020 conference.

Secret descriptions:
- main.py - the script processes data from experiments describing the process of filling the tanks. As a result, we get ready-to-use models in the form of pickle files
- main_pcc.py - script performs piore predictive computed for the selected trajectory. As a result, we get ready-to-use models in the form of pickle files.
- depth_visualisation.py - the script allows you to analyze the depth of probability using Mahalanobis distance for simulation data
- depth_visualisation_exp.py - the script allows you to analyze the depth of probability using the Mahalanobis distance for data from experiments on a real object

Commissioning in the terminal:
1. Go to the src folder
2. Enter: `python main.py`
3. Enter: `python main_ppc.py`
4. Enter: `python depth_visualisation.py`
5. Enter: `python depth_visualisation_exp.py`

**Important**: all scripts create appropriate data structures and save them as pickle files. for proper operation, run them in the order given.
