# The Effect of Local Optimization on the Multiple Traveling Salesman Problem
Code used for the thesis "The Effect of Local Optimization on the Multiple Traveling Salesman Problem".

The repository contains 3 folders:
* datasets
* mtsp
* results

The **datasets** folder contains the datasets used in the research. The areas sub-folder contains 3 areas defined by polygons, of which Amsterdam is used in the Thesis. The thesis sub-folder contains all datasets generated and used for the research.

The **mtsp** folder contains all code used during the research.
* The **algorithms** sub-folder contains all algorithms written for the research 
* The **classes** sub-folder contains the mtsp as well as the solution classes
* The **data_scripts** sub-folder contains the code used to generate data sets from the area files, as well as their distance matrices
* The **research** sub-folder contains examples of running the algorithms on the data sets
* The **visualize** sub-folder contains files used to create the plots in the research, note that they were hard-coded for specific plots, and tweaking is required to create own plots.

The **results** folder contains the **figures** used in the thesis. It also contains table.csv, which holds contains the means, standard deviations and minimum and maximum value found for every algorithm/optimization combination.
In **results\thesis** the first results to the first 10 instances can be found, due to memory constraints. This folder also contains the averages over all 1000 instances.

Running the code can be done from the top level repository, in **test.py** some examples are given.

The **thesis.yml** file contains the environment, and can be read with i.e. conda.

Bram van den Heuvel, 2024