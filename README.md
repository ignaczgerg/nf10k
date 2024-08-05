This repository contains the code to reproduce the predictions described in the work "Tipping the chemical separation iceberg: hybrid modeling for nanofiltration in energy-conscious manufacturing" By Gergo Ignacz, Aron K. Beke, Viktor Toth, and Gyorgy Szekely. 

## Requirements
The code is built upon [chemprop-v1](https://github.com/chemprop/chemprop/blob/v1.7.1/README.md) and [descriptastorus](https://github.com/bp-kelley/descriptastorus). The required versions are provided in this repository. All credit for the chemprop and descriptastorus packages goes to their respective authors. Our models needs the following requirements:
- python 3.9
- pytorch 1.10.1
- rdkit 2023.03.3
- numpy 1.23.1
- pandas 1.4.3 
The full list of requirements can be found in the `requirements.txt` file. The code was tested on Ubuntu 20.04.3 WSL. 

## Installation
1) install conda/miniconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2) create a new conda environment: `conda create -n nf10k python=3.9`
3) activate the environment: `conda activate nf10k`
4) install the required packages: `pip install -r requirements.txt`

## Running predictions
Solute rejection prediction can be made by running the `prediction.py` script. The script requires the following arguments:
-    `--smiles <"SMILES string of the molecule">` 
-    `--membrane <membrane>` the script only accepts: `DM300, GMT-oNF-2, PBI, NF90, PMS600, SM122, NF270`
-    `--solvent <solvent_name>` the script only accepts: `Water, Toluene, "Methyl tetrahydrofuran", Methanol, Ethanol, "Dimethyl formamide", Acetonitrile, Acetone, Ethyl acetate`
-    `--configuration <configuration>` the script only accepts: `OSN, "Loose NF", NF`
For example, to predict the rejection of paracematol in methanol in Duramem 300 membrane, run this command: 
-   `python prediction.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --membrane DM300 --solvent Methanol --configuration OSN`
Expected output: `{'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'membrane': 'DM300', 'solvent': 'Methanol', 'configuration': 'OSN', 'rejection': 0.841'}` where the rejection is the predicted rejection of the molecule in the given conditions. There is also `user_predictions.csv` file that contains the predictions of the last run. Re-running the script will overwrote the new predictions to the file.

## Recreating the results of the paper
The results of the paper can be recreated by running the `run_paper_predictions.py` script. The script will run the predictions for all the molecules.. The results will be saved in the `results/paper_predictions.csv`. The script takes around 10 minutes to finish.

## License
This code is released under the MIT License.