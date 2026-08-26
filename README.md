# Amine–Water Mutual Solubility Prediction Using Machine Learning
This repository contains the processed datasets and optimized machine learning stacking models developed for the prediction of mutual solubility in amine–water systems.
The repository is intended to support the reproducibility and transparency of the associated scientific study by providing the datasets used for model development and the final optimized models.

## Repository Contents
The repository includes:
-	two processed datasets used for machine learning model development and evaluation;
-	the optimized code associated with the water-in-amine solubility prediction model;
-	the optimized code associated with the amine-in-water solubility prediction model.

## Datasets
The datasets provided in this repository correspond to the processed data used as input for the machine learning analyses reported in the associated study.
The files included in the `datasets` directory represent the final processed datasets used for model development, optimization and evaluation. The original solubility data were obtained from the IUPAC-NIST Solubility Data Series, specifically from the datasets reported in https://doi.org/10.1063/1.4755288 and https://doi.org/10.1063/1.4755953. The processed datasets provided here correspond to the data used as input to the machine learning workflows described in the associated manuscript. The preprocessing steps applied before model development are described in the associated scientific paper.
### Dataset 1
`water_in_amine_processed.xlsx`
This dataset contains the processed observations used for the development and optimization of the first machine learning model (water-in-amine solubility prediction).
### Dataset 2
`amine_in_water_processed.xlsx`
This dataset contains the processed observations used for the development and optimization of the second machine learning model (amine-in-water solubility prediction).
Detailed descriptions are provided in the associated manuscript and supplementary material.

## Machine Learning Models
Two optimized machine learning models are provided.
### Model 1: Water-in-amine solubility
The script `models/Water_in_amine.py` reproduces the final optimized stacking model developed for the `water_in_amine_processed.xlsx` dataset. The script includes the final model architecture and the hyperparameters selected through the optimization procedure described in the associated manuscript.
### Model 2: Amine-in-water solubility
The script `models/Amine_in_water.py` reproduces the final optimized stacking model developed for the `amine_in_water_processed.xlsx` dataset. The script includes the final model architecture and the hyperparameters selected through the optimization procedure described in the associated manuscript.

## Software Requirements and Usage
The ML models provided in this repository were developed in Python.
To run the scripts, users should have Python installed together with the following packages: `numpy`, `pandas`, `scikit-learn`, `catboost`, `openpyxl`.
The required packages can be installed using:
```bash
pip install numpy pandas scikit-learn catboost openpyxl
```

The original folder structure should be maintained because the Python scripts automatically locate the corresponding datasets using relative file paths.
After downloading or cloning the repository, open a terminal in the main repository directory. 
To run the model for the prediction of amine solubility in water, use:
```bash
python models/Amine_in_water.py
```
To run the model for the prediction of water solubility in amine, use:
```bash
python models/Water_in_amine.py
```
No modification of the dataset paths is required as long as the original repository folder structure is preserved. 

## Reproducibility
The purpose of this repository is to provide the processed datasets and final optimized stacking models used in the associated scientific study.
A fixed random seed (`random_state = 42`) is used where applicable to improve reproducibility of the train/test split, cross-validation procedure, and machine learning models.
Minor differences in numerical results may occur depending on the Python version, operating system, and installed package versions.

## Associated Publication
This repository supports the following scientific study: 
**Explainable machine learning for liquid–liquid equilibrium modeling of amine–water systems in temperature swing solvent extraction desalination**
Stefano Cairone, Tiziano Zarra, Vincenzo Belgiorno, Ngai Yin Yip and Vincenzo Naddeo
*[Journal name]*, [Year].
DOI: [DOI]
The bibliographic information will be updated after publication.

## Citation
If you use the datasets or code provided in this repository, please cite the associated scientific article.
A recommended citation will be added once the article is published.

## Data and Code Availability
The processed datasets and the code required to reproduce the final optimized stacking models are publicly available in this repository. Additional scripts used during model development, hyperparameter optimization, and ancillary analyses are available from the corresponding author upon reasonable request.

## License
The source code in this repository is distributed under the terms specified in the LICENSE file.
Any conditions specifically applicable to the datasets should be considered separately where required.

## Contact
For questions regarding the datasets, methodology, or ML models, please contact:
**Dr. Stefano Cairone**
Sanitary Environmental Engineering Division
Department of Civil Engineering
University of Salerno
Italy
`scairone@unisa.it`
