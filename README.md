# AMLS_assignment23_24

This is the code base for a Binary and Multi-class medical data classification. This was coded as the final assessment for ELEC0134: AMLS.

## Description

As an exploration of the field of supervised learning and classification, this paper will cover the development, implementation, experimentation and analysis of both binary and multivariate classification techniques and their application within real-world medical scenarios. All the data used within this project was taken from the MedMNIST database, using binary classification to diagnose the presence of Pnumonia within the PnumonaMNIST dataset and deploying multivariate classification to categorise pathology samples of colon cancer within the PathMNIST datasets.

### Dependencies

The code was built in Python, within Windows 10.

It was built within Python Version 3.9.18

A range of packages were used. Including the environment file to duplicate the environment, the code was run within. The key packages used can be found below.

| Package | Version |
| --- | --- |
| numpy | 1.24.1 |
| seaborn | 0.13.0 |
| SciPy | 1.11.3 |
| Scikit learn | 1.3.2 |
| Tensor Flow | 2.15.0 |
| matplotlib | 3.6.3 |

### Installing

Everything is included within the code base. 
Install locally to a file. Next either manually build the environment or create a new environment, importing the provided yml file. The Databases are not included. 
Drag and drop the nzp files of the PnumoniaMNIST and PathMNIST datasets into the Dataset file.

### Executing program

The code is run through the command line. It can accept up to two arguments. If you wish to run the final model please run
```python
python3 main.py
```

If you wish to select what models for each task it is set in the format
```python
python3 main.py <Task A> <Task B>
```

An example to select the models can be seen below,

```python
python3 main.py <cnn> <resnet>
```

Available models include

| Task A | Task B |
| --- | --- |
| svm | resnet |
| svmhog | deep |
| rf | taska |
| adaboost |  |
| cnn |  |

## Hardware and training

Two machines were used to build, train and evaluate the models. The first is a XPS15 9500 and the second is a custom-built water-cooled tower. The second machine saw a significant improvement in training time. The specs of both machines, along with a comprehensive benchmark for the speed of the model are given below.

| Componenet | XPS | Tower |
| --- | --- | --- |
| Processor | Intel i7-10750M | AMD Ryzen 7 5700x |
| OS | Windows 10 |  Windows 10 |
| Ram | 16 GB | 32 GB |
| Graphics Card | 1650 Ti | RTX 3070 |

## File Structure

├───A
│   ├───Graphics
│   │   ├───CNN
│   │   │   └───Layers
│   │   ├───RF
│   │   └───SVM
│   ├───Models
│   │   ├───PreTrainedModels
│   │   └───__pycache__
│   └───__pycache__
├───B
│   ├───Graphics
│   │   └───CNN
│   │       └───Layers
│   ├───Models
│   │   ├───PreTrainedModels
│   │   └───__pycache__
│   └───__pycache__
└───Datasets

## Authors

Contributors names and contact info

Harry R J Softley-Graham  - SN: 19087176

## Aknowledgements

[MedMNIST]([https://choosealicense.com/licenses/mit/](https://medmnist.com/)https://medmnist.com/)
