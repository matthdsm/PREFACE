# Background

A share of all cell-free DNA fragments isolated from maternal plasma during pregnancy is fetal-derived. This amount is referred to as the 'fetal fraction' and represents an important estimate during routine noninvasive prenatal testing (NIPT). Its most essential role is informing geneticists whether an assay is conclusive: if the fetal fraction is insufficient (this limit has often been debated to be 4%) claims on fetal aneuploidies cannot be made accurately. Several techniques exist to deduce this figure, but the far most require additional experimental procedures, which impede routine execution. Therefore, we set out to develop PREFACE, a software to accurately predict fetal fraction based on solely shallow-depth whole-genome sequencing data, which is the fundamental base of a default NIPT assay. In contrast to previous efforts, PREFACE enables user-friendly model training with a limited amount of retrospective data, which eliminates between-laboratory bias. For sets of roughly 1100 male NIPT samples, a cross-validated correlation of 0.9 between predictions and fetal fractions according to Y chromosomal read counts was noted (FFY). Our approach enables training with both male and unlabeled female fetuses: using our complete cohort (nfemale=2468, nmale=2723), the correlation metric reached 0.94. In addition, PREFACE provides the fetal fraction based on the copy number state of chromosome X (FFX). The presented statistics indirectly predict mixed multiple pregnancies, the source of observed events and sex chromosomal aneuploidies. All details can be found in our [corresponding paper](https://www.ncbi.nlm.nih.gov/pubmed/31219182).  

# Manual

## Required files

### Copy number alteration .bed files

Each sample (whether it is used for training or for predicting) should be passed to PREFACE in the format shown below. During benchmarking, using a bin size of 100 kb (others might work equally well), copy number normalization was performed by [WisecondorX](https://github.com/CenterForMedicalGeneticsGhent/WisecondorX/), yet PREFACE is not limited to any copy number alteration software, however, the default output of WisecondorX is directly interpretable by PREFACE.  

- Example: ```./examples/ratios.bed```  
- Tab-separated file with at least four columns.  
- The name of these columns (passed as a header) must be 'chr', 'start', 'end' and 'ratio'.  
    - The possible values of 'chr' are 1 until 22, and X and Y (uppercase).  
    - The 'ratio' column contains the log2-transformed ratio between the observed and expected copy number.  
    - The ratio can be unknown at certain loci (e.g. often seen at centromeres). Here, values should be expressed as 'NaN' or 'NA'.  
- The order of rows does not matter. Yet, it is paramount that, for a certain line, file x deals with the same locus as file y. This implies, of course, that all copy number alteration files have the same number of lines.  

### PREFACE's samplesheet.tsv

For training, PREFACE requires a samplesheet file.  

- Example: ```./examples/samplesheet.tsv```  
- TSV file with at least four columns.  
- The name of these columns (passed as a header) must be 'ID', 'filepath', 'sex' and 'FF'.  
    - 'ID' is used to specify a mandatory unique identifier to each of the samples.  
    - The 'filepath' column holds the full absolute path of the training copy number alteration files (.bed).  
    - The possible values for 'sex' are either 'M' (male) or 'F' (female), representing fetal gender.  
    - The 'FF' column contains the response variable (the 'true' fetal fraction). One can use any method he/she believes performs best at quantifying the actual fetal fraction. PREFACE was benchmarked using the number of mapped Y-reads, referred to as FFY.

## Installation & Setup

PREFACE is a Python package that can be installed using `pip`.

```bash
pip install .
```

This will install the `PREFACE` command-line tool.

## Model training

```bash
PREFACE train --samplesheet path/to/samplesheet.tsv [optional arguments]
```

| Optional argument | Function |
| :--- | :--- |
| `--impute` | Impute missing values instead of assuming zero. |
| `--exclude-chrs` | Chromosomes to exclude from training (default: 13, 18, 21, X, Y). |
| `--nfolds x` | Number of folds for cross-validation (default: 5). |
| `--nfeat x` | Number of features (PCA components) (default: 50). |
| `--tune` | Enable automatic hyperparameter tuning. |
| `--model [neural\|xgboost]` | Type of model to train (default: neural). |

## Predicting

```bash
PREFACE predict --infile path/to/infile.bed --model path/to/model_directory
```

| Argument | Function |
| :--- | :--- |
| `--infile` | Path to input BED file. |
| `--model` | Path to the trained model directory. |

## Model optimization  

- The most important parameter is `--nfeat`:  
    - It represents the number of principal components (PCs) that will be used as features during model training. Depending on the used copy number alteration software, bin size and the number of training samples, it might have different optimal values. In general, I recommend to train a model using the default parameters. The output will contain a plot that enables you to review the selected `--nfeat`. Two parts should be seen in the proportion of variance across the principal components (indexed in order of importance):  
        - A 'random' phase (representing PCs that explain variance caused by, inter alia, fetal fraction).  
        - A 'non-random' phase (representing PCs that explain variance caused by natural Gaussian noise).  
    - An optimal `--nfeat` captures the 'random' phase (as shown in the example at `./examples/overall_performance.png`). Capturing too much of the 'non-random' phase could lead to convergence problems during modeling.  
    - If you are not satisfied with the performance of your model or with the position of `--nfeat`, re-run with a different number of features.  
- Note that the final model will probably be a bit more accurate than what is claimed by the performance statistics. This is because PREFACE uses a cross-validation strategy where a subset of samples are excluded from training, after which these serve as validation cases. This process is repeated `n` times (default 5). Therefore, the final performance measurements are based on models trained with only partial data, yet the resulting model is trained with all provided cases.  

# Utilities

## NPZ to Parquet Converter

This script converts NumPy `.npz` files into one or more Parquet files, facilitating easier exploration and analysis of the stored numerical data using tools like Pandas.

### Usage

```bash
PREFACE utils npz-to-parquet <npz_file1> [<npz_file2> ...] [-o <output_directory>]
```

- `<npz_file1> [<npz_file2> ...]`: One or more paths to the input `.npz` files.
- `-o, --output-dir`: (Optional) Directory to save the output Parquet files. Defaults to the current directory (`.`).

## FFY Calculator

Calculates Fetal Fraction from Y-chromosome reads (from WisecondorX NPZ output).

### Usage

```bash
PREFACE utils ffy <wisecondorx_npz> [--sex-cutoff <cutoff>]
```

- `<wisecondorx_npz>`: Path to WisecondorX output NPZ file.
- `--sex-cutoff`: (Optional) Cutoff for sex determination (default: 0.2).