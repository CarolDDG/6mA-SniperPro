# 6mA-SniperPro

6mA-SniperPro is a pipeline for identifying N6-methyladenine (6mA) sites in mammalian PacBio SMRT sequencing data. It integrates case-control IPD signal analysis with a CCS-level attention model and post-distribution tests to generate high-confidence candidate sites.

## Repository structure

- `Identification/` - core analysis pipeline and workflow scripts
  - `main.py` - main entrypoint for 6mA identification
  - `main_check.py` - alternate entrypoint with the same argument interface
  - `utils.py` - helper functions for alignment reading, signal extraction, statistics, and feature generation
  - `workerProcess.py` - parallel worker processes for windowed analysis
  - `resultProcess.py` - result collection and TSV writer
- `model/` - model definition and pretrained weights
  - `model_ccs_attn.py` - CCS-level attention model with sigmoid product pooling
  - `state_dict.pt` - pretrained model weights
- `demo/` - sample data for quick testing

## Overview

The pipeline supports analysis of PacBio CCS alignments in a case-control framework. It performs:

1. hit preprocessing and filtering for case and control samples
2. statistical site selection based on IPD signal and shuffled background statistics
3. feature extraction from subreads around candidate positions
4. deep-learning model inference to predict methylation probability
5. post-filtering using dip test and KS 2-sample test on case/control distributions

## Requirements

The project depends on the following Python packages:

- `pbcore`
- `pbcommand`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `torch`
- `pytorch-lightning`
- `recordtype`
- `unidip`
- `matplotlib`

> Note: There is no `requirements.txt` in this repository, so install dependencies manually or create a Conda environment.

## Installation

Example Conda setup:

```bash
conda create -n 6mA-SniperPro python=3.10
conda activate 6mA-SniperPro
pip install numpy pandas scipy scikit-learn torch pytorch-lightning recordtype unidip matplotlib pbcore pbcommand
```

## Usage

Run the main pipeline from the `Identification/` directory:

```bash
python Identification/main.py <case_alignment_set> \
  --control <control_alignment_set> \
  --reference <reference.fasta> \
  --outputFile <output_root> \
  --modelPath model \
  --modelDict model/state_dict.pt
```

Required arguments:

- `alignment_set` - case sample BAM or AlignmentSet
- `--control` - control sample BAM or AlignmentSet
- `--reference` - reference FASTA or Reference DataSet
- `--outputFile` - root name for output files

Common optional parameters:

- `--outputType` - `full` or `clean`
- `--numWorkers` / `-j` - number of worker processes
- `--sigQcutoff` - significance q-value cutoff for initial site selection
- `--minCoverage` - minimum CCS coverage per read
- `--minCoverage_site` - minimum CCS coverage per candidate site
- `--COVERAGE_model` - minimum coverage for model prediction
- `--FREQ_model` - model frequency cutoff for site-level prediction
- `--THRESHOLD_model` - model probability threshold
- `--dip_P` - dip test p-value cutoff
- `--ks2_P` - KS 2-sample test p-value cutoff

For a complete list of parameters, run:

```bash
python Identification/main.py -h
```

## Method details

### Data preprocessing

- Alignments are read using `pbcore`.
- Each genomic window is processed in parallel using worker processes.
- Reads are filtered by mapping quality and length before downstream analysis.

### Statistical site selection

- Case and control signal distributions are compared to identify candidate sites.
- The pipeline applies multiple filtering steps and shuffled background controls.

### Deep learning model

- The model in `model/model_ccs_attn.py` is a CCS-level attention network.
- It uses convolutional feature extraction across CCS subreads and sigmoid product pooling.
- Pretrained weights are loaded from `model/state_dict.pt` during inference.

### Post-filtering

- Candidate sites are filtered again using distribution tests:
  - dip test (dip.P)
  - KS 2-sample test (ks2.P)

## Output

The main output file is `<outputFile>.stat.tsv`. It contains site-level information such as:

- `Loc` - genomic position
- `ID` - site identifier
- `SigQvNum_stat` - number of significant q-values
- `Coverage_stat` - supporting CCS coverage
- `ShufSigQvNum_stat` - shuffled background significance count
- `Coverage_shufStat` - shuffled background coverage
- `Freq_ypred` - model-predicted methylation frequency
- `Coverage_ypred` - model coverage used in prediction
- `dip.pval` - dip test p-value
- `ks2.pval` - KS 2-sample test p-value
- `Type` - candidate classification label

## Demo

A small demo dataset is provided in `demo/` for quick verification. Use it to test the pipeline structure before running a full analysis.

Example:

```bash
python Identification/main.py demo/WGA.chr11_70000000_70100000.mm10.bam \
  --control demo/WGA.chr11_70000000_70100000.mm10.bam \
  --reference <reference.fasta> \
  --outputFile demo/demo_out \
  --modelPath model \
  --modelDict model/state_dict.pt
```

> Replace `<reference.fasta>` with your own reference sequence.

## Notes

- This tool is optimized for mammalian PacBio CCS data.
- Ensure that control and case alignments are matched to the same reference.
- Provide correct paths for `--modelPath` and `--modelDict` because the model files are loaded from those locations.
