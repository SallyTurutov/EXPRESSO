## Data Preprocessing

This directory holds the raw GTEx tissue TPM data and scripts for preprocessing it into a format compatible with the tissue representation pipeline.

### Structure

* `gtex_v10_tissue_tpms/` — directory containing downloaded `.gct` files (one per tissue).
* `gtex_tpm_preprocess.py` — filters and reorders gene rows to match a given Reactome gene list and saves the result as `RNA_all_TPM.txt` for each tissue.
* `process_omics_subdirs.py` — processes each `RNA_all_TPM.txt` file into a final `.npy` format for downstream model usage.

### Workflow

1. **Download data**
   Go to the [GTEx download page](https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression) and download the **"Gene TPMs by tissue"** data from **GTEx Analysis V10**.

2. **Place raw `.gct` files**
   Put the downloaded `.gct` files into the `gtex_v10_tissue_tpms/` directory.

3. **Prepare the gene list**
   Ensure `gene_all.txt` is available under `../data/`.

4. **Run `gtex_tpm_preprocess.py`**
   This script reads each `.gct` file, filters/reorders it to match the gene list, and saves a `RNA_all_TPM.txt` file for each tissue under `../data/{tissue_name}/`.

5. **Run `process_omics_subdirs.py`**
   This will convert each `RNA_all_TPM.txt` file to a NumPy array and save it as `data_all.npy` for use in training.

### Note

To use a tissue in your model:

* Make sure it's been preprocessed as described above.
* Add the corresponding `(tissue_enum, filename)` entry to the `TISSUE_TO_DATA_DIR` dictionary in `consts.py`.

