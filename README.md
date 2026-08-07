# VAEAC â€” Variational Autoencoders for Explainability

A study of Variational Autoencoders for model explainability, with an emphasis on the role of the Gaussian prior and Shapley-based explanations.

## Research question

The project investigates how a VAE-based conditional generation mechanism can be used to analyze model behavior and how prior assumptions influence the resulting explanations.

## Contents

- `Projet_VAEAC_MÃ©thode_de_simulation.ipynb` â€” simulation and experimental notebook
- `analyse du prior gaussien sur le VAEAC et lâ€™impact sur lâ€™explicabiliteÌ via Shapleys.pdf` â€” written analysis of the Gaussian prior and Shapley explanations

## Running the notebook

Open the notebook in Jupyter or Google Colab and execute the cells in order. The exact dependencies and data assumptions are documented in the notebook itself.

```bash
jupyter notebook Projet_VAEAC_MÃ©thode_de_simulation.ipynb
```

## Scope

This repository is an academic research project and an exploratory implementation. Explanation quality depends on the generative model, the data distribution and the assumptions used to define the conditional samples; the results should not be interpreted as universally valid explanations.