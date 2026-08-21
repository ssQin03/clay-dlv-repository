# Data repository for "A Universal Explicit Model for Double-Layer Repulsion between Clay Particles in Discrete Element Simulations"

This repository contains the datasets, source code, and trained neural network models supporting the findings of the manuscript submitted to the European Journal of Environmental and Civil Engineering.

## Repository structure

- `matlab/` -- Numerical solutions of the nonlinear Poisson-Boltzmann equation
  - `poten_mid_full.mat` -- Dimensionless mid-plane potential under the constant-potential (CP) condition: 500 surface potentials z x 500 separation distances D (250,000 data points)
  - `poten_mid_ce_full.mat` -- Dimensionless mid-plane potential under the constant-charge (CC) condition: 78 charge densities p x 500 separation distances D (39,000 data points)
  - `midpotential_full_cp.m`, `midpotential_full_ce.m` -- MATLAB scripts that generate the numerical PBE solutions
  - `midpotential_cp_validation.m`, `midpotential_ce_validation.m` -- Validation scripts
- `nn_unified_model_final.py` -- Training and evaluation code for the unified mid-plane potential neural network (CP and CC in a single model; architecture 3-128-64-32-1 with Layer Normalization and SiLU; 11,329 trainable parameters)
- `nn_energy_model.py` -- Training and evaluation code for the dimensionless double-layer repulsive energy neural network
- `models/`
  - `unified_nn_model.pt` -- Trained mid-plane potential NN (PyTorch state dict)
  - `energy_nn_model.pt` -- Trained energy NN (PyTorch state dict)
  - `metrics_midplane.json` -- Test-set metrics reported in Table 2 (NN and explicit formulas, Appendix I/II coefficients)
  - `metrics_energy.json` -- Test-set metrics reported in Table 3

## Requirements

- MATLAB R2018b or later (for the numerical PBE solutions)
- Python >= 3.9 with numpy, scipy, torch >= 2.0, matplotlib

## Reproduction

1. Regenerate the numerical PBE solutions: run `matlab/midpotential_full_cp.m` and `matlab/midpotential_full_ce.m`.
2. Retrain the mid-plane potential NN: `python nn_unified_model_final.py` (test-set metrics are written to `models/metrics_midplane.json`).
3. Retrain the energy NN: `python nn_energy_model.py` (test-set metrics are written to `models/metrics_energy.json`).

The explicit fitting formulas with the piecewise coefficients are given in Appendices I-IV of the manuscript.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

Please cite this dataset as: Shang, X., et al. (2026). Data repository for "A Universal Explicit Model for Double-Layer Repulsion between Clay Particles in Discrete Element Simulations" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
