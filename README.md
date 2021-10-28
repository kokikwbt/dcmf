# DCMF: Dynamic Contextual Matrix Factorization

**Unofficial** Python implementation of the DCMF algorithm:  
Y. Cai, H. Tong, W. Fan, and P. Ji. [Fast Mining of a Network of Coevolving Time Series](https://doi.org/10.1137/1.9781611974010.34). In SDM, 2015.  


## DEMO

Please RUN:
```sh
python3 dcmf.py --demo
```
It will start processing the DCMF algorithm over a toy dataset
with a synthetic contextual matrix. The results will appear in `out` directory.

## Comand line options

- `--output_dir`: path of output directory (default: 'out')
- `--n_components`: dimension of latent variables (default: 2)
- `--alpha`: weight of contextual infromatioin (default: 0.1)
- `--max_iter`: maximum number of the EM algorithm (default: 100)
- `--tol`: tolerance for early stopping (default: 0.1)

## Citation
```bibtex
@inproceedings{cai2015fast,
  title={Fast mining of a network of coevolving time series},
  author={Cai, Yongjie and Tong, Hanghang and Fan, Wei and Ji, Ping},
  booktitle={Proceedings of the 2015 SIAM International Conference on Data Mining},
  pages={298--306},
  year={2015},
  organization={SIAM}
```