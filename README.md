# NKCS Coevolution

NKCS model for exploring aspects of ([surrogate-assisted](https://en.wikipedia.org/wiki/Surrogate_model)) [coevolution](https://en.wikipedia.org/wiki/Cooperative_coevolution).

The NKCS model is the multi-species version of the [NK model](https://en.wikipedia.org/wiki/NK_model) where S species coevolve with X other species. Each species is composed of N genes where each gene is affected by K genes within the same species (internal connections) and C genes within each of the coevolving species (external connections).

For details see Preen and Bull (2017) [On Design Mining: Coevolution and Surrogate Models](https://arxiv.org/abs/1506.08781) *Artificial Life* 23(2):186-205.

## Usage

Parameters are located in `constants.py`

To run and generate results in the `res` folder:

```
$ python3 -m nkcs.main
```
