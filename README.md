# Constraint guaranteed evasion attacks

[![Build](https://github.com/aucad/new-experiments/actions/workflows/build.yml/badge.svg)](https://github.com/aucad/new-experiments/actions/workflows/build.yml)

This implementation demonstrates an approach to introduce constraints to unconstrained adversarial machine learning evasion attacks.
We introduce a constraint validation algorithm, CGE, that guarantees generated evasive adversarial examples satisfy domain constraints.

This implementation allows running various adversarial evasion attacks, enhanced with our constraint validation algorithm, on different data sets and classifiers.
The following options are included.

- **Attacks**: Projected Gradient Descent (PGD), Zeroth-Order Optimization (ZOO), HopSkipJump attack. These attacks are modified to use CGE.
- **Classifiers**: Keras deep neural network and tree-based ensemble XGBoost.
- **Data sets**: 4 different data sets from different domains.

Data set constraints are configurable inputs of an experiment. The configuration files in `config/` show examples of how to specify constraints.

**Comparison.** We also include a comparison attack, Constrained Projected Gradient Descent (C-PGD).
It uses a different constraint evaluation approach introduced by [Simonetto et al](https://arxiv.org/abs/2112.01156).

**Data sets**

- [**IoT-23**](https://doi.org/10.5281/zenodo.4743746) - Malicious and benign IoT network traffic; 10,000 rows, 2 classes (sampled).
- [**UNSW-NB15**](https://doi.org/10.1109/MilCIS.2015.7348942) - Network intrusion dataset with 9 attacks; 10,000 rows, 2 classes (sampled). 
- [**URL**](https://doi.org/10.1016/j.engappai.2021.104347) - Legitimate and phishing URLs; 11,430 rows, 2 classes.
- [**LCLD**](https://www.kaggle.com/datasets/wordsforthewise/lending-club) - Kaggle's All Lending Club loan data; 20,000 rows, 2 classes (sampled).

<details>
<summary>Notes on preprocessing and sampling</summary>
<ul>
<li>The input data must be numeric and parse to a numeric type (<code>NULL</code> should be a null, not <code>-</code>, etc.).</li>
<li>Categorical attributes must be one-hot encoded.</li>
<li>Data sets should not be normalized, because this will be done automatically (otherwise the constraints must include manual scaling).</li>
<li>All data sets have 50/50 class label distribution.</li>
<li>The provided sampled data sets were generated by random sampling, without replacement, using Weka's supervised instance <a href="https://waikato.github.io/weka-blog/posts/2019-01-30-sampling/" target="_blank">SpreadSubsample</a>.</li>
</ul>
</details>

## Experiment workflow

A single experiment consists of training a classification model on a choice data set and applying an adversarial attack to that model. A constraint-validation approach can be enabled or disabled during the attack to impact the validity of the generated adversarial examples.

<pre>
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐ 
○───┤  args-parser  ├─────┤     setup     ├─────┤      run      ├─────┤      end      ├───◎
    └───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
     inputs:              * preprocess data      k times:                write result
     * data set           * init classifier      1. train model     
     * constraints        * init attack          2. attack
     * other configs      * init validation      3. score
</pre>

## Usage

**Software requirements**

* [Python](https://www.python.org/downloads/) -- version 3.9 or higher
* [GNU make](https://www.gnu.org/software/make/manual/make.html) -- version 3.81 or later

Check your environment using the following command, and install/upgrade as necessary.

```
python3 --version & make --version
```

**Install dependencies**

```
pip install -r requirements.txt --user
```

### Reproducing paper experiments

![duration](https://img.shields.io/badge/%F0%9F%95%92%2024%E2%80%9448%20hours/each-FFFF00?style=flat-square) **Run attack evaluations.**   
Run experiments for combinations of data sets x classifiers x attacks (24 experiment cases). 

<pre>
make attacks   -- run all attacks, using constraint enforcement.
make original  -- run all attacks, but without validation (ignore constraints).
make reset     -- run all attacks, but use naive reset strategy.
</pre>

![duration](https://img.shields.io/badge/%F0%9F%95%92%2030%20min%20%E2%80%94%203%20h-FFFF00?style=flat-square) **Run constraint performance test.**   
Uses varying number of constraints to measure performance impact on current hardware. 
This experiment tests PGD, CPGD, VPGD on same data set, UNSW-NB15.

```
make perf
```

### Visualizations

**Plots.** Generate plots of experiment results.

```
make plots
```

**Comparison plot.** To plot results from some other directory, e.g. `ref_result` append directory name.

```
make plots DIR=ref_result
```

## Extended/Custom usage

The default experiment options are defined statically in `config` files.
An experiment run can be customized further with command line arguments, to override the static options.
To run such custom experiments, call the `exp` module directly.

```
python3 -m exp [PATH] {ARGS}
```

For a list of supported arguments, run:

```
python3 -m exp --help
```

All plotting utilities live separately from experiments in `plot` module.
For plotting help, run:

```
python3 -m plot --help
```

### Repository organization

| Directory    | Description                               |
|:-------------|:------------------------------------------|
| `.github`    | Automated workflows, dev instructions     |
| `cge`        | CGE algorithm implementation              |
| `comparison` | C-PGD attack implementation               |
| `config`     | Experiment configuration files            |
| `data`       | Preprocessed input data sets              |
| `exp`        | Source code for running experiments       |
| `plot`       | Utilities for plotting experiment results |
| `ref_result` | Referential result for comparison         |
| `test`       | Unit tests                                |

- The Makefile contains pre-configured commands to ease running experiments.
- All software dependencies are listed in `requirements.txt`.
