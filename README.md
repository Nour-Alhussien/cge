# Constrained adversarial attacks

Experimental setup for introducing constraints to universal adversarial attacks.

## Repository organization

| Directory                     | Description                                |
|:------------------------------|:-------------------------------------------|
| `.github`                     | Automated workflows                        |
| [`comparison`][c_a]           | CPGD implementation + license              |
| `config`                      | Experiment configuration files             |
| [`data`](#data-sets)          | Experiment data sets                       |
| [`exp`](#experiment-workflow) | Experiment implementation                  |
| `reference`                   | Original unmodified attacks, for reference |
| `test`                        | Implementation unit tests                  |

The Makefile contains pre-configured commands to ease running experiments,
particularly `make all` will run all experiments at once.

The software dependencies are specified in `requirements.txt` and `requirements-dev.txt`.

## Experiment workflow

<pre>
      __main__.py                          experiment.py
     ┌───────────┐      ┌────────────┐      ┌────────────┐      ┌────────────┐
○────┤   parse   ├──────┤   setup    ├──────┤    run     ├──────┤    end     ├────◎
     │   args    │      │ experiment │      │ experiment │      │ experiment │
     └───────────┘      └────────────┘      └────────────┘      └────────────┘
      from config        preprocess &        k times:           * write result
      & cmd args         init model          1. train model     * plot graph
                         attack, and         2. attack
                         validation          3. score
</pre>

## Usage

Install dependencies

```
pip install -r requirements.txt
```

Run all experiments for all configuration options

```
make all
```

Usage help

```
python3 -m exp --help
```

**Development instructions**

First install all dev dependencies:

```
pip install -r requirements-dev.txt
```

Available code quality checks

<pre>
make test    -- Run unit tests
make lint    -- Run linter
make dev     -- Test and lint, all at once
</pre>


## Data sets

All data sets are preprocessed and located in `data/` directory.

| Name                                                                                                                                              |  Rows  | Inclusion |
|:--------------------------------------------------------------------------------------------------------------------------------------------------|:------:|:---------:|
| [IoT-23][iot]<br/>Malicious and benign IoT network traffic<br/>DOI: [10.5281/zenodo.4743746](https://doi.org/10.5281/zenodo.4743746)              | 10,000 |  Sampled  |
| [UNSW-NB15][uns]<br/>Network intrusion dataset with 9 attacks<br/>DOI: [10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942) | 10,000 |  Sampled  |
| [URL][url]<br/>Legitimate and phishing URLs<br/>DOI: [10.1016/j.engappai.2021.104347](https://doi.org/10.1016/j.engappai.2021.104347)             | 11,430 | Complete  |
| [LCLD][lcld]<br/>Kaggle's All Lending Club loan data<br/>URL: [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)              | 20,000 |  Sampled  | 


**Sampling.** The sampled data sets were generated to obtain equal class distribution using Weka's supervised instance
[`SpreadSubsample`](https://waikato.github.io/weka-blog/posts/2019-01-30-sampling/)).

[iot]: https://www.stratosphereips.org/datasets-iot23
[uns]: https://research.unsw.edu.au/projects/unsw-nb15-dataset
[url]: https://data.mendeley.com/datasets/c2gw7fy2j4/3
[lcld]: https://www.kaggle.com/datasets/wordsforthewise/lending-club
[c_a]: https://github.com/serval-uni-lu/constrained-attacks