# Introducing the Attribution Stability Indicator: a Measure for Time Series XAI Attributions

Code and result repository for "Introducing the Attribution Stability Indicator: a Measure for Time Series XAI Attributions".

## The Attribution Stability Indicator

The attribution stability indicator (ASI) can be found in the `asi/exp_stability_indicator.py` file as `attribution_stability_indicator`.  
The measure incorporates a perturbation analysis and, as a measure, the flips of classification but extends such an approach with the distance between the probability, the time series, and the attributions before and after the perturbation.
The probability change is measured by the Jensen-Shannon distance. The time series and attribution changes are measured by the Pearson correlation coefficient.
Further details can be found in the paper.

## Results

The results can be seen in the notebooks.  
However, rerunning the notebooks with the trained models from the paper leads to the same results.

## Reproducibility

For reproducibility, please install Python as mentioned in the version below and the requirements.txt.  
The models can be downloaded from https://zenodo.org/record/8328543

## Extensions

The juypter notebooks can be used as a guideline for future extensions of the experiments.  
The dataset needs to be exchanged and the models, but all the different analyses should work for other datasets.

## Libraries

-   Python v3.10
-   Pytorch (https://pytorch.org/)
-   Captum (https://captum.ai/)
-   Numpy (https://numpy.org/)
-   Scipy (https://scipy.org/)
-   Pandas (https://pandas.pydata.org/)
-   SKTime (https://www.sktime.net/)
-   Sci-Kit Learn (https://scikit-learn.org/)

## License

Released under MIT License. See the LICENSE file for details.

## Reference

```
@conference{,
 author = {Schlegel, Udo and Keim, Daniel A.},
 booktitle = {ECML-PKDD Workshop XAI-TS: Explainable AI for Time Series: Advances and Applications},
 title = {Introducing the Attribution Stability Indicator: a Measure for Time Series XAI Attributions},
 year = {2023}
}
```

