# Scatter Component Analysis (SCA)

MATLAB implementation of Scatter Component Analysis for domain generalization proposed in paper

Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). [Scatter component analysis: A unified framework for domain adaptation and domain generalization](https://ieeexplore.ieee.org/document/7542175/#full-text-section). IEEE transactions on pattern analysis and machine intelligence, 39(7), 1414-1430.

## Getting Started

### Prerequisites

The code is tested using MATLAB R2017b on Windows 10. Any later version should work normally.

## Running the tests

In MATLAB, change your *current folder* to "SCA" and run one of the file **demo.m** to see whether it could run normally.

The file **demo.m** does the following:

1. Load synthetic data from "./syn_data/data.m";

2. Prepare source sample sets (put sample sets 1, 2 in a MATLAB *cell array*), validation set (sample sets 3, 4 in a matrix), and test set (sample set 5 in a matrix);

3. Learn transformations using SCA on the source sample sets and validate hyperparameters on the validation set.

4. Apply the optimal transformation on the test set.

## Apply on your data

### Usage

Change your current folder to "SCA" and use the following commands

```matlab
[test_accuracy, predicted_labels, Zs, Zt] = SCA(X_s_cell, Y_s_cell, X_t, Y_t, params)
```

### Description

#### Function **SCA()**

| Input  | Description  |
|---|---|
|  X_s_cell           | cell of (n_s\*d) matrix, each matrix corresponds to the instance features of a source domain |
|  Y_s_cell           | cell of (n_s\*1) matrix, each matrix corresponds to the instance labels of a source domain |
|  X_t           | (n_t\*d) matrix, rows correspond to instances and columns correspond to features |
|  Y_t           | (n_t\*1) matrix, each row is the class label of corresponding instances in X_t |
|  params           | optional parameters, details can be found in SCA.m |

| Output  | Description  |
|---|---|
| test_accuracy | test accuracy on target instances |
| predicted_labels | predicted labels of target instances |
| Zs         | projected source domain instances |
| Zt         | projected target domain instances |


## Authors

* **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com

See also the list of [contributors](https://github.com/amber0309/SCA/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Hat tip to Ya Li for his [Conditional Invariant Domain Generalization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16595) (CIDG) code.
