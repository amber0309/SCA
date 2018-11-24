# Scatter Component Analysis (SCA)

MATLAB implementation of Scatter Component Analysis proposed in paper

Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). [Scatter component analysis: A unified framework for domain adaptation and domain generalization](https://ieeexplore.ieee.org/document/7542175/#full-text-section). IEEE transactions on pattern analysis and machine intelligence, 39(7), 1414-1430.

## Getting Started

### Prerequisites

The code is tested using MATLAB R2017b on Windows 10. Any later version should work normally.

## Running the tests

In MATLAB, change your *current folder* to "SCA" and run one of the files *demo_[]_[].m* to see whether it could run normally.

The file *demo_[]_[].m* does the following:

1. Load synthetic data in folder "syn_data" 

2. Put all 4 sample sets in a MATLAB *cell array*. (3 distinct domains. The last two sample sets are from the 3rd domain.)

3. Train SCA on the first 2 domains and validate hyperparameters on the 3rd domain.

4. Test the transformation with highest validation accuracy on the 4th sample set.

## Apply on your data

### Usage

Change your current folder to "SCA" and use the following commands

```matlab
[P, T, D, Q, K_bar] = SCA_quantities(K, X, Y)
[B, A] = SCA_transformation(P, T, D, Q, K_bar, beta, delta, epsilon)
[ACC, pre_labels, Zs, Zt] = SCA_test(B, A, K_s, K_t, Y_s, Y_t, eig_ratio)
```

### Description

#### Function **SCA_quantities()**

| Input  | Description  |
|---|---|
|  K | kernel matrix of data of all source domains |
|  X           | cell of L by d matrix, each matrix corresponds to the data of a domain |
|  Y           | cell of L by 1 matrix, each matrix corresponds to the label of a domain |

| Output  | Description  |
|---|---|
|  P           | between-class scatter |
|  T           | total scatter |
|  D           | domain scatter |
|  Q           | within-class scatter |
|  K_bar           | the centered kernel matrix |

#### Function **SCA_transformation()**

| Input  | Description  |
|---|---|
|  P           | between-class scatter |
|  T           | total scatter |
|  D           | domain scatter |
|  Q           | within-class scatter |
|  K_bar           | the centered kernel matrix |
|  \beta, \delta            | trade-off parameters |
|  \epsilon            | coefficient of the identity matrix |

| Output  | Description  |
|---|---|
|  B           | matrix of projection |
|  A           | corresponding eigenvalues |

#### Function **SCA_test()**

| Input  | Description  |
|---|---|
| B | transformation matrix |
| A | corresponding eigenvalues |
| K_s | kernel matrix of training data |
| K_t | kernel matrix of target data |
| Y_s | training label in L by 1 matrix |
| Y_t | target label in L by 1 matrix |
| eig_ratio | eigvalue ratio used for test |

| Output  | Description  |
|---|---|
| ACC | test accuracy on target domain |
| pre_labels | predicted labels of target domain data |
| Zs         | projected source domain data |
| Zt         | projected target domain data |

## Authors

* **Shoubo Hu** - shoubo DOT sub AT gmail DOT com

See also the list of [contributors](https://github.com/amber0309/SCA/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Hat tip to Ya Li for his [Conditional Invariant Domain Generalization](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16595) (CIDG) code.
