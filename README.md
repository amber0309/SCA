# Scatter Component Analysis (SCA)

MATLAB implementation of Scatter Component Analysis proposed in paper

Ghifary, M., Balduzzi, D., Kleijn, W. B., & Zhang, M. (2017). [Scatter component analysis: A unified framework for domain adaptation and domain generalization](https://ieeexplore.ieee.org/document/7542175/#full-text-section). IEEE transactions on pattern analysis and machine intelligence, 39(7), 1414-1430.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The code is tested using MATLAB R2017b on Windows 10. Any later version should work normally.

## Running the tests

In MATLAB, change your *current folder* to "SCA" and run the file *demo.m* to see whether it could run normally.

The file *demo.m* does the following:

1. Load synthetic data "dataX.mat" and "dataY.mat" 

2. Put all 4 samples in a MATLAB *cell array*. (3 domains in total. The last two samples are from the same domain.)

3. Train SCA on the first 2 domains with 3rd domain being the validation set.

4. Test the trained transformation on the 4th domain.

## Apply on your data

### Usage

Change your current folder to "SCA" and use the following commands

```matlab
[B, A] = SCA(X, Y, beta, delta, epsilon, sigma)
[ACC, pre_labels, Zs, Zt] = SCA_test(B, A, X, Y, X_t, Y_t, sigma, eig_ratio)

### Description

Input of function **SCA()**

| Argument  | Description  |
|---|---|
|  X           | cell of L by d matrix, each matrix corresponds to the data of a domain |
|  Y           | cell of L by 1 matrix, each matrix corresponds to the label of a domain |
|  beta, delta | trade-off parameters in Eq.(20) in [1] |
|  epsilon     | a small constant for numerical stability |
|  sigma       | kernel width |

Output of function **SCA()**

| Argument  | Description  |
|---|---|
|  A           | eigenvalues |
|  B           | transformation matrix |


Input of function **SCA_test()**

| Argument  | Description  |
|---|---|
| A | eigenvalues |
| B | transformation matrix|
| X_all | train data in cell format, each element is a L by d matrix |
| X_s | train data in L by d matrix format |
| Y_s | train label in L by 1 matrix format |
| X_t | target domain data in L by d matrix |
| Y_t | target domain label in L by 1 matrix |
| sigma | kernel width |
| eig_ratio | eigvalue ratio used for test |

Output of function **SCA_test()**

| Argument  | Description  |
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

* Hat tip to Ya Li for his [Conditional Invariant Domain Generalization] (https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16595) (CIDG) code.
