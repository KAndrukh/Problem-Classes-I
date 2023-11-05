## You should run everything in a following order
1. *sets_creation.py* - this will generate a synthetic dataset of size k, which you can change on 119 line of code.
2. *metrics_calculations.py* - this will calculate fairness measures for synthetic data, here you can also change it on 28 line of code.
3. *metrics_comparison.py* - where all magic happens, compares output of different meassures and creates plots for them and their correlations. You can change the correlation method on 140 line of code. The sample size can be changed on 14 line of code.
