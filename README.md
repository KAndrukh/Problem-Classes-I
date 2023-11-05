## You should run everything a following order
1. *sets_creation.py* - this will generate a synthetic dataset of size k, which you can change it on 119 line of code in the file.
2. *metrics_calculations.py* - this will calculate fairness measures for synthetic data, here you can also change it on 28 line of code.
3. *metrics_comparison.py* - where all magic happens, compares output of different meassures and creates plots for them and their correlations. The sample size can be changed on 14 line of code.