#!/usr/bin/env python
# coding: utf-8

# In[11]:


if __name__ == "__main__":
    import sys
    import numpy as np

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    
    if len(sys.argv) != 4:
        print("Usage:python IBN.py <training-dataset> <validation-dataset> <testing-dataset>")
    else:
        dataset = np.vstack((np.loadtxt(sys.argv[1],delimiter=',', dtype=bool), np.loadtxt(sys.argv[2],delimiter=',', dtype=bool)))
        pos_prob = (np.sum(dataset, axis=0)+1)/(dataset.shape[0]+2)
        neg_prob = 1-pos_prob
        testset=np.loadtxt(sys.argv[3],delimiter=',', dtype=bool)
        pos_test = np.sum(testset, axis=0)
        neg_test = testset.shape[0]-pos_test
        LL = np.dot(pos_test, np.log(pos_prob))+np.dot(neg_test, np.log(neg_prob))
        print("Avg. Log Likelihood:", LL/testset.shape[0])

