# HF2-Meta
2021 NeurIPS: HF2-Meta: A Resource-Efficient Framework for Personalized Federated Learning

# Introduction

The ubiquity of smart mobile devices has contributed to an unprecedented amount of raw user data that remain unprocessed due to privacy concerns. Recent methods in Federated Learning (FL) attempt to utilize such data while preserving privacy by generating a common global model using local models trained in a decentralized manner. However, this scheme not only generates a non-personalized output, but also offloads the bulk of the computational overhead to resource-constrained edge devices. Since user data are heterogeneous (non-i.i.d.) by nature, we advocate an alternative approach for FL that learns a *shared adaptable* model that can quickly be personalized with minimal user information. In doing so, we investigate how to effectively utilize *proxy data* and computational resources in the *server* for better accuracy, an area relatively unexplored in the FL regime. Thus, we present HF$^2$(Hessian-Free Federated) Meta Learning and conduct empirical evaluations on various benchmarks. Our results demonstrate that HF$^2$-Meta is up to 2.57% more accurate, converges up to 34.2% faster, and incurs less computation overhead in clients compared to the latest methods in personalized federated learning, while also comparing favorably to conventional (centralized) methods in meta-learning.

# Usage

``` bash
python main.py -m {Specify which method to use} -ds {Specify which dataset to use} -sd {Specify the amount of proxy data} -r {Specify the number of times the training process will be run} -cr {Specify the total number of training rounds}
```

