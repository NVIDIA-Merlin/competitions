## GPU-accelerated Inference

### Introduction

<p align="center">
  <img src="../imgs/speedup.png" width="50%" >
</p>

This year, the hosts introduced latency constraints to provide a more realistic production environment. Around 15M tweet-user pairs had to be predicted in less than 24 hours using a single core CPU with 64GB memory (average of 6ms per example). After the competition ended, we wanted to analyze the benefits of GPUs in that environment. Using open source libraries, such as [Forrest Inference Library](https://github.com/rapidsai/cuml), [NVTabular](https://github.com/NVIDIA/NVTabular/), [RAPIDS cuDF](https://github.com/rapidsai/cudf/), [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/), we accelerated our prediction pipeline end-to-end with a single NVIDIA A100 GPU with 40GB memory.<br><br>

The experiments reduced the prediction time from **23 hours and 40 minutes** on a single core CPU down to **5 minutes and 30 seconds. That is a speed-up of ~260x!!** For a fair comparison, we used the cheapest Google Cloud Platform instance with 64GB memory and enabled all CPU cores. The instance of type e2-highmem-8 with 8 CPU cores and 64GB memory requires **8 hours and 24 minutes**. Our GPU-accelerated solution is still **92x faster**. The left barchart shows a breakdown per step. Running an e2-highmem-8 instance for 8 hours and 24 minutes costs ~$3 versus running an a2-highgpu-1g for 5 minutes and 30 seconds costs $0.34 (see right barchart). **Using GPU accelerates inference pipelines by 92x and simultaneously, reduces the costs by ~8x-9x**.<br><br>
You can find more details in [our paper](../GPU-Accelerated-Boosted-Trees-and-Deep-Neural-Networks-for-Better-Recommender-Systems.pdf).

### Requirements

Our code requires the latest version of RAPIDs cuDF, NVTabular==0.5.3, TensorFlow and PyTorch. 

```
conda create -n recsys2021 -c rapidsai-nightly -c nvidia -c conda-forge rapids=21.08 python=3.7 cudatoolkit=11.2
pip install jupyter matplotlib seaborn pandas scikit-learn
pip install torch
pip install tensorflow-gpu==2.4
pip install nvtabular
pip uninstall dask
pip uninstall distributed
pip install dask[distributed]==2021.4.0
pip install xgboost
pip install transfomers
```

### Execution

The `run_GPU` script executes the inference models step-by-step. The repository does not contain the model files or preprocessed features.
