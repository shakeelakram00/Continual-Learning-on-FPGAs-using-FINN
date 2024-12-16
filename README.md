# Continual Learning on FPGAs using FINN
FINN Extension to Enable Training on FPGA Hardware Accelerators Using Hardware-Software Co-Design
Using MPQ-DNNs FPGA-Accels are also built for efficient diagnosis of Cardiac Diseases (CD).

The Framework consists of three main self-explanatory files, 1. FPGA-Accel-Build Process, 2. CL on FPGA-Accels, 3. FPGA-Accel Inference. 
1. [Build Files for Each Accel:](CL_Files/Accel5_BuildFile.ipynb) This file includes the model structure, training and then FINN transformation to generate the bitstreams. The pre-build bitstreams are in [bitfiles_zcu102 directory](bitfiles_zcu102).
2. [Continual Learning for Each Accel:](CL_Files/Accel5_CL_5Rounds.ipynb) This file includes the whole process of continual learning, from dataset preparation, number of training rounds, to weight interpreter for runtime configuration of FPGA-Accel weights. This also saves the generated results and plots in the recurring repositories.
3. [Simulatnious Inference File:](CL_Files/runBitFileforInference.ipynb) This file runs the inference simultaneously, during continual training. This generates the inference results as well as the accelerator performance achieved, such as throughput, runtime, etc. 


























### Citation

The current implementation of the framework is based on the following publication. If you find it useful, please consider citing it.

Akram, Muhammad Shakeel, Bogaraju Sharatchandra Varma, and Dewar Finlay. "Continual Learning on FPGAs for Efficient Cardiac Diagnosis through Mix-Precision Quantized DNNs." Authorea Preprints (2024).

```bibtex
@article{akram2024continual,
  title={Continual Learning on FPGAs for Efficient Cardiac Diagnosis through Mix-Precision Quantized DNNs},
  author={Akram, Muhammad Shakeel and Varma, Bogaraju Sharatchandra and Finlay, Dewar},
  journal={Authorea Preprints},
  year={2024},
  publisher={Authorea}
}

