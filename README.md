# CSharpGPUExperiments
 
HOW TO BUILD AND RUN:
This is a Visual Studio solution with two projects in it: CSharpCPUExperiments and CudaKernelExperiment.
You must build CudaKernelExperiment first, which creates a library that the other project will consume. Then you can build and run CSharpCPUExperiments. Right now (Feb 20, 2020), all it does it compare the speed of a vector dot-product and vector-sum in both a C# loop an using a CUDA kernel.
