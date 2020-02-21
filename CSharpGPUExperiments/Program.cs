using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Collections.Generic;
using System;
using System.Linq;

namespace CSharpGPUExperiments
{
    class Program
    {
        const int VECTOR_SIZE = 1048576*512;
        const int THREADS_PER_BLOCK = 1024;
        const int BLOCKS_PER_GRID = 1024;
        readonly int GRIDS = VECTOR_SIZE / THREADS_PER_BLOCK / BLOCKS_PER_GRID + 1;
        const string PTX_NAME = @"C:\Users\mikdc\Documents\Programming\CSharpGPUExperiments\CSharpCudaExperiment\x64\Debug\kernel.ptx";

        static CudaKernel fillVectorWithCuda;
        static CudaKernel multiplyVectorWithCuda;
        static CudaKernel increaseVectorWithCuda;
        static CudaKernel dotProduct;
        static CudaKernel sumVectorWithCuda;

        static CudaKernel BuildKernelFromFunction(string functionName, ref CudaContext context)
        {
            //CudaContext newContext = new CudaContext();
            CudaKernel kernel = context.LoadKernelPTX(PTX_NAME, functionName);
            kernel.BlockDimensions = THREADS_PER_BLOCK;
            kernel.GridDimensions = BLOCKS_PER_GRID;
            return kernel;
        }

        static void InitKernels()
        {
            CudaContext cntxt = new CudaContext();
            fillVectorWithCuda = BuildKernelFromFunction("SetKernel", ref cntxt);
            multiplyVectorWithCuda = BuildKernelFromFunction("FactorKernel", ref cntxt);
            increaseVectorWithCuda = BuildKernelFromFunction("AddKernel", ref cntxt);
            dotProduct = BuildKernelFromFunction("VectorDotProduct", ref cntxt);
            sumVectorWithCuda = BuildKernelFromFunction("VectorSum", ref cntxt);
        }

        static Func<int[], int, CudaKernel, int[]> runKernel = (m, value, func) =>
        {
            // init parameters
            CudaDeviceVariable<int> vector_host = m;
            // run cuda method
            func.Run(vector_host.DevicePointer, value);
            // copy return to host
            int[] output = new int[m.Length];
            vector_host.CopyToHost(output);
            return output;
        };

        static void FirstTrial(string[] args)
        {
            InitKernels();
            List<CudaKernel> kernels = new List<CudaKernel> { fillVectorWithCuda, multiplyVectorWithCuda, increaseVectorWithCuda };
            foreach(CudaKernel k in kernels)
            {
                int[] vector = Enumerable.Range(1, VECTOR_SIZE).ToArray();
                Console.WriteLine();
                Console.WriteLine("Use {0}:", k.KernelName);
                Console.ReadKey();
                vector = runKernel(vector, 13, k);
                for (int i = 0; i < 10; i++){ Console.Write("{0,8}", vector[i]); }
                Console.WriteLine();
                Console.WriteLine("...");
                for (int i = VECTOR_SIZE - 10; i < VECTOR_SIZE; i++) { Console.Write("{0,8}", vector[i]); }
                Console.WriteLine();
                Console.WriteLine("Done");
                Console.ReadKey();
            }
        }

        static void Main(string[] args)
        {
            InitKernels();
            float[] v1 = new float[VECTOR_SIZE];
            float[] v2 = new float[VECTOR_SIZE];
            float[] v3 = new float[VECTOR_SIZE / THREADS_PER_BLOCK + 1];
            v1[0] += 1;
            v2[0] += 1;
            v1[1] += 1;
            v2[2] += 1;
            v1[4] += 3;
            v2[4] += 5;
            v1[100] += 100;
            v2[100] += 100;
            v1[1058] += 20;
            v2[1058] += 100;
            v1[500000] += 1000;
            v2[500000] += 3000;
            CudaDeviceVariable<float> host_v1 = v1;
            CudaDeviceVariable<float> host_v2 = v2;
            CudaDeviceVariable<float> host_v3 = v3;

            float[] sum_v = new float[VECTOR_SIZE / THREADS_PER_BLOCK + 1];
            CudaDeviceVariable<float> host_sum = sum_v;



            foreach (int j in new int[1]){
                Console.WriteLine();

                System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();
                float sum1;

                Console.WriteLine("Taking the dot product of two vectors of size {0} with ...", VECTOR_SIZE);
                Console.Write("... CUDA: ");
                stopwatch.Start();
                // run cuda method
                dotProduct.Run(VECTOR_SIZE, host_v1.DevicePointer, host_v2.DevicePointer, host_v3.DevicePointer);
                // copy return to host
                host_v3.CopyToHost(v3);
                //Console.WriteLine(string.Join(", ", v3));
                sum1 = v3.Sum();
                stopwatch.Stop();
                Console.WriteLine("v1 . v2 = {0}  ({1}ms)", sum1, stopwatch.ElapsedMilliseconds);

                stopwatch.Reset();
                Console.Write("... C#:   ");
                stopwatch.Start();
                sum1 = 0.0F;
                for (int i = 0; i < VECTOR_SIZE; i++)
                {
                    sum1 += v1[i] * v2[i];
                }
                stopwatch.Stop();
                Console.WriteLine("v1 . v2 = {0}  ({1}ms)", sum1, stopwatch.ElapsedMilliseconds);

                Console.WriteLine();
                Console.WriteLine();

                Console.WriteLine("Summing the elements of a vector of size {0} with ...", VECTOR_SIZE);
                stopwatch.Reset();
                Console.Write("... CUDA: ");
                stopwatch.Start();
                sumVectorWithCuda.Run(VECTOR_SIZE, host_v1.DevicePointer, host_sum.DevicePointer);
                host_sum.CopyToHost(sum_v);
                sum1 = sum_v.Sum();
                stopwatch.Stop();
                Console.WriteLine("sum(v)  = {0}  ({1}ms)", sum1, stopwatch.ElapsedMilliseconds);

                stopwatch.Reset();
                Console.Write("... C#:   ");
                stopwatch.Start();
                sum1 = 0.0F;
                foreach (float x in v1)
                {
                    sum1 += x;
                }
                stopwatch.Stop();
                Console.WriteLine("sum(v)  = {0}  ({1}ms)", sum1, stopwatch.ElapsedMilliseconds);

                stopwatch.Reset();
                Console.Write("... LINQ: ");
                stopwatch.Start();
                sum1 = v1.Sum();
                stopwatch.Stop();
                Console.WriteLine("sum(v)  = {0}  ({1}ms)", sum1, stopwatch.ElapsedMilliseconds);

            }
        }
    }
}