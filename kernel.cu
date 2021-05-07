#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math_functions.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

cudaError_t addWithCuda(double *positions, double *velocities, double *personalBests,
	double *globalBest);

#define M_PI 3.14159265358979323846

const int particlesNum = 1024;
const int dimensionsNum = 2;
const int iterationsNum = 2000;


double beale_function_h(double *x);
double easome_function_h(double *x);
double rosenbrock_function_h(double *x);


__device__ double beale_function(double *x);
__device__ double easome_function(double *x);
__device__ double rosenbrock_function(double *x);


__global__ void psoKernel(double *positions, double *velocities,
	double *personalBests, double *globalBest, double r1,
	double r2)
{

	const double c1 = 2;
	const double c2 = 2;
	const double w = 0.5;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = i % dimensionsNum;

	if (i >= particlesNum * dimensionsNum)
		return;

	//vel= inertia component + cognitive component + social component
	velocities[i] = w * velocities[i] + c1 * r1 * (personalBests[i] - positions[i])
		+ c2 * r2 * (globalBest[tid] - positions[i]);

	//updated position
	positions[i] += velocities[i];


	//finding personal best
	double temp1[dimensionsNum];
	double temp2[dimensionsNum];

	if (tid != 0)
		return;

	for (int j = 0; j < dimensionsNum; j++)
    {
        temp1[j] = positions[i + j];
    }

    for (int j = 0; j < dimensionsNum; j++)
    {
        temp2[j] = personalBests[i + j];
    }
	
	if (easome_function(temp2) > easome_function(temp1))
	{
		for (int j = 0; j < dimensionsNum; j++)
			personalBests[i + j] = positions[i + j];
	}
}

void globalBestFunction(double *personalBests, double *globalBest) {

	double temp[dimensionsNum];
	for (int i = 0; i < particlesNum * dimensionsNum; i += dimensionsNum)
	{
		for (int k = 0; k < dimensionsNum; k++)
			temp[k] = personalBests[i + k];

		if (easome_function_h(globalBest) > easome_function_h(temp))
		{
			for (int k = 0; k < dimensionsNum; k++)
				globalBest[k] = temp[k];
		}
	}
}


int main()
{

	double positions[particlesNum * dimensionsNum];
	double velocities[particlesNum * dimensionsNum];
	double personalBests[particlesNum * dimensionsNum];
	double globalBest[dimensionsNum];


	for (int i = 0; i < particlesNum * dimensionsNum; i++)
	{
		positions[i] = (-4.5f) + double(((4.5f - (-4.5f)) + 1) * rand() / (RAND_MAX + 1.0));
		personalBests[i] = positions[i];
		velocities[i] = 0;
	}

	for (int i = 0; i < dimensionsNum; i++)
		globalBest[i] = personalBests[i];

	clock_t start = clock();

	cudaError_t cudaStatus = addWithCuda(positions, velocities, personalBests, globalBest);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}


	for (int i = 0; i < dimensionsNum; i++)
		printf("x%d = %f\n", i, globalBest[i]);

	printf("Minimum = %f", easome_function_h(globalBest));
	printf("\n");

	clock_t stop = clock();

	printf("Computing time: %f ms\n",
		(double)(stop - start) / CLOCKS_PER_SEC);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


double beale_function_h(double *x)
{
	double res = 0;

	//res += pow((1.5-x[0]+x[0]*x[0]), 2) + pow((2.25-x[0]+x[0]*x[1]*x[1]), 2) + pow((2.625-x[0]+x[0]*x[1]*x[1]*x[1]), 2);


	for (int i = 0; i < dimensionsNum - 1; i++)
	{
		double y = x[i];
		double y1 = x[i + 1];

		res += pow((1.5 - y + y * y1), 2) + pow((2.25 - y + y * y1*y1), 2) + pow((2.625 - y + y * y1*y1*y1), 2);
	}

	return res;
}

double easome_function_h(double *x)
{
	double res = 0;

	//res = -cos(x[0])*cos(x[1])*exp((-pow( x[0]-M_PI, 2)) - pow(x[1]-M_PI, 2));


	for (int i = 0; i < dimensionsNum - 1; i++)
	{
		double y = x[i];
		double y1 = x[i + 1];

		res += -cos(y)*cos(y1)*exp((-pow((y - M_PI), 2)) - pow((y1 - M_PI), 2));
	}

	return res;
}

double rosenbrock_function_h(double *x)
{

	double res = 0.0;
	double sum = 0.0;

	//res = -cos(x[0])*cos(x[1])*exp((-pow( x[0]-M_PI, 2)) - pow(x[1]-M_PI, 2));


	for (int i = 0; i < dimensionsNum - 1; i++)
	{
		double y = x[i];
		double y1 = x[i + 1];

		res += 100 * pow((y1 - (y*y)), 2) + pow((y - 1), 2);
	}

	return res;
}


__device__ double beale_function(double *x)
{

	double res = 0;
	double yn = x[dimensionsNum - 1];

	// res += pow((1.5-x[0]+x[0]*x[0]), 2) + pow((2.25-x[0]+x[0]*x[1]*x[1]), 2) + pow((2.625-x[0]+x[0]*x[1]*x[1]*x[1]), 2);

	for (int i = 0; i < dimensionsNum - 1; i++)
	{
		double y = x[i];
		double y1 = x[i + 1];

		res += pow((1.5 - y + y * y1), 2) + pow((2.25 - y + y * y1*y1), 2) + pow((2.625 - y + y * y1*y1*y1), 2);
	}

	return res;
}

__device__ double easome_function(double *x)
{

	double res = 0;

	//res = -cos(x[0])*cos(x[1])*exp((-pow( x[0]-M_PI, 2)) - pow(x[1]-M_PI, 2));


	for (int i = 0; i < dimensionsNum - 1; i++)
	{
		double y = x[i];
		double y1 = x[i + 1];

		res += -cos(y)*cos(y1)*exp((-pow((y - M_PI), 2)) - pow((y1 - M_PI), 2));
	}

	return res;
}


__device__ double rosenbrock_function(double *x)
{

	double res = 0.0;

	//res = -cos(x[0])*cos(x[1])*exp((-pow( x[0]-M_PI, 2)) - pow(x[1]-M_PI, 2));


	for (int i = 0; i < dimensionsNum - 1; i++)
	{
		double y = x[i];
		double y1 = x[i + 1];

		res += 100 * pow((y1 - (y*y)), 2) + pow((y - 1), 2);
	}

	return res;
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(double *positions, double *velocities, double *personalBests,
	double *globalBest)
{
	int SIZE = particlesNum * dimensionsNum;

	double *dev_positions;
	double *dev_velocity;
	double *dev_particleBest;
	double *dev_gloabalBest;

	//double temp[dimensionsNum];

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_positions, SIZE * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_velocity, SIZE * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_particleBest, SIZE * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_gloabalBest, sizeof(double) * dimensionsNum);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}



	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_positions, positions, SIZE * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_velocity, velocities, SIZE * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_particleBest, personalBests, SIZE * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_gloabalBest, globalBest, sizeof(double) * dimensionsNum, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.


	int threadsNum = 64;
	int blocksNum = particlesNum / threadsNum;

	for (int i = 0; i < iterationsNum; i++)
	{
		psoKernel << <blocksNum, threadsNum >> > (dev_positions, dev_velocity,
			dev_particleBest, dev_gloabalBest,
			(double)rand() / (double)RAND_MAX,
			(double)rand() / (double)RAND_MAX);



		cudaStatus = cudaMemcpy(personalBests, dev_particleBest, sizeof(double) * particlesNum * dimensionsNum, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		globalBestFunction(personalBests, globalBest);

		cudaStatus = cudaMemcpy(dev_gloabalBest, globalBest, sizeof(double) * dimensionsNum, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

	cudaStatus = cudaMemcpy(positions, dev_positions, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(velocities, dev_velocity, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(personalBests, dev_particleBest, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(globalBest, dev_gloabalBest, sizeof(double) * dimensionsNum, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}


Error:
	cudaFree(dev_positions);
	cudaFree(dev_velocity);
	cudaFree(dev_particleBest);
	cudaFree(dev_gloabalBest);

	return cudaStatus;
}