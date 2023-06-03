// Last update: 16/12/2020
#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}



__device__ int bCount = 0;
volatile __device__ int bCount1 = 0;

__global__ void scanKernel(const uint32_t *in, int n, volatile uint32_t* blkSums, uint32_t* nOnesBefore, uint32_t bitIdx){
    
    __shared__ int bi;
    if(threadIdx.x==0){
        bi = atomicAdd(&bCount, 1);
    }
    __syncthreads();
	extern __shared__ int s_data[];
    
    // 1. Extract bits
    int i1 = bi * 2 * blockDim.x + threadIdx.x;
	int i2 = i1 + blockDim.x;
	if (i1 < n){
        s_data[threadIdx.x] = (in[i1] >> bitIdx) & 1;
    }
	if (i2 < n){
        s_data[threadIdx.x + blockDim.x] = (in[i2] >> bitIdx) & 1;
    }
    __syncthreads();

    // compute nOnesBefore
	// 2. Each block does scan with data on SMEM
	// 2.1. Reduction phase
	for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
	{
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // To avoid warp divergence
		if (s_dataIdx < 2 * blockDim.x)
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
		__syncthreads();
	}

    if(threadIdx.x==0){
        int index = 2 * blockDim.x - 1;
        if(blkSums != NULL){
            blkSums[bi] = s_data[index];
        }
        s_data[index] = 0;
    }
    __syncthreads();
 
	// 2.2. Post-reduction phase
	for (int stride = blockDim.x; stride > 0; stride /= 2)
	{
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // Wow
		if (s_dataIdx < 2 * blockDim.x){
            int neededValue = s_data[s_dataIdx];
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
            s_data[s_dataIdx - stride] = neededValue;
        }
		__syncthreads();
	}
    if(threadIdx.x == 0){
        if(bi > 0){
            while(bCount1 < bi){}
            blkSums[bi] += blkSums[bi-1];
            __threadfence();
        }
        bCount1+=1;
    }
    __syncthreads();

    if(bi > 0){
        s_data[threadIdx.x] += blkSums[bi - 1];
        s_data[threadIdx.x + blockDim.x] += blkSums[bi - 1];
    }
    __syncthreads();
    if(i1 < n){
        nOnesBefore[i1] = s_data[threadIdx.x];
    }
    if(i2 < n) {
        nOnesBefore[i2] = s_data[threadIdx.x + blockDim.x];
    }
}

__global__ void rankKernel(int bitIdx, int n, uint32_t nZeros, uint32_t* nOnesBefore, uint32_t* in, uint32_t* out){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < n){
        uint32_t inVal = in[i];
        uint32_t nOnesBeforeVal = nOnesBefore[i];
        int rank;
        if (((inVal >> bitIdx) & 1) == 0)
            rank = i - nOnesBeforeVal;
        else
            rank = nZeros + nOnesBeforeVal;
        out[rank] = inVal;
    }
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO
    int blkDataSize = 2 * blockSize;
    int zero = 0;
    uint32_t nOnesBeforeNeeded;
    uint32_t numNeeded;
    uint32_t * d_in, * d_blkSums, *d_nOnesBefore, *d_out;
    size_t nBytes = n * sizeof(uint32_t);
    size_t smemScan = blkDataSize * sizeof(uint32_t);
    dim3 gridSizeScan((n - 1) / blkDataSize + 1);
    dim3 gridSizeRank((n - 1) / blockSize + 1);

    CHECK(cudaMalloc(&d_in, nBytes)); 
    CHECK(cudaMalloc(&d_out, nBytes)); 
    CHECK(cudaMalloc(&d_nOnesBefore, nBytes)); 
    CHECK(cudaMalloc(&d_blkSums, gridSizeScan.x * sizeof(uint32_t)));

    CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));


    for(int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++){
        scanKernel<<<gridSizeScan, blockSize, smemScan>>>(d_in, n, d_blkSums, d_nOnesBefore, bitIdx);
        CHECK(cudaMemcpy(&nOnesBeforeNeeded, &d_nOnesBefore[n-1], sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&numNeeded, &d_in[n-1], sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint32_t nZeros = n - nOnesBeforeNeeded - ((numNeeded >> bitIdx) & 1);

        rankKernel<<<gridSizeRank, blockSize>>>(bitIdx,n,nZeros, d_nOnesBefore, d_in, d_out);
        CHECK(cudaMemcpy(d_in, d_out,nBytes, cudaMemcpyDeviceToDevice)); 
        CHECK(cudaMemcpyToSymbol(bCount, &zero, sizeof(zero)));
        CHECK(cudaMemcpyToSymbol(bCount1, &zero, sizeof(zero)));
    }
    CHECK(cudaMemcpy(out, d_in, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_nOnesBefore));
    CHECK(cudaFree(d_blkSums));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            // printf("%i, %i\n", out[i], correctOut[i]);
            printf("INCORRECT :(\n");
            return;
        
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        //in[i] = rand() % 255; // For test by eye
        in[i] = rand();
        // in[i] = i + 1;
    }
    // printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    // printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    // printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
