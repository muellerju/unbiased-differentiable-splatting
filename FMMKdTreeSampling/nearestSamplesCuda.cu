#include <torch/extension.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include "linalg.cuh"
#include "logging.h"

template<typename scalar_t, typename indice_t>
__device__ void insert(
	indice_t* nearest,
	indice_t surfaceIdx,
	indice_t candidateIdx,
	scalar_t const* points,
	unsigned int numNearest
)
{
	unsigned int surfaceOffset = 3 * surfaceIdx; // Assumes points.shape[0] == 1
	ConstVector3<scalar_t> surfacePoint(&points[surfaceOffset]);

	// Check if another pixel already contributed the current candidate
	for (unsigned int k = 0; k < numNearest; k++)
	{
		unsigned int offset = k + numNearest * surfaceIdx;// Assumes points.shape[0] == 1
		indice_t nearestIdx = nearest[offset];
		if (candidateIdx == nearestIdx)
			return;
	}

	for (unsigned int k = 0; k < numNearest; k++)
	{
		unsigned int candidateOffset = 3 * candidateIdx; // Assumes points.shape[0] == 1
		ConstVector3<scalar_t> candidatePoint(&points[candidateOffset]);

		scalar_t candidateDistance = (surfacePoint - candidatePoint).norm();

		unsigned int offset = k + numNearest * surfaceIdx;// Assumes points.shape[0] == 1
		indice_t nearestIdx = nearest[offset];
		if (nearestIdx < 0)
		{
			nearest[offset] = candidateIdx;
			return;
		}

		unsigned int nearestOffset = 3 * nearestIdx;// Assumes points.shape[0] == 1
		ConstVector3<scalar_t> nearestPoint(&points[nearestOffset]);

		scalar_t nearestDistance = (surfacePoint - nearestPoint).norm();

		if (candidateDistance < nearestDistance)
		{
			nearest[offset] = candidateIdx;
			candidateIdx = nearestIdx;
		}
	}
}

template<typename scalar_t, typename indice_t>
__global__ void nearestKernel (
	indice_t * nearest,
	indice_t const* indices,
	scalar_t const* weights,
	unsigned int numPixels, unsigned int numSamples,
	scalar_t const* points,
	unsigned int * locks,
	unsigned int batchSize, unsigned int numPoints,
	unsigned int numNearest
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int pixelOffset = numSamples * (i + numPixels * b);
	
	// Identify the surface point
	indice_t surfaceIdx = -1;
	scalar_t maxWeight = -1.0;
	for (unsigned int k = 0; k < numSamples; k++)
	{
		unsigned int offset = k + pixelOffset;
		scalar_t weight = weights[offset];

		if (weight > maxWeight)
		{
			maxWeight = weight;
			surfaceIdx = indices[offset];
		}
	}
	if(surfaceIdx < 0)
	    return;

	// Compute distances from other points
	assert(surfaceIdx >= 0);
	unsigned int surfaceOffset = 3 * surfaceIdx; // Assumes points.shape[0] == 1
	ConstVector3<scalar_t> suracePoint(&points[surfaceOffset]);

	unsigned int lockOffset = surfaceIdx;  // Assumes points.shape[0] == 1;
	for (unsigned int k = 0; k < numSamples; k++)
	{
		unsigned int offset = k + pixelOffset;
		indice_t neighbourIdx = indices[offset];
		if ( (neighbourIdx < 0) || (neighbourIdx == surfaceIdx) )
			continue;

		bool leaveCritical = false;
		while (!leaveCritical) {
			if (atomicExch(&(locks[lockOffset]), 1u) == 0u) {
				// Begin critical section
				insert(nearest, surfaceIdx, neighbourIdx, points, numNearest);
				// End critical section
				leaveCritical = true;
				atomicExch(&(locks[lockOffset]), 0u);
			}
		}
		
	}

}

// Note: The current implementation most likely results in 2-way bank conflicts.
// In a future implementation, the indices and distances should be stored sequentially
// i_1, ..., i_n, d_1, ..., d_n
struct alignas(8) entry
{
	__device__ entry(int _i, float _d) : i(_i), d(_d) {}
	int i;
	float d;
};

template<typename scalar_t, typename indice_t>
__device__ void merge(
	entry * result,
	entry const* left,
	unsigned int leftSize,
	entry const* right,
	unsigned int rightSize
)
{
	unsigned int resultIdx = 0;
	unsigned int leftIdx = 0;
	unsigned int rightIdx = 0;
	while(leftIdx < leftSize && rightIdx < rightSize)
	{
		if ( left[leftIdx].d < right[rightIdx].d )
		{
			result[resultIdx] = left[leftIdx];
			leftIdx += 1;
		} 
		else
		{
			result[resultIdx] = right[rightIdx];
			rightIdx += 1;
		}
		resultIdx += 1;
	}

	while(leftIdx < leftSize)
	{
	    result[resultIdx] = left[leftIdx];
		resultIdx += 1;
		leftIdx += 1;
	}

	while(rightIdx < rightSize)
	{
		result[resultIdx] = right[rightIdx];
		resultIdx += 1;
		rightIdx += 1;
	}
}

template<typename scalar_t, typename indice_t>
__device__ void copy(
	entry * out,
	entry const* in,
	unsigned int length
)
{
	for(unsigned int i = 0; i < length; i++)
	{
		out[i] = in[i];
	}
}

template<typename scalar_t, typename indice_t>
__device__ void mergeNearest(
	indice_t * nearest,
	entry const* candidates,
	unsigned int numNearest,
	entry const* samples,
	unsigned int numSamples
)
{
	unsigned int nearestIdx = 0;
	unsigned int candidateIdx = 0;
	unsigned int sampleIdx = 0;

	// Assumes that numNearest < numSamples
	while (nearestIdx < numNearest && candidateIdx < numNearest && sampleIdx < numSamples)
	{
		indice_t nextIndex;
		if (candidates[candidateIdx].d < samples[sampleIdx].d)
		{
			nextIndex = candidates[candidateIdx].i;
			candidateIdx += 1;
		}
		else
		{
			nextIndex = samples[sampleIdx].i;
			sampleIdx += 1;
		}

		if (nearestIdx == 0 || nearest[nearestIdx - 1] != nextIndex)
		{
			nearest[nearestIdx] = nextIndex;
			nearestIdx += 1;
		}
	}
}

template<typename scalar_t, typename indice_t>
__global__ void stridedNearestKernel(
	indice_t * nearest,
	indice_t const* indices,
	scalar_t const* weights,
	unsigned int numPixels, unsigned int numSamples,
	scalar_t const* points,
	unsigned int * locks,
	unsigned int batchSize, unsigned int numPoints,
	unsigned int numNearest,
	unsigned int pixelsPerBlock,
	unsigned int numBlocks
)
{
	//extern __shared__ entry shared[];
	extern __shared__ __align__(sizeof(entry)) unsigned char shared_data[];
    entry * shared = reinterpret_cast<entry *>(shared_data);   
	if (threadIdx.x / numSamples >= pixelsPerBlock)
		return;

	unsigned int batchIdx = blockIdx.x / numBlocks;
	unsigned int chunkIdx = blockIdx.x % numBlocks;
	unsigned int pixelIdx = chunkIdx * pixelsPerBlock + threadIdx.x / numSamples;
	unsigned int sampleIdx = threadIdx.x % numSamples;

	unsigned int pixelOffset = numSamples * (pixelIdx + numPixels * batchIdx);
	unsigned int sampleOffset = sampleIdx + pixelOffset;

	// Search the surface index based on the weights
	indice_t surfaceIdx = indices[pixelOffset];
	scalar_t surfaceWeight = weights[pixelOffset];
	for(unsigned int k = 1; k < numSamples; k++)
	{
		unsigned int offset = k + pixelOffset;
		scalar_t weight = weights[offset];

		if (weight > surfaceWeight)
		{
			surfaceWeight = weight;
			surfaceIdx = indices[offset];
		}
	}
	unsigned int surfaceOffset = 3*surfaceIdx; // Assumes points.shape[0] == 1
	ConstVector3<scalar_t> surfacePoint(&points[surfaceOffset]);

	// Compute distance between the surface point and sample point
	//unsigned int surfaceOffset = 3 * surfaceIdx; // Assumes points.shape[0] == 1
	unsigned int pointOffset = 3 * indices[sampleOffset]; // Assumes points.shape[0] == 1
	scalar_t distance = INFINITY;
	if (indices[sampleOffset] >= 0)
	{
		ConstVector3<scalar_t> samplePoint(&points[pointOffset]);
		distance = (surfacePoint - samplePoint).norm();
	}
	
	// Store index and distance in shared map
	unsigned int sharedOffset = 2*numSamples*(threadIdx.x / numSamples);
	shared[sharedOffset + sampleIdx] = entry(indices[sampleOffset], distance);
	__syncthreads();
	
	// Perform pxiel-wise merge-sort on the shared memory map
	for(unsigned int mergeSize = 2; mergeSize < 2*numSamples; mergeSize *=2 )
	{
		if (sampleIdx % mergeSize == 0)
		{
			unsigned int mergeIdx = sampleIdx / mergeSize;
            unsigned int leftOffset = 2*mergeIdx*mergeSize/2;
            unsigned int rightOffset = (2*mergeIdx+1)*mergeSize/2;
            unsigned int resultOffset = sharedOffset + numSamples + mergeSize*mergeIdx;

            unsigned int leftLength = mergeSize/2;
            unsigned int rightLength = mergeSize/2;
            if (rightOffset + rightLength >= numSamples )
                rightLength = std::max(0, (int)numSamples - (int)rightOffset);

			merge<scalar_t, indice_t>(
				&shared[sharedOffset + resultOffset],
				&shared[sharedOffset + leftOffset],
				leftLength,
				&shared[sharedOffset + rightOffset],
				rightLength
			);

			copy<scalar_t, indice_t>(
				&shared[sharedOffset + leftOffset],
				&shared[sharedOffset + resultOffset],
				leftLength + rightLength
			);
		}
		__syncthreads();
	}

	// One thread for each pixel merges the sorted shared memory map 
	// with the samples in the nearest samples
	if (sampleIdx == 0)
	{
		bool leaveCritical = false;
		unsigned int lockOffset = surfaceIdx; // Assumes points.shape[0] == 1
		while (!leaveCritical) {
			if (atomicExch(&(locks[lockOffset]), 1u) == 0u) {
			// Begin critical section
			// Compute distances of current nearest samples in the output tensor
			// and store them in the second half of the shared map
			for(unsigned int k = 0; k < numNearest; k++)
			{
				unsigned int nearestOffset = k + numNearest * surfaceIdx;  // Assumes points.shape[0] == 1
				indice_t nearestIdx = nearest[nearestOffset];
				if (nearestIdx < 0)
				{
					shared[sharedOffset + numSamples + k] = entry(-1, INFINITY);
					continue;
				}

				ConstVector3<scalar_t> nearestPoint(&points[3*nearestIdx]); // Assumes points.shape[0] == 1
				scalar_t nearestDistance = (surfacePoint - nearestPoint).norm();

				shared[sharedOffset + numSamples + k] = entry(nearestIdx, nearestDistance);
			}

			mergeNearest<scalar_t, indice_t>(	
				&nearest[numNearest * surfaceIdx],
				&shared[sharedOffset + numSamples], //entry const& candidates,
				numNearest,
				&shared[sharedOffset],
				numSamples
			);
			
			// End critical section
			leaveCritical = true;
			atomicExch(&(locks[lockOffset]), 0u);
			}
		}
	}
}

template<typename scalar_t, typename indice_t>
__global__ void fastNearestKernel(
	indice_t * nearest,
	indice_t const* indices,
	scalar_t const* weights,
	unsigned int numPixels, unsigned int numSamples,
	scalar_t const* points,
	unsigned int * locks,
	unsigned int batchSize, unsigned int numPoints,
	unsigned int numNearest
)
{
    // Cast shared memory and initial the shared memory
	extern __shared__ __align__(sizeof(entry)) unsigned char shared_data[];
    entry * shared = reinterpret_cast<entry *>(shared_data);
    shared[threadIdx.x] = entry(-1, INFINITY);
    shared[numSamples + threadIdx.x] = entry(-1, INFINITY);
	
	unsigned int batchIndex = blockIdx.x / numPixels;
	unsigned int pixelIndex = blockIdx.x % numPixels;
	unsigned int sampleIndex = threadIdx.x;

	unsigned int pixelOffset = numSamples * (pixelIndex + numPixels * batchIndex);

	// Linear search for the surface point as the maximum of the filtered weights
	indice_t surfaceIdx = indices[pixelOffset];
	scalar_t surfaceWeight = weights[pixelOffset];
	for(unsigned int k = 1; k < numSamples; k++)
	{
		unsigned int offset = k + pixelOffset;
		scalar_t weight = weights[offset];

		if (weight > surfaceWeight)
		{
			surfaceWeight = weight;
			surfaceIdx = indices[offset];
		}
	}	
	if (surfaceIdx < 0)
	    return;
	unsigned int surfaceOffset = 3 * surfaceIdx; // Assumes that points.shape[0] == 1
	ConstVector3<scalar_t> surfacePoint(&points[surfaceOffset]);

	// Compute distance to the surface point for every sampled entry in parallel
	unsigned int sampleOffset = sampleIndex + pixelOffset;

	indice_t sample = indices[sampleOffset];
	scalar_t distance = INFINITY;
	if (sample >= 0 && sample != surfaceIdx)
	{
		unsigned int pointOffset = 3*sample; // Assumes that points.shape[0] == 1
		ConstVector3<scalar_t> samplePoint(&points[pointOffset]);
		distance = (surfacePoint - samplePoint).norm();
	}
	shared[sampleIndex] = entry(sample, distance);
	__syncthreads();

	// Peform merge sort on the memory map in parallel
	for(unsigned int mergeSize = 2; mergeSize < 2*numSamples; mergeSize *=2 )
	{
		if (sampleIndex % mergeSize == 0)
		{
			unsigned int mergeIndex = sampleIndex / mergeSize;

			unsigned int leftOffset = 2*mergeIndex*mergeSize/2;
			unsigned int rightOffset = (2*mergeIndex+1)*mergeSize/2;
			unsigned int sortedOffset = numSamples + leftOffset; // TODO!

			unsigned int leftLength = mergeSize/2;
			if(leftOffset + leftLength >= numSamples)
                leftLength = max(0, (int)numSamples - (int)leftOffset);
            unsigned int rightLength = mergeSize/2;
            if (rightOffset + rightLength >= numSamples )
                rightLength = max(0, (int)numSamples - (int)rightOffset);

			merge<scalar_t, indice_t>(
				&shared[sortedOffset],
				&shared[leftOffset],
				leftLength,
				&shared[rightOffset],
				rightLength
			);

			copy<scalar_t, indice_t>(
				&shared[leftOffset],
				&shared[sortedOffset],
				leftLength + rightLength
			);
		}
		__syncthreads();
	}
		
	// Merge sorted sampled indices with the indices that are currently the
	// nearest samples
	if (threadIdx.x == 0)
	{
	    
		// Critical section begin
		unsigned int lockOffset = surfaceIdx; // Assumes points.shape[0] == 1
		while (atomicCAS(&locks[lockOffset], 0, 1) != 0);
				
		// Store the distances from current nearest candidates 
		// to the surface point in shared memory.
		for (unsigned int k = 0; k < numNearest; k++)
		{
			unsigned int nearestOffset = k + numNearest * surfaceIdx;  // Assumes points.shape[0] == 1
			indice_t nearestIdx = nearest[nearestOffset];
			if (nearestIdx < 0)
			{
				shared[numSamples + k] = entry(-1, INFINITY);
				continue;
			}

			ConstVector3<scalar_t> nearestPoint(&points[3 * nearestIdx]); // Assumes points.shape[0] == 1
			scalar_t nearestDistance = (surfacePoint - nearestPoint).norm();
			shared[numSamples + k] = entry(nearestIdx, nearestDistance);
		}

		// Merge sorted sampled indices with the list of nearest candidates
		mergeNearest<scalar_t, indice_t>(
			&nearest[numNearest * surfaceIdx],  // Assumes points.shape[0] == 1
			&shared[numSamples],
			numNearest,
			shared,
			numSamples
			);


		// Critical section end
		__threadfence();
		atomicExch(&locks[lockOffset], 0);
	}
}

void cudaNearestSamples(
	at::Tensor & nearest,
	at::Tensor const& indices,
	at::Tensor const& weights,
	at::Tensor const& points,
	unsigned int numNearest
)
{

	unsigned int batchSize = indices.sizes()[0];
	unsigned int width = indices.sizes()[1];
	unsigned int height = indices.sizes()[2];
	unsigned int numSamples = indices.sizes()[3];

	dim3 threads(256, 1);
	dim3 grid((width*height + threads.x - 1) / threads.x, (batchSize + threads.y - 1) / threads.y);

	thrust::device_vector<unsigned int> pointsLock(points.sizes()[0] * points.sizes()[1]);
	thrust::fill(pointsLock.begin(), pointsLock.end(), 0u);

	AT_DISPATCH_FLOATING_TYPES(
		points.scalar_type(), "nearestKernel", ([&] {
			nearestKernel<scalar_t, int> << <grid, threads >> > (
				nearest.data_ptr<int>(),
				indices.data_ptr<int>(),
				weights.data_ptr<scalar_t>(),
				width*height, numSamples,
				points.data_ptr<scalar_t>(),
				thrust::raw_pointer_cast(pointsLock.data()),
				points.sizes()[0], points.sizes()[1],
				numNearest
				);
			})
	);
}

void cudaFastNearestSamples(
	at::Tensor & nearest,
	at::Tensor const& indices,
	at::Tensor const& weights,
	at::Tensor const& points,
	unsigned int numNearest
)
{
	unsigned int batchSize = indices.sizes()[0];
	unsigned int numPixels = indices.sizes()[1] * indices.sizes()[2];
	unsigned int numSamples = indices.sizes()[3];
	
	unsigned int numThreads = numSamples;
	unsigned int numBlocks = batchSize * numPixels;
	logging::log<logLevel_t::LOG_DEBUG, unsigned int>("cudaFastNearestSamples", " Threads per block", numThreads);
	logging::log<logLevel_t::LOG_DEBUG, unsigned int>("cudaFastNearestSamples", " Number of blocks", numBlocks);

	// Compute required shared memory footprint
	unsigned int sharedSize = 2 * sizeof(entry) * numSamples;
	logging::log<logLevel_t::LOG_DEBUG, unsigned int>("cudaFastNearestSamples", " Allocated shared bytes", sharedSize);

	thrust::device_vector<unsigned int> pointsLock(points.sizes()[0] * points.sizes()[1]);
	thrust::fill(pointsLock.begin(), pointsLock.end(), 0u);

	AT_DISPATCH_FLOATING_TYPES(
		points.scalar_type(), "fastNearestKernel", ([&] {
			fastNearestKernel<scalar_t, int> << <numBlocks, numThreads, sharedSize>> > (
				nearest.data_ptr<int>(),
				indices.data_ptr<int>(),
				weights.data_ptr<scalar_t>(),
				numPixels, numSamples,
				points.data_ptr<scalar_t>(),
				thrust::raw_pointer_cast(pointsLock.data()),
				points.sizes()[0], points.sizes()[1],
				numNearest
				);
			})
	);

}
