#include <torch/extension.h>
#include <cuda_runtime.h>

#include <thrust/copy.h>

#include "transformation.cuh"
#include "ewaSplatting.cuh"

#include "radixTree.cuh"
#include "buildRadixTree.cuh"
#include "sampleRadixTree.cuh"
//#include "maxRadixTree.cuh"
#include "buildDynamicRadixTree.cuh"
#include "sequentialPoissonRadixTree.cuh"
#include "sequentialPoissonDynamicTree.cuh"
#include "transformRadixTree.cuh"

#include "logging.h"

#define DIMS 2
#define HERMITEDEGREE 6

template<typename scalar_t>
struct CullingFunctor
{
	scalar_t width, height, focallength;
	const scalar_t* devPointsPtr;
	const scalar_t* devBoundsPtr;

	CullingFunctor(
		unsigned int _width, unsigned int _height, scalar_t _focallength,
		const scalar_t* _points,
		const scalar_t* _devBoundsPtr
	) :
		focallength(_focallength), devPointsPtr(_points), devBoundsPtr(_devBoundsPtr)
	{
		width = static_cast<scalar_t>(_width);
		height = static_cast<scalar_t>(_height);
	}

	__device__
		bool operator()(unsigned int idx)
	{
		const ConstVector3<scalar_t> point(&devPointsPtr[3 * idx]);
		Vector2<scalar_t> screenpoint = toScreenspace<scalar_t>(width, height, focallength, point);

		bool visible = (point[2] <= 0.0);
		for (unsigned int i = 0; i < 2; i++)
		{
			visible &= (devBoundsPtr[2 * i] <= screenpoint[i]);
			visible &= (devBoundsPtr[2 * i + 1] >= screenpoint[i]);
		}
		return visible;
	}
};

template<typename scalar_t>
struct IsotropicFunctor
{
	const scalar_t* devPointsPtr;
	const scalar_t* devNormalsPtr;
	const scalar_t* devStdDevsPtr;
	scalar_t width, height, focallength, smoothing;

	IsotropicFunctor(
		scalar_t _width, scalar_t _height, scalar_t _focallength,
		const scalar_t* _devPointsPtr,
		const scalar_t* _devNormalsPtr,
		const scalar_t* _devStdDevsPtr,
		scalar_t _smoothing
	) :
		devPointsPtr(_devPointsPtr),
		devNormalsPtr(_devNormalsPtr),
		devStdDevsPtr(_devStdDevsPtr)
	{
		width = _width;
		height = _height;
		focallength = _focallength;
		smoothing = _smoothing;
	}

	__device__
		scalar_t operator()(unsigned int pointIdx)
	{
		const scalar_t limit = 1.0;
		// Get the point to the point, normal, and stdDev for a valid index
		const ConstVector3<scalar_t> point(&devPointsPtr[pointIdx * 3]);
		const ConstVector3<scalar_t> normal(&devNormalsPtr[pointIdx * 3]);
		const ConstVector2<scalar_t> stdDev(&devStdDevsPtr[pointIdx * 2]);

		// Project cam points to screen space
		Vector2<scalar_t> screenpoint = toScreenspace<scalar_t>(width, height, focallength, point);

		// Build inverse jacobian
		Matrix22<scalar_t> invJk = buildInverseJacobian<scalar_t>(width, height, focallength, screenpoint, point, normal);

		// Build covariance matrix
		Matrix22<scalar_t> Vk = buildCovariance<scalar_t>(invJk, stdDev, smoothing);

		// Estimate a isotropic (clamped to an upper bound to avoid numerical instability)
		scalar_t h = 0.5 * (Vk[0] + Vk[3]);
		return min(h, limit);
	}
};

template<typename scalar_t>
struct IsotropicEigenFunctor
{
	const scalar_t* devPointsPtr;
	const scalar_t* devNormalsPtr;
	const scalar_t* devStdDevsPtr;
	scalar_t width, height, focallength, smoothing, precision;

	IsotropicEigenFunctor(
		scalar_t _width, scalar_t _height, scalar_t _focallength,
		const scalar_t* _devPointsPtr,
		const scalar_t* _devNormalsPtr,
		const scalar_t* _devStdDevsPtr,
		scalar_t _smoothing,
		scalar_t _precision
	) :
		devPointsPtr(_devPointsPtr),
		devNormalsPtr(_devNormalsPtr),
		devStdDevsPtr(_devStdDevsPtr)
	{
		width = _width;
		height = _height;
		focallength = _focallength;
		smoothing = _smoothing;
		precision = _precision;
	}

	__device__
		thrust::tuple<scalar_t, unsigned int> operator()(unsigned int pointIdx)
	{
		const scalar_t limit = 1.0;
		// Get the point to the point, normal, and stdDev for a valid index
		const ConstVector3<scalar_t> point(&devPointsPtr[pointIdx * 3]);
		const ConstVector3<scalar_t> normal(&devNormalsPtr[pointIdx * 3]);
		const ConstVector2<scalar_t> stdDev(&devStdDevsPtr[pointIdx * 2]);

		// Project cam points to screen space
		Vector2<scalar_t> screenpoint = toScreenspace<scalar_t>(width, height, focallength, point);

		// Build inverse jacobian
		Matrix22<scalar_t> invJk = buildInverseJacobian<scalar_t>(width, height, focallength, screenpoint, point, normal);

		// Build covariance matrix
		Matrix22<scalar_t> Vk = buildCovariance<scalar_t>(invJk, stdDev, smoothing);
		Matrix22<scalar_t> invVk = Vk.inv();
		
		// Compute eigen values
		scalar_t traceInvVk = invVk.trace();
		scalar_t l1 = 0.5 * traceInvVk + sqrt<scalar_t>(0.25*square<scalar_t>(traceInvVk) - invVk.det());
		scalar_t l2 = 0.5 * traceInvVk - sqrt<scalar_t>(0.25*square<scalar_t>(traceInvVk) - invVk.det());
		scalar_t lk = min(l1, l2);

		// Estimate a isotropic (clamped to an upper bound to avoid numerical instability)
		//scalar_t hk = 1.0 / sqrt<scalar_t>(lk);
		scalar_t hk = min(1.0, 1.0 / sqrt<scalar_t>(lk));
		return thrust::make_tuple<scalar_t, unsigned int>(hk, 1);
	}
};

template<typename scalar_t>
struct MeanFunctor
{
	__device__
		thrust::tuple<scalar_t, unsigned int> operator()(
			thrust::tuple<scalar_t, unsigned int> const& a, 
			thrust::tuple<scalar_t, unsigned int> const& b)
	{
		unsigned int leftCount = thrust::get<1>(a);
		scalar_t leftValue = thrust::get<0>(a);

		unsigned int rightCount = thrust::get<1>(b);
		scalar_t rightValue = thrust::get<0>(b);

		scalar_t mean = scalar_t(0);
		unsigned int total = 0;
		if (isfinite(leftValue))
		{
			mean += leftCount * leftValue;
			total += leftCount;
		}
		if (isfinite(rightValue))
		{
			mean += rightCount * rightValue;
			total += rightCount;
		}
		mean /= total;
		
		return thrust::make_tuple(mean, total);
	}
};

template<typename scalar_t, typename indice_t>
void batchSample(
	at::Tensor& indices, // Shape (bn, w, h, k)
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& points, // Shape (bn, n, 3)
	const at::Tensor& normals, // Shape (bn, n, 3)
	const at::Tensor& colors, // Shape (bn, n, 3) or shape(1, n, 3)
	const at::Tensor& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	double smoothing,
	double precision
)
{
	unsigned int numSamples = indices.sizes()[3];
	unsigned int batchSize = points.sizes()[0];
	unsigned int numSources = points.sizes()[1];
	logging::log<LOG_DEBUG, unsigned int>("cudaSample", "point count", numSources);

	thrust::device_vector<unsigned int> pointIndices(numSources);
	for (unsigned int k = 0; k < batchSize; k++)
	{

		// Compute point to current batch
		unsigned int outOffset = width * height * numSamples * k;
		indice_t* devBatchIndicesPtr = indices.data_ptr<indice_t>() + outOffset;

		unsigned int pointOffset = 3 * numSources * k;
		const scalar_t* devBatchPointsPtr = points.data_ptr<scalar_t>() + pointOffset;
		const scalar_t* devBatchNormalsPtr = normals.data_ptr<scalar_t>() + pointOffset;

		unsigned int colorOffset = colors.sizes()[0] > 1 ? 3 * colors.sizes()[1] * k : 0;
		const scalar_t* devBatchColorsPtr = colors.data_ptr<scalar_t>() + colorOffset;

		unsigned int stdDevOffset = stdDevs.sizes()[0] > 1 ? 2 * stdDevs.sizes()[1] * k : 0;
		const scalar_t* devBatchStdDevsPtr = stdDevs.data_ptr<scalar_t>() + stdDevOffset;

		// Estimate the isotropic standart deviation assumed for all points
		auto pointIdxFirst = thrust::make_counting_iterator<unsigned int>(0);
		auto pointIdxLast = thrust::make_counting_iterator<unsigned int>(numSources);

		IsotropicEigenFunctor<scalar_t> isotropicOp(
			width, height, focallength,
			devBatchPointsPtr,
			devBatchNormalsPtr,
			devBatchStdDevsPtr,
			smoothing, precision
		);
		thrust::tuple<scalar_t, unsigned int> stdDevResult = thrust::transform_reduce(pointIdxFirst, pointIdxLast, isotropicOp, thrust::make_tuple(scalar_t(0), 0), MeanFunctor<scalar_t>());
		scalar_t stdDev = thrust::get<0>(stdDevResult);
		unsigned int stdDevCount = thrust::get<1>(stdDevResult);
		logging::log<LOG_DEBUG, scalar_t>("approxFMMWeights", "Isotropic standard deviation", stdDev);
		logging::log<LOG_DEBUG, unsigned int>("approxFMMWeights", "Isotropic standard deviation", stdDevCount);

		// Define boundary of the screen space
		scalar_t normalization = 2.0 * M_PI * sqrt(stdDev);
		scalar_t radius = std::sqrt(2.0) * stdDev * std::sqrt(-std::log(precision) - std::log(normalization));
		logging::log<LOG_DEBUG, scalar_t>("cudaSample", "Query radius", radius);
		std::vector<scalar_t> bounds = {
			-radius, width + radius,
			-radius, height + radius
		};
		thrust::device_vector<scalar_t> devBounds = bounds;

		// Perform culling
		CullingFunctor<scalar_t> cullingOp(
			width, height, focallength,
			devBatchPointsPtr,
			thrust::raw_pointer_cast(devBounds.data()));

		auto pointIndicesFirst = pointIndices.begin();
		auto pointIndicesLast = thrust::copy_if(pointIdxFirst, pointIdxLast, pointIndicesFirst, cullingOp);
		unsigned int numPoints = std::distance(pointIndicesFirst, pointIndicesLast);
		logging::log<LOG_DEBUG, unsigned int>("cudaSample", "Number of remaining points", numPoints);

		DynamicTree<scalar_t, DIMS> radixTree(numPoints);
		buildDynamicTree<scalar_t, DIMS>(
			radixTree,
			pointIndicesFirst, pointIndicesLast,
			thrust::raw_pointer_cast(devBounds.data()),
			width, height, focallength,
			devBatchPointsPtr,
			devBatchColorsPtr,
			precision, stdDev
		);

		sequentialPoissonDynamicTree<scalar_t, DIMS>(
			devBatchIndicesPtr,
			radixTree,
			width, height,
			numSamples,
			precision, stdDev
		);
	}
}

void cudaSample(
	at::Tensor& indices, // Shape (bn, w, h, k)
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& points, // Shape (bn, n, 3)
	const at::Tensor& normals, // Shape (bn, n, 3)
	const at::Tensor& colors, // Shape (bn, n, 3) or shape(1, n, 3)
	const at::Tensor& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	double smoothing,
	double precision
)
{
	const auto& inputType = points.scalar_type();
	switch (inputType)
	{
	case at::ScalarType::Float:
	{
		batchSample<float, int>(
			indices,
			width, height, focallength,
			points, normals, colors, stdDevs,
			smoothing, precision
		);
	}
	break;
	default:
		logging::log<LOG_CRITICAL>("cudaSample", "Unsupported input type");
		break;
	}
}

template<typename scalar_t>
__global__ void covarianceKernel(
	scalar_t * covariance,
	unsigned int width, unsigned int height,
	scalar_t focallength,
	scalar_t const* points,
	scalar_t const* normals,
	scalar_t const* stdDevs,
	unsigned int batchSize, unsigned int numPoints,
	scalar_t smoothing
)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < batchSize * numPoints)
	{
		unsigned int batchIdx = idx / numPoints;
		unsigned int pointIdx = idx % numPoints;
		unsigned int offset = pointIdx + numPoints*batchIdx;

		ConstVector3<scalar_t> camPoint(&points[3*offset]);
		ConstVector3<scalar_t> camNormal(&normals[3*offset]);
		ConstVector2<scalar_t> stdDev(&stdDevs[2*pointIdx]);

		scalar_t widthf = static_cast<scalar_t>(width);
		scalar_t heightf = static_cast<scalar_t>(height);

		// Build inverse jacobian
		Vector2<scalar_t> screenpoint = toScreenspace(widthf, heightf, focallength, camPoint);
		Matrix22<scalar_t> invJk = buildInverseJacobian<scalar_t>(widthf, heightf, focallength, screenpoint, camPoint, camNormal);

		// Build covariance matrix
		Matrix22<scalar_t> Vk = buildCovariance<scalar_t>(invJk, stdDev, smoothing);

		// Write back the resulting variances
		for (unsigned int i = 0; i < 4; i++)
			covariance[i + 4*offset] = Vk[i];
 	}
}

void cudaCovariance(
	at::Tensor & covariances,
	unsigned int width, unsigned int height,
	double focallength,
	at::Tensor const& points,
	at::Tensor const& normals,
	at::Tensor const& stdDevs,
	double smoothing
)
{
	unsigned int batchSize = points.sizes()[0];
	unsigned int numPoints = points.sizes()[1];

	unsigned int numThreads = 256;
	unsigned int numBlocks = (batchSize*numPoints + numThreads - 1) / numThreads;

	logging::log<logLevel_t::LOG_DEBUG, unsigned int>("cudaCovariance", " Threads per block", numThreads);
	logging::log<logLevel_t::LOG_DEBUG, unsigned int>("cudaCovariance", " Number of blocks", numBlocks);

	AT_DISPATCH_FLOATING_TYPES(
		points.scalar_type(), "covarianceKernel", ([&] {
			covarianceKernel<scalar_t> <<<numBlocks, numThreads>>> (
				covariances.data_ptr<scalar_t>(),
				width, height,
				focallength,
				points.data_ptr<scalar_t>(),
				normals.data_ptr<scalar_t>(),
				stdDevs.data_ptr<scalar_t>(),
				batchSize, numPoints,
				smoothing
			);
		})
	);
}

template<typename scalar_t>
void batchApproxWeights(
	at::Tensor & weights, // Shape (bn, w, h, k)
	unsigned int width, unsigned int height,
	double focallength,
	at::Tensor const& points, // Shape (bn, n, 3)
	at::Tensor const& normals, // Shape (bn, n, 3)
	at::Tensor const& colors, // Shape (bn, n, 3) or Shape (1, n, 3)
	at::Tensor const& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	double smoothing,
	double precision
)
{
	unsigned int batchSize = points.sizes()[0];
	unsigned int numSources = points.sizes()[1];
	logging::log<LOG_DEBUG, unsigned int>("approxFMMWeights", "point count", numSources);

	thrust::device_vector<unsigned int> pointIndices(numSources);
	for (unsigned int k = 0; k < batchSize; k++)
	{

		// Compute point to current batch
		unsigned int outOffset = width * height * k;
		scalar_t* devBatchWeightPtr = weights.data_ptr<scalar_t>() + outOffset;

		unsigned int pointOffset = 3 * numSources * k;
		const scalar_t* devBatchPointsPtr = points.data_ptr<scalar_t>() + pointOffset;
		const scalar_t* devBatchNormalsPtr = normals.data_ptr<scalar_t>() + pointOffset;

		unsigned int stdDevOffset = stdDevs.sizes()[0] > 1 ? 2 * stdDevs.sizes()[1] * k : 0;
		const scalar_t* devBatchStdDevsPtr = stdDevs.data_ptr<scalar_t>() + stdDevOffset;

		// Estimate the isotropic standart deviation assumed for all points
		auto pointIdxFirst = thrust::make_counting_iterator<unsigned int>(0);
		auto pointIdxLast = thrust::make_counting_iterator<unsigned int>(numSources);
		/*IsotropicFunctor<scalar_t> isotropicOp(
			width, height, focallength,
			devBatchPointsPtr,
			devBatchNormalsPtr,
			devBatchStdDevsPtr,
			smoothing
		);*/
		IsotropicEigenFunctor<scalar_t> isotropicOp(
			width, height, focallength,
			devBatchPointsPtr,
			devBatchNormalsPtr,
			devBatchStdDevsPtr,
			smoothing, precision
		);
		thrust::tuple<scalar_t, unsigned int> stdDevResult = thrust::transform_reduce(pointIdxFirst, pointIdxLast, isotropicOp, thrust::make_tuple(scalar_t(0), 0), MeanFunctor<scalar_t>());
		scalar_t stdDev = thrust::get<0>(stdDevResult);
		unsigned int stdDevCount = thrust::get<1>(stdDevResult);
		logging::log<LOG_DEBUG, scalar_t>("approxFMMWeights", "Isotropic standard deviation", stdDev);
		logging::log<LOG_DEBUG, unsigned int>("approxFMMWeights", "Isotropic standard deviation", stdDevCount);

		// Define boundary of the screen space
		scalar_t normalization = 2.0 * M_PI * sqrt(stdDev);
		scalar_t radius = std::sqrt(2.0) * stdDev * std::sqrt(-std::log(precision) - std::log(normalization));
		logging::log<LOG_DEBUG, scalar_t>("approxFMMWeights", "Query radius", radius);
		std::vector<scalar_t> bounds = {
			-radius, width + radius,
			-radius, height + radius
		};
		thrust::device_vector<scalar_t> devBounds = bounds;

		// Perform culling
		CullingFunctor<scalar_t> cullingOp(
			width, height, focallength,
			devBatchPointsPtr,
			thrust::raw_pointer_cast(devBounds.data()));

		auto pointIndicesFirst = pointIndices.begin();
		auto pointIndicesLast = thrust::copy_if(pointIdxFirst, pointIdxLast, pointIndicesFirst, cullingOp);
		unsigned int numPoints = std::distance(pointIndicesFirst, pointIndicesLast);
		logging::log<LOG_DEBUG, unsigned int>("approxFMMWeights", "Number of remaining points", numPoints);

		// Construct radix tree
		RadixTree<scalar_t, DIMS> radixTree(numPoints, HERMITEDEGREE);
		buildRadixTreeTransform<scalar_t, HERMITEDEGREE>(
			radixTree,
			pointIndicesFirst, pointIndicesLast,
			thrust::raw_pointer_cast(devBounds.data()),
			width, height, focallength,
			devBatchPointsPtr,
			stdDev
		);

		// Compute weights
		gaussTransformRadixTree < scalar_t, DIMS, HERMITEDEGREE >(
			devBatchWeightPtr,
			width, height, precision,
			radixTree, stdDev
		);

	}
}

void cudaApproxWeights(
	at::Tensor& weights,
	unsigned int width, unsigned int height,
	double focallength,
	at::Tensor const& points, // Shape (bn, n, 3)
	at::Tensor const& normals, // Shape (bn, n, 3)
	at::Tensor const& colors, // Shape (bn, n, 2) or Shape (1, n, 2)
	at::Tensor const& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	double smoothing,
	double precision
)
{
	const auto& inputType = points.scalar_type();
	switch (inputType)
	{
	case at::ScalarType::Float:
	{
		batchApproxWeights<float>(
			weights,
			width, height, focallength,
			points, normals, colors, stdDevs,
			smoothing, precision
		);
	}
	break;
	default:
		logging::log<LOG_CRITICAL>("cudaApproxWeights", "Unsupported input type");
		break;
	}
}

