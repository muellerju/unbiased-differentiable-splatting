#pragma once

#include <iostream>
#include <cuda_runtime.h>

enum logLevel_t {
	LOG_NOTHING,
	LOG_CRITICAL,
	//LOG_ERROR,
	//LOG_WARNING,
	//LOG_INFO,
	LOG_DEBUG
};

/*namespace log_impl
{
	template<typename BaseType, typename InputIterator>
	class formatted_log_t
	{
	public:
		formatted_log_t(logLevel_t _level, const std::string& _msg, const InputIterator& _first, const InputIterator& _last) : msg(_msg), first(_first), last(_last), level(_level) {}
		~formatted_log_t()
		{
			// GLOBAL_LEVEL is a global variable and could be changed at runtime
			// Any customization could be here
			if (level <= GLOBAL_LEVEL)
			{
				std::cout << msg;
				thrust::copy(first, last, std::ostream_iterator<BaseType>(std::cout, " "));
				std::cout << std::endl;
			}
		}

	protected:
		logLevel_t level;
		const std::string& msg;
		const InputIterator& first;
		const InputIterator& last;
	};
}//namespace log_impl*/

namespace logging
{
	const logLevel_t GLOBAL_LEVEL = logLevel_t::LOG_DEBUG;

	template <logLevel_t level>
	void log(const std::string& prefix, const std::string& msg)
	{
		if (level <= GLOBAL_LEVEL) //TODO: Switch to c++17 and use constexpr
			std::cout << "> " << prefix << ": " << msg << std::endl;
	}

	template <logLevel_t level, typename scalar_t>
	void log(const std::string& prefix, const std::string& msg, scalar_t value)
	{
		if (level <= GLOBAL_LEVEL) //TODO: Switch to c++17 and use constexpr
			std::cout << "> " << prefix << ": " << msg << " " << value <<std::endl;
	}

	template <logLevel_t level>
	void log(const std::string& prefix, const std::string& msg, const int64_t* begin, const int64_t* end)
	{
		if (level <= GLOBAL_LEVEL) //TODO: Switch to c++17 and use constexpr
		{
			size_t length = std::distance(begin, end);
			std::cout << "> " << prefix << ": " << msg << " ("; //std::endl;
			for (size_t i = 0; i < length; i++)
				std::cout << *(begin + i) << " ";
			std::cout << ")" << std::endl;
		}
	}

	template<logLevel_t level>
	void log(const std::string& prefix, dim3 block, dim3 grid)
	{
		if (level <= GLOBAL_LEVEL) //TODO: Switch to c++17 and use constexpr
		{
			std::cout << "> " << prefix << 
				": Grid size (" << grid.x << ", " << grid.y << ", " << grid.z 
				<< ") with (" << block.x << ", " << block.y << ", " << block.z 
				<< ") threads per block" << std::endl;
		}
	}
}
