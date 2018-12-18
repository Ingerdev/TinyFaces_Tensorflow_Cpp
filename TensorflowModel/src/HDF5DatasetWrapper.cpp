#include "tensorflow_model/HDF5DatasetWrapper.h"
#pragma warning(push, 0) 
#include <algorithm>
#include <utility>
#include <iostream>
#pragma warning(pop) 

namespace tiny_face_model
{
	namespace internal
	{
		HDF5DatasetWrapper::HDF5DatasetWrapper(H5::H5File* file, const std::string& dataset_name)
		{
			dataset_ = file->openDataSet(dataset_name);
			H5::DataSpace data_space = dataset_.getSpace();

			dims_ = get_dimensions(data_space);

		}

		void HDF5DatasetWrapper::swap(HDF5DatasetWrapper & r)
		{
			std::swap(memory_size_, r.memory_size_);
			std::swap(dims_, r.dims_);
			std::swap(dataset_, r.dataset_);
		}

		std::vector<size_t> HDF5DatasetWrapper::get_dimensions(const H5::DataSpace& data_space)
		{
			int dims_count = data_space.getSimpleExtentNdims();

			std::vector<hsize_t> dims_out(dims_count);
			data_space.getSimpleExtentDims(dims_out.data());

			std::vector<size_t> dims;
			std::transform(dims_out.begin(), dims_out.end(),
				std::back_inserter(dims), [](hsize_t d) -> size_t { return static_cast<size_t>(d); });

			return dims;
		}

		HDF5DatasetWrapper::~HDF5DatasetWrapper()
		{
			dataset_.close();
		}

		HDF5DatasetWrapper::HDF5DatasetWrapper(HDF5DatasetWrapper && r) noexcept
		{
			r.swap(*this);
		}

		H5::DataSet& HDF5DatasetWrapper::get_dataset() noexcept
		{
			return dataset_;
		}

		const std::vector<size_t> HDF5DatasetWrapper::get_dims() const noexcept
		{
			return dims_;
		}

		size_t HDF5DatasetWrapper::memory_size() const noexcept
		{
			return memory_size_;
		}
	}
}
