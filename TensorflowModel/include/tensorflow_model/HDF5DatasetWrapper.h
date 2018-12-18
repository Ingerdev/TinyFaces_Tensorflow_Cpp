#pragma once
#pragma warning(push, 0) 
#include <H5Cpp.h>
#include <string>
#include <vector>
#pragma warning(pop) 
namespace tiny_face_model
{
	namespace internal
	{
		class HDF5DatasetWrapper
		{
		public:
			HDF5DatasetWrapper(H5::H5File* file, const std::string& dataset_name);
			~HDF5DatasetWrapper();
			HDF5DatasetWrapper(HDF5DatasetWrapper&& o) noexcept;
			H5::DataSet& get_dataset() noexcept ;
			//dataset dimensions
			const std::vector<size_t> get_dims() const noexcept;

			//memory size of dataset
			size_t memory_size() const noexcept;

		private:
			//deny copying/assignment ops
			HDF5DatasetWrapper(HDF5DatasetWrapper const &) = delete;
			HDF5DatasetWrapper& operator=(HDF5DatasetWrapper const &) = delete;
			void swap(HDF5DatasetWrapper& r);

			static std::vector<size_t> get_dimensions(const H5::DataSpace& data_space);
			std::vector<size_t> dims_;
			size_t memory_size_;
			H5::DataSet dataset_;

		};
	}
}

