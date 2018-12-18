#pragma once
//helper class
#pragma warning(push, 0) 
#include <vector>
#include <algorithm>
#include <iterator>
#include <type_traits>
#include <cstdint>

#include <H5Cpp.h>
#include "tensorflow/core/framework/tensor.h"
#pragma warning(pop)  

namespace tf_helpers
{

	namespace internal
	{
		namespace tf = tensorflow;
		
		//dynamic type selection
		//num_entries_printed - how much tensor values to print.-1 - print all
		template <typename TStream>
		void print_tensor_impl_dynamic(TStream & output_stream, const tensorflow::Tensor & tensor,
			const size_t num_entries_printed = 0)
		{
			switch (tensor.dtype())
			{
			case tf::DataType::DT_FLOAT:
				internal::print_tensor_impl<float>(output_stream, tensor, num_entries_printed);
				break;
			case tf::DataType::DT_BOOL:
				internal::print_tensor_impl<bool>(output_stream, tensor, num_entries_printed);
				break;
			case tf::DataType::DT_DOUBLE:
				internal::print_tensor_impl<double>(output_stream, tensor, num_entries_printed);
				break;
			case tf::DataType::DT_INT16:
				internal::print_tensor_impl<int16_t>(output_stream, tensor, num_entries_printed);
				break;
			case tf::DataType::DT_INT32:
				internal::print_tensor_impl<int32_t>(output_stream, tensor, num_entries_printed);
				break;
			case tf::DataType::DT_INT64:
				internal::print_tensor_impl<int64_t>(output_stream, tensor, num_entries_printed);
				break;
			default:
				throw std::invalid_argument("tf_helpers: cannot print tensor of this type");

			}
		}

		template <typename T, typename TStream>
		void print_tensor_impl(TStream& output_stream,
			const  tf::Tensor& tensor,
			const size_t num_entries_printed = 0)
		{
			//dont print empty tensors
			if (!tensor.NumElements())
				return;


			int dims_count = tensor.dims();
			std::vector<tf::int64> dims(dims_count), dims_counter(dims_count);

			for (int k = 0; k < dims_count; k++)
			{
				dims[k] = tensor.dim_size(k);
				dims_counter[k] = 0;
			}

			//we should iterate on right-to left dimension
			std::reverse(dims.begin(), dims.end());

			auto tensor_data = tensor.flat<T>().data();

			// index of current element
			size_t index = 0;
			//when we should stop?
			bool last_element = false;

			while (true)
			{
				//check we not exceed requested number elements to print
				if ((num_entries_printed != 0) && (index >= num_entries_printed))
					break;

				output_stream << tensor_data[index] << " ";
				index++;

				//scalars havent dims count so after first value they have nothing
				if (dims_count == 0) break;

				//walk over out dimensions index array
				for (int k = 0; k < dims_count; k++)
				{   //increment current index cell.			
					dims_counter[k]++;

					//if this dimension is filled up then we reached limit of current dimension
					if (dims_counter[k] == dims[k])
					{
						output_stream << "\n";
						//reset current dimension counter 
						dims_counter[k] = 0;
						//check is this dimension higher dimension?
						if (k + 1 == dims_count)
						{
							//set up finish flag and leave cycle
							last_element = true;
							break;
						}
						else
							// inc and examine higher dimension 
							continue;
					}
					else
						//we not yet reached end of dimension, leave dimension indexing loop
						break;
					//
				}

				//we iterated fully.
				if (last_element)
					break;
			}
		}

		
	}

	void dump_tensor_to_file(const tensorflow::Tensor& tensor, const std::string& file_name);
	
	void print_tensor(const tensorflow::Tensor& tensor );

	//convert vector of integral type T to vector of type tensorflow::int64
	template <typename T>
	std::vector<tensorflow::int64> inline convert_vec_to_tf_int64(const std::vector<T>& h5vec)
	{
		static_assert(std::is_integral<T>::value, "cannot convert non-integral type to int64");

		std::vector<tensorflow::int64> tfvec;
		std::transform(h5vec.begin(), h5vec.end(),
			std::back_inserter(tfvec), [](T d) -> tensorflow::int64 { return tensorflow::int64(d); });

		return tfvec;
	}

	template <tensorflow::DataType T>
	class tf_hdf_helper
	{
	private:
		//make class pure static 
		tf_hdf_helper() = delete;

		//tensor from hdf5 dataset read/create implementation
		template <typename U>
		static tensorflow::Tensor create_load_tensor_impl(const H5::PredType hdf5_type,
			const std::vector<tensorflow::int64>& dims,const H5::DataSet& dataset)
		{
			auto shape = tensorflow::TensorShape(dims);

			tensorflow::Tensor tensor = tensorflow::Tensor::Tensor(T, shape);
			U *tensor_data_ptr = tensor.flat<U>().data();

			dataset.read(tensor_data_ptr, hdf5_type);

			//dump_tensor_to_file(tensor, ".\\tensor.txt");
			return tensor;
		}

	public:

		//create tensor from hdf5 dataset
		//dims - array of tensor dimensions
		//dataset - hdf5 dataset
		static tensorflow::Tensor create_load_tensor(const std::vector<tensorflow::int64>& dims,const H5::DataSet& dataset)
		{
			static_assert(false, "tf_hdf_helper have no specialization for your type.");
			return tensorflow::Tensor();
		}
	};

	//tf_hdf_helper specializations
	template<>
	inline tensorflow::Tensor tf_hdf_helper<tensorflow::DataType::DT_INT32>::create_load_tensor(
		const std::vector<tensorflow::int64>& dims,const H5::DataSet& dataset)
	{
		return create_load_tensor_impl<int32_t>(H5::PredType::NATIVE_INT32, dims, dataset);
	}

	template<>
	inline tensorflow::Tensor tf_hdf_helper<tensorflow::DataType::DT_FLOAT>::create_load_tensor(
		const std::vector<tensorflow::int64>& dims,const H5::DataSet& dataset)
	{
		return create_load_tensor_impl<float>(H5::PredType::NATIVE_FLOAT, dims, dataset);
	}

	template<>
	inline tensorflow::Tensor tf_hdf_helper<tensorflow::DataType::DT_DOUBLE>::create_load_tensor(
		const std::vector<tensorflow::int64>& dims, const H5::DataSet& dataset)
	{
		return create_load_tensor_impl<double>(H5::PredType::NATIVE_DOUBLE, dims, dataset);
	}


	//read hdf data to vector
	template<typename T>
	class vector_hdf_helper
	{
	private:
		//pure static class
		vector_hdf_helper() = delete;

		static std::vector<T> load_1d_vector_impl(const size_t dim, const H5::PredType hdf_type,
			const H5::DataSet& dataset)
		{
			std::vector<T> data(dim);
			dataset.read(data.data(), hdf_type);
			return data;
		}

		static std::vector<std::vector<T>> load_small_2d_array_impl(const std::vector<size_t>& dims,
			const H5::PredType hdf_type, const H5::DataSet& dataset)
		{
			std::unique_ptr<T> data_memory(new T[dims[0] * dims[1]]);
			dataset.read(data_memory.get(), hdf_type);

			std::vector<std::vector<T> > data_2d_array(dims[0], std::vector<T>(dims[1]));

			for (int k = 0; k < dims[0]; k++)
			{
				for (int k1 = 0; k1 < dims[1]; k1++)
				{
					data_2d_array[k][k1] = data_memory.get()[k*dims[1] + k1];
				}
			}


			return data_2d_array;
		}

	public:
		static std::vector<T> load_1d_vector(const size_t dim,const H5::DataSet& dataset);

		static std::vector<std::vector<T>> load_small_2d_array(
			const std::vector<size_t>& dims,
			const H5::DataSet& dataset);

	};

	//template specializations.
	template <typename T>
	inline std::vector<T> vector_hdf_helper<T>::load_1d_vector(const size_t dim,const H5::DataSet& dataset)
	{
		//prevent unspecialized instantiation
		static_assert(false, "vector_hdf_helper class doesnt specialized for this type. Please provide specialization.");
		return std::vector<T>();
	}

	template <>
	inline std::vector<float> vector_hdf_helper<float>::load_1d_vector(const size_t dim,
		const H5::DataSet& dataset)
	{
		return load_1d_vector_impl(dim, H5::PredType::NATIVE_FLOAT, dataset);
	}

	template <typename T>
	inline std::vector<std::vector<T>> vector_hdf_helper<T>::load_small_2d_array(
		const std::vector<size_t>& dims,
		const H5::DataSet& dataset)
	{
		//prevent unspecialized instantiation
		static_assert(false, "vector_hdf_helper class doesnt specialized for this type. Please provide specialization.");
		return std::vector<std::vector<T>>(0, std::vector<T>(0));
	}

	template <>
	inline std::vector<std::vector<double>> vector_hdf_helper<double>::load_small_2d_array(
		const std::vector<size_t>& dims,
		const H5::DataSet& dataset)
	{
		return load_small_2d_array_impl(dims, H5::PredType::NATIVE_DOUBLE, dataset);
	}

	template <tensorflow::DataType T>
	struct tf_dtype_to_standart_type_mapper
	{

	};

	//template parameter is value of type "tensorflow::DataType"
	template<tensorflow::DataType T>
	class tf_stl_helper
	{
	private:
		//make class pure static 
		tf_stl_helper() = delete;

		//non-specialized type
		class NonExistentType {};

		//specialization type storage
		class impl_class {
			NonExistentType val;
		};

		template <typename It>
		static tensorflow::Tensor create_load_tensor_impl(
			const std::vector<tensorflow::int64> dims,
			It begin, It end)
		{
			//get specialization type
			using V = decltype(impl_class::val);

			//assert for not specialized type
			static_assert(!std::is_same<V, NonExistentType>::value,
				"tf_stl_helper have no specialization for your type.");

			//assert for mismatch integral/floating_point target type and source container		
			//supress assert if type not specialized (avoid second error message)
			static_assert(
				(std::is_integral<V>::value && std::is_integral<It::value_type>::value) ||
				(std::is_floating_point<V>::value && std::is_floating_point<It::value_type>::value) ||
				(std::is_same<V, NonExistentType>::value),
				"tf_stl_helper target type and container type mismatch. They both should be integral or floating_point");

			//real code. here we have and use every type we want.
			auto shape = tensorflow::TensorShape(dims);

			auto tensor = tensorflow::Tensor::Tensor(T, shape);
			auto tensor_mem_ptr = tensor.flat<V>().data();

			std::copy(begin, end, tensor_mem_ptr);

			return tensor;
		};

	public:

		//create tensor and fill it with values from iterators
		template <typename It>
		static tensorflow::Tensor create_load_tensor(
			const std::vector<tensorflow::int64> dims,
			It begin, It end)
		{
			return create_load_tensor_impl(dims, begin, end);
		};

		//create tensor and fill it with values from container
		template <typename Cont>
		static tensorflow::Tensor create_load_tensor(
			const std::vector<tensorflow::int64> dims,
			Cont container)
		{
			return create_load_tensor_impl(dims, begin(container), end(container));
		};
	};


	//tf_stl_helper template specializations.
	template <>
	class tf_stl_helper<tensorflow::DataType::DT_INT32>::impl_class { int32_t val; };

	template <>
	class tf_stl_helper<tensorflow::DataType::DT_FLOAT>::impl_class { float val; };





	class test
	{
		test()
		{
			auto dims = std::vector<tensorflow::int64>(5);

			auto vec_float = std::vector<float>(5);

			auto vec_int = std::vector<int>(5);

			auto x_float = tf_stl_helper<tensorflow::DataType::DT_FLOAT>::create_load_tensor(dims, vec_float.begin(), vec_float.end());
			//auto x_float_error_types_mismatch = tf_stl_helper<tensorflow::DataType::DT_FLOAT>::create_load_tensor(dims, vec_int.begin(), vec_int.end());
			//auto x_float_error_unknown_type = tf_stl_helper<tensorflow::DataType::DT_DOUBLE>::create_load_tensor(dims, vec_float.begin(), vec_float.end());

			auto xx_int = tf_stl_helper<tensorflow::DataType::DT_INT32>::create_load_tensor(dims, vec_int.begin(), vec_int.end());
			auto xx_float = tf_stl_helper<tensorflow::DataType::DT_FLOAT>::create_load_tensor(dims, vec_float.begin(), vec_float.end());
			//auto xx_float_error = tf_stl_helper<tensorflow::DataType::DT_FLOAT>::create_load_tensor(dims, vec_int.begin(), vec_int.end());
			//auto x_int_error_types_mismatch = tf_stl_helper<tensorflow::DataType::DT_INT32>::create_load_tensor(dims, vec_float.begin(), vec_float.end());
			//auto x_int_error_unknown_type = tf_stl_helper<tensorflow::DataType::DT_INT64>::create_load_tensor(dims, vec_int.begin(), vec_int.end());
			auto y_float = tf_stl_helper<tensorflow::DataType::DT_FLOAT>::create_load_tensor(dims, vec_float);
			auto y_int = tf_stl_helper<tensorflow::DataType::DT_INT32>::create_load_tensor(dims, vec_int);
			//auto y_int_type_mismatch = tf_stl_helper<tensorflow::DataType::DT_INT32>::create_load_tensor(dims, vec_float);
			//auto y_float_type_mismatch = tf_stl_helper<tensorflow::DataType::DT_FLOAT>::create_load_tensor(dims, vec_int);
			//auto y_error_unknown_type = tf_stl_helper<tensorflow::DataType::DT_INT64>::create_load_tensor(dims, vec_int);
			//auto y_error_msmatch_type = tf_stl_helper<tensorflow::DataType::DT_INT32>::create_load_tensor(dims, vec_float);

			auto dataset = H5::DataSet();

			auto m = tf_hdf_helper<tensorflow::DataType::DT_FLOAT>::create_load_tensor(dims, dataset);
			m = tf_hdf_helper<tensorflow::DataType::DT_INT32>::create_load_tensor(dims, dataset);
			//auto m_error_unknown_type = tf_hdf_helper<tensorflow::DataType::DT_INT64>::create_load_tensor(dims, dataset);

			auto v = vector_hdf_helper<float>::load_1d_vector(2, dataset);
			//auto v_error_unknown_type = vector_hdf_helper<int>::load_1d_vector(2, dataset);		
			auto vd = vector_hdf_helper<double>::load_small_2d_array(std::vector<size_t>(5), dataset);
			//auto vd_error_unknown_type = vector_hdf_helper<float>::load_small_2d_array(std::vector<size_t>(5), dataset);

		}
	};
}