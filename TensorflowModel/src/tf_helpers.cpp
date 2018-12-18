#include "tensorflow_model/tf_helpers.h"
#pragma warning(push, 0) 
#include <fstream>
#include <iostream>
#include <stdexcept>
#pragma warning(pop) 


namespace tf_helpers
{
	namespace tf = tensorflow;


	void  dump_tensor_to_file(const tf::Tensor& tensor, const std::string& file_name)
	{
		std::ofstream wfile(file_name);

		internal::print_tensor_impl_dynamic(wfile, tensor);
		wfile.close();
	}

	void print_tensor(const tf::Tensor & tensor)
	{

		int dims_count = tensor.dims();
		std::cout << "Tensor dimensions: " << dims_count << ":[ ";
		for (int k = 0; k < dims_count; k++)
		{
			std::cout << tensor.dim_size(k) << " ";
		}
		std::cout << "]\n";
		std::cout << "elements count: " << tensor.NumElements() << "\n";

		auto& cout_ref = std::cout;
		internal::print_tensor_impl_dynamic(cout_ref, tensor/*,10*/);
	}

}