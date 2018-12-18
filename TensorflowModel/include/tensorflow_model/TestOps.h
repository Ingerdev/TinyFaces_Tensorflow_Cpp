#pragma once
#pragma warning(push, 0) 
#include <H5Cpp.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/scope.h"
#include <string>
#pragma warning(pop) 

namespace tf = tensorflow;

class TestOps
{
public:
	TestOps();
	~TestOps();
	void test_slice_op();
	void test_concat_op();
	
	//failed
	void test_scatter_nd_update();
	

	void test_gather_nd();

	void test_tensor_rework();
	void test_broadcast();
private:
	void check_status(const tf::Scope& scope);
	std::string get_current_path();
	void print_tensor_to_file(const tensorflow::Tensor, const std::string& file_name);
	
	
	
};

