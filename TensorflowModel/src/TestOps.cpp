#include "tensorflow_model/TestOps.h"
#pragma warning(push, 0)
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/logging_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/core/framework/types.pb.h"

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#pragma warning(pop)
#include "tensorflow_model/tf_helpers.h"

TestOps::TestOps() {}

TestOps::~TestOps() {}

void TestOps::test_slice_op()
{
	auto scope = tf::Scope::NewRootScope();
	// auto data = std::array<float, 8>({1.0,2,3,4,5,6,7,8});
	// auto tensor = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({ 3,2 }));
	// std::copy_n(std::begin(data),std::end(data),tensor.flat<float>().data());
	auto const_tensor = tf::ops::Const(scope, { 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	check_status(scope);
	auto sliced_tensor = tf::ops::Slice(scope, const_tensor, { 0, 0 }, { 2, 1 });
	check_status(scope);

	tf::ClientSession session(scope);

	std::vector<tf::Tensor> outputs(10);
	std::cout << std::endl << "Session.Run started" << std::endl;
	session.Run({ sliced_tensor }, &outputs);
	std::cout << "result: " << outputs[0].DebugString();

	std::cout << std::endl << "Session.Run ended" << std::endl;

	// so... first argument is n-dimensional start point (0-based)
	// second argument is n-dimensional matrix should be exctracted.
	// for single value it should be {1,1.....1} where count of 1 equal dimensions
}

void TestOps::test_concat_op()
{
	auto scope = tf::Scope::NewRootScope();
	// auto data = std::array<float, 8>({1.0,2,3,4,5,6,7,8});
	// auto tensor = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({ 3,2 }));
	// std::copy_n(std::begin(data),std::end(data),tensor.flat<float>().data());
	auto const_tensor_one = tf::ops::Const(scope, { 1 }, { 1 });
	auto const_tensor_two = tf::ops::Const(scope, { 2 }, { 1 });
	check_status(scope);
	auto concat_tensor =
		tf::ops::Concat(scope, { const_tensor_one, const_tensor_two }, 0);
	check_status(scope);

	tf::ClientSession session(scope);

	std::vector<tf::Tensor> outputs(10);
	std::cout << std::endl << "Session.Run started" << std::endl;
	session.Run({ concat_tensor }, &outputs);
	std::cout << "result: " << outputs[0].DebugString();

	std::cout << std::endl << "Session.Run ended" << std::endl;

	// so... axis 0 is for _rows (adding vertically)
	// for horizontal concat axis 1 should be used
	// beware:
	// if tensor is 0-d (single number) then allowed axis is 0 only
}

void TestOps::test_scatter_nd_update()
{
	auto scope = tf::Scope::NewRootScope();
	
	auto const_tensor =
		tf::ops::Const(scope, { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 });
	auto filled_tensor = tf::ops::Fill(scope, { 2, 3 }, 2.0);
	auto mult_tensor =
		tf::ops::Multiply(scope, tf::ops::Cast(scope, const_tensor, tf::DT_FLOAT),
			tf::ops::Cast(scope, filled_tensor, tf::DT_FLOAT));
	auto data_tensor = tf::ops::Variable(scope, { 2, 3 }, tf::DataType::DT_FLOAT);
	tf::ops::Assign(scope, data_tensor, mult_tensor,
		tf::ops::Assign::Attrs().ValidateShape(false));
	// auto print_node = tf::ops::Print(scope, data_tensor, {
	// tf::Input(tf::ops::Shape(scope,data_tensor)) });
	check_status(scope);
	auto scattered_tensor = tf::ops::ScatterNdUpdate(
		scope, data_tensor, { {0, 1} }, { {float(10.0), float(0), float(0)} });
	check_status(scope);

	tf::ClientSession session(scope);

	std::vector<tf::Tensor> outputs(10);
	std::cout << std::endl << "Session.Run started" << std::endl;
	session.Run({ scattered_tensor }, &outputs);
	std::cout << "result: " << outputs[0].DebugString();
	print_tensor_to_file(outputs[0], "\\tensor.txt");

	std::cout << std::endl << "Session.Run ended" << std::endl;

	// so... ScatterNd can produce only zero-initialized tensors with
	// slices initialized as you want. Miis our needs.
	// Scatter*Update requires mutating tensor (variable) and have complex
	// requirments fro arguments.
}

void TestOps::test_gather_nd()
{
	auto scope = tf::Scope::NewRootScope();

	auto data_tensor =
		tf::ops::Const(scope, { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 });
	auto data_float_tensor = tf::ops::Cast(scope, data_tensor, tf::DataType::DT_FLOAT);

	auto filled_tensor = tf::ops::Fill(scope, { 2, 3 }, 2.0);
	
	
	check_status(scope);
	auto gathered_tensor = tf::ops::GatherNd(scope, data_float_tensor, { { 1, 2 },{0,1} });
	check_status(scope);

	tf::ClientSession session(scope);

	std::vector<tf::Tensor> outputs(10);
	std::cout << std::endl << "Session.Run started" << std::endl;
	session.Run({ gathered_tensor }, &outputs);
	std::cout << "result: " << outputs[0].DebugString();
	print_tensor_to_file(outputs[0], "\\tensor.txt");

	std::cout << std::endl << "Session.Run ended" << std::endl;

	// so... GatherND works as intented.
}

void TestOps::test_tensor_rework()
{
	auto scope = tf::Scope::NewRootScope();
	auto const_tensor =
		tf::ops::Const(scope, { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 2, 3 });

	auto filled_tensor = tf::ops::Fill(scope, { 2, 3 }, 2.0);
	auto mult_tensor =
		tf::ops::Multiply(scope, tf::ops::Cast(scope, const_tensor, tf::DT_FLOAT),
			tf::ops::Cast(scope, filled_tensor, tf::DT_FLOAT));

	// 1. get tensor last dim with proper
	// auto last_dim =

	tf::ClientSession session(scope);

	std::vector<tf::Tensor> outputs(10);
	std::cout << std::endl << "Session.Run started" << std::endl;
	session.Run({ mult_tensor }, &outputs);
	std::cout << "result: " << outputs[0].DebugString();
	print_tensor_to_file(outputs[0], "\\tensor.txt");

	std::cout << std::endl << "Session.Run ended" << std::endl;
}

void TestOps::test_broadcast()
{
	auto scope = tf::Scope::NewRootScope();
	auto const_tensor =
		tf::ops::Const(scope, { float(1.0), float(0.0), float(1.0) }, { 3 });
	auto broadcasted_tensor =
		tf::ops::BroadcastTo(scope, const_tensor, { 1, 5, 5, 3 });

	check_status(scope);

	tf::ClientSession session(scope);

	std::vector<tf::Tensor> outputs(10);
	std::cout << std::endl << "Session.Run started" << std::endl;
	session.Run({ broadcasted_tensor }, &outputs);
	std::cout << "result: " << outputs[0].DebugString();
	print_tensor_to_file(outputs[0],"\\tensor.txt");

	std::cout << std::endl << "Session.Run ended" << std::endl;
	// it works! only case - dimensions should match
	//(for case of 1d array last target tensor's dimension should be the same as
	//length of 1d tensor)
}

void TestOps::check_status(const tf::Scope& scope)
{
	std::cout << scope.status().error_message() << std::endl;
	if (!scope.status().ok()) std::cout << "Error!";
}

std::string TestOps::get_current_path()
{
	char result[MAX_PATH];
	return std::string(result, GetModuleFileName(NULL, result, MAX_PATH));
}

void TestOps::print_tensor_to_file(const tensorflow::Tensor tensor, const std::string & file_name)
{
	tf_helpers::dump_tensor_to_file(tensor, get_current_path() + file_name);
}
