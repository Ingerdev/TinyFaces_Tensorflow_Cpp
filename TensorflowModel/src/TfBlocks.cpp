#include "tensorflow_model/TfBlocks.h"

#pragma warning(push, 0)   
#include <iostream>
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/logging_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/client/client_session.h"							   
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma warning(pop)  

#include "tensorflow_model/tf_helpers.h"
#include "tensorflow_model/Exceptions.h"
#include "tensorflow_model/HDF5DatasetWrapper.h"

namespace tiny_face_model
{
	namespace internal
	{
		

		tf::Output TfBlocks::float_variable_on_cpu(const tf::Scope& scope, const std::string & name, const std::string& storage_postfix,
			const std::string& variable_postfix, const tf::TensorShape& shape, const tf::DataType dtype) const
		{
			(void)dtype;
			(void)variable_postfix;

			//compose storage name (it is name in hdf5 base)
			std::string storage_name = name + storage_postfix;


			//create tensor and use it in initializer		
			auto initializer = tf::Input::Initializer(tensor_on_cpu<tf::DataType::DT_FLOAT>(storage_name));

			assert(initializer.tensor.shape() == shape);
			//copy data
			auto const_tensor = tf::ops::Const(scope, initializer);
			return const_tensor;

			//create variable
			//auto var = tf::ops::Variable(scope, shape, dtype, tf::ops::Variable::Attrs().SharedName(name + variable_postfix));

			//auto print_node = tf::ops::Print(scope, const_tensor, tf::InputList({ const_tensor }),
			//tf::ops::Print::Attrs().Message("printing const_node... ").FirstN(5).Summarize(0));

			//return tf::ops::Assign(scope, var, const_tensor, tf::ops::Assign::Attrs().ValidateShape(true));
			check_status(scope);

		}

		tf::Output TfBlocks::weight_variable_on_cpu(const tf::Scope& scope, const std::string & name,
			const tf::TensorShape& shape, const tf::DataType dtype) const
		{
			check_status(scope);
			return float_variable_on_cpu(scope, name, std::string("_filter"), std::string("_f"), shape, dtype);
		}

		tf::Output TfBlocks::bias_variable_on_cpu(const tf::Scope& scope, const std::string & name,
			const tf::TensorShape& shape, const tf::DataType dtype) const
		{
			check_status(scope);
			return float_variable_on_cpu(scope, name, std::string("_bias"), std::string("_b"), shape, dtype);
		}

		std::vector<tf::Output> TfBlocks::bn_variable_on_cpu(const tf::Scope& scope, const std::string & name,
			const tf::TensorShape& shape, const tf::DataType dtype) const
		{
			std::string complete_name = "bn" + name.substr(3);

			//if name start with "conv" substring
			if (name.find("conv") == 0)
				complete_name = "bn_" + name;

			std::vector<tf::Output> output_vars;
			output_vars.push_back(float_variable_on_cpu(scope, complete_name, std::string("_scale"), std::string("_scale"), shape, dtype));
			output_vars.push_back(float_variable_on_cpu(scope, complete_name, std::string("_offset"), std::string("_offset"), shape, dtype));
			output_vars.push_back(float_variable_on_cpu(scope, complete_name, std::string("_mean"), std::string("_mean"), shape, dtype));
			output_vars.push_back(float_variable_on_cpu(scope, complete_name, std::string("_variance"), std::string("_variance"), shape, dtype));
			return output_vars;
		}

		tf::Output TfBlocks::conv_block(const tf::Scope& scope, const  tf::Input& prev_layer,
			const std::string& name, const tf::TensorShape& shape,
			const std::array<int, 4> strides, const std::string& padding, const bool has_bias,
			const bool add_relu, const bool add_bn, const float eps) const
		{
			(void)eps;

			assert(shape.dims() == 4);

			auto weights = weight_variable_on_cpu(scope, name, shape);

			auto conv = tf::ops::Conv2D(scope, prev_layer, weights, strides, padding);

			tf::Output pre_activation = conv;
			if (has_bias)
			{
				tf::Output bias = bias_variable_on_cpu(scope, name, { shape.dim_size(3) });
				pre_activation = tf::ops::BiasAdd(scope, pre_activation, bias);
			}

			if (add_bn)
			{
				auto properties = bn_variable_on_cpu(scope, name, { shape.dim_size(shape.dims() - 1) });
				auto scale = properties[0];
				auto offset = properties[1];
				auto mean = properties[2];
				auto variance = properties[3];

				//todo: add eps parameter to attr
				pre_activation = tf::ops::FusedBatchNorm(scope, pre_activation, scale, offset, mean, variance,
					tf::ops::FusedBatchNorm::Attrs().IsTraining(false)).y;
			}



			if (add_relu)
				pre_activation = (tf::Output)tf::ops::Relu(scope, pre_activation);
			
			return pre_activation;

		}

		tf::Output TfBlocks::conv_trans_layer(const tf::Scope& scope, const  tf::Input& prev_layer,
			const std::string& name, const tf::TensorShape& shape,
			const std::array<int, 4> strides,
			const std::string& padding, const bool has_bias) const
		{
			assert(shape.dims() == 4);
			auto weights = weight_variable_on_cpu(scope, name, shape);

			//nb, h, w, nc = ...
			auto list = tf::ops::Split(scope, 0, tf::ops::Shape(scope, prev_layer), 4);
			//tf::ops::Shape(_scope, prev_layer).

			auto nb = list[0];//tf::ops::Gather(scope, tf::Input(list), { 0 }); //list[0];
			auto h = list[1];
			auto w = list[2];
			auto nc = list[3];
			auto first = nb;
			auto second_1 = tf::ops::Multiply(scope, tf::ops::Sub(scope, h, 1), strides[1]);

			auto second_2 = tf::ops::Add(scope, tf::ops::Sub(scope, second_1, 3),
				tf::ops::Cast(scope, shape.dim_size(0), tf::DataType::DT_INT32));//shape.dim_size(0));	

			auto third_1 = tf::ops::Multiply(scope, tf::ops::Sub(scope, w, 1), strides[2]);

			auto third_2 = tf::ops::Add(scope, tf::ops::Sub(scope, third_1, 3),
				tf::ops::Cast(scope, shape.dim_size(1), tf::DataType::DT_INT32));

			auto fourth = nc;
			auto output_shape = tf::ops::Concat(scope, { tf::Input(first), second_2, third_2, fourth }, 0);

			//output_shape = tf.stack([nb, (h - 1) * strides[1] - 3 + shape[0], (w - 1) * strides[2] - 3 + shape[1], nc])[:, 0]
			//auto stacked_shape = tf::ops::Stack(scope, stack_list);

			//[:, 0]
			//begin = 0; size = -1;
			//begin = 0; size = 1;
			//auto filtered_shape = tf::ops::Slice(scope, stacked_shape, { 0,0 }, { -1,1 });
			//auto output_shape = tf::ops::Shape(scope, filtered_shape);

			//original code:
			//conv = tf.nn.conv2d_transpose(bottom, weight, output_shape, strides, padding = padding)

			//examples:
			//deconv = conv2d_transpose(activations, W_conv1, output_shape = [1, 28, 28, 1], padding = 'SAME')
			//deconv = conv2d_backprop_input([1,28,28,1],W_conv1,activations, strides=strides, padding='SAME')

			//translation:
			//conv2d_backprop_input(output_shape,weights,prev_layer, strides=strides, padding='SAME')
			//auto pl_shape = tf::ops::Shape(scope, prev_layer);
			//auto pl_shape_4 = tf::ops::Slice(scope, pl_shape, { 1 }, { 3 });
			//input size aka output_shape: 1 5 5 1 (must be 1 5 5 125)
			//weight_shape: 4 4 125 125 
			//prev_layer_shape: 1 3 3 125		
			auto conv = tf::ops::Conv2DBackpropInput(scope, output_shape, weights, prev_layer, strides, padding);


			//if has_bias:
			//bias = self._bias_variable_on_cpu(name, shape[3])
			if (has_bias)
			{
				auto bias = bias_variable_on_cpu(scope, name, { shape.dim_size(3) });
				return tf::ops::BiasAdd(scope, conv, bias);
			}
			return conv;
		}

		/*
		Args:
		prev_layer: A layer before this block.
		name : Name of the block.
		in_channel : number of channels in a input tensor.
		neck_channel : number of channels in a bottleneck block.
		out_channel : number of channels in a output tensor.
		trunk : a tensor in a identity path.
		Returns :
		a block of layers
		*/
		tf::Output TfBlocks::residual_block(const tf::Scope & scope, const  tf::Input & prev_layer, const std::string & name,
			const int in_channel, const int neck_channel, const int out_channel, const tf::Input & trunk) const
		{
			std::array<int, 4> new_strides = { 1, 1, 1, 1 };

			auto res3_prefix = std::string("res3a");
			auto res4_prefix = std::string("res4a");
			if ((!name.compare(0, res3_prefix.size(), res3_prefix)) ||
				(!name.compare(0, res4_prefix.size(), res4_prefix)))
			{
				new_strides = { 1, 2, 2, 1 };
			}

			auto res_1 = conv_block(scope, prev_layer, name + "_branch2a",
				tf::TensorShape({ 1, 1, in_channel, neck_channel }),
				new_strides, "VALID", false, true);

			auto res_2 = conv_block(scope, res_1, name + "_branch2b",
				tf::TensorShape({ 3, 3, neck_channel, neck_channel }),
				{ 1, 1, 1, 1 }, "SAME", false, true);

			auto res_3 = conv_block(scope, res_2, name + "_branch2c",
				tf::TensorShape({ 1, 1, neck_channel, out_channel }),
				{ 1, 1, 1, 1 }, "VALID", false, false);

			auto trunk_shape = tf::ops::Shape(scope, trunk);
			auto trunk_shape_last_dim = tf::ops::Slice(scope, trunk_shape, { 3 }, { 1 });
			auto res_3_shape = tf::ops::Shape(scope, res_3);
			auto res_3_shape_last_dim = tf::ops::Slice(scope, res_3_shape, { 3 }, { 1 });
			
			auto final_res = tf::ops::Add(scope, trunk, res_3);
			/*
			auto save_op = tf::ops::Save(scope, tf::Input( std::string(".\dump_tensor_res3.txt") )
			, tf::Input({ std::string("res_3_tensor") }), { res_3 });
			*/

			return tf::ops::Relu(scope, final_res);

		}

		/*
		tensorflow::Output TfBlocks::ConcatColumnWithValue(const tensorflow::Scope & scope,
			const tensorflow::Input & tensor, int value, bool concat_right) const
		{
			auto height = GetTensorShapeN(scope, tensor, 0); 
			auto width = tf::ops::Fill(scope, { 1,1 }, value);
			auto new_shape = tf::ops::Concat(scope, { 1,height },1);
			auto new_column = tf::ops::Fill(scope, new_shape, value);
			
			if (concat_right)
				return tf::ops::Concat(scope, { tensor,new_column }, 1);
			else
				return tf::ops::Concat(scope, { new_column,tensor }, 1);
		}

		tensorflow::Output TfBlocks::GetTensorShapeN(const tensorflow::Scope & scope,
			const tensorflow::Input & tensor, int dim) const
		{
			auto shape = tf::ops::Shape(scope, tensor);
			auto dim_value = tf::ops::Slice(scope, tensor, { dim }, { 1 });
			return dim_value;			
		}
		*/

		void TfBlocks::check_status(const tf::Scope& scope) const
		{
			
			if (scope.status().ok())
				return;			
			
			std::cout << scope.status().error_message() << std::endl;			
			std::cout << "Error!";
		}

		TfBlocks::TfBlocks(H5::H5File* hdf_file) : hdf_file_(hdf_file)
		{
		}
}
}
