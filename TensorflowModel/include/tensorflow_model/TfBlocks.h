#pragma once
#pragma warning(push, 0)
#include <string>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include <H5Cpp.h>
#pragma warning(pop)

namespace tiny_face_model
{
	namespace internal
	{
		namespace tf = tensorflow;

		//tf elementary blocks creation code sit here
		//we need class because methods are dependant on hdf file.
		//so we bind hdf handle to this file and then 
		//use its methods in almost-static style
		class TfBlocks
		{
		public:
			explicit TfBlocks(H5::H5File* hdf_file);
			
			template <tf::DataType Dtype>
			tf::Tensor tensor_on_cpu(const std::string & name) const
			{
				internal::HDF5DatasetWrapper dataset_wrapper(hdf_file_, name);
				auto tensor = tf_helpers::tf_hdf_helper<Dtype>::create_load_tensor(
					tf_helpers::convert_vec_to_tf_int64(dataset_wrapper.get_dims()), dataset_wrapper.get_dataset());

				return 	tensor;
			}


			tensorflow::Output float_variable_on_cpu(const tensorflow::Scope& scope, const std::string& name,
				const std::string& storage_postfix,
				const std::string& variable_postfix,
				const tensorflow::TensorShape& shape,
				const tensorflow::DataType dtype) const;

			tensorflow::Output weight_variable_on_cpu(
				const tensorflow::Scope& scope, const std::string& name,
				const tensorflow::TensorShape& shape,
				const tensorflow::DataType dtype = tensorflow::DataType::DT_FLOAT) const;
			tensorflow::Output bias_variable_on_cpu(
				const tensorflow::Scope& scope, const std::string& name,
				const tensorflow::TensorShape& shape,
				const tensorflow::DataType dtype = tensorflow::DataType::DT_FLOAT) const;
			std::vector<tensorflow::Output> bn_variable_on_cpu(
				const tensorflow::Scope& scope, const std::string& name,
				const tensorflow::TensorShape& shape,
				const tensorflow::DataType dtype = tensorflow::DataType::DT_FLOAT) const;
			tensorflow::Output conv_block(const tensorflow::Scope& scope, const tensorflow::Input& prev_layer,
				const std::string& name, const tensorflow::TensorShape& shape,
				const std::array<int, 4> strides = { 1, 1, 1, 1 },
				const std::string& padding = "SAME",
				const bool has_bias = false, const bool add_relu = true,
				const bool add_bn = true, const float eps = 1.0e-5) const;
			tensorflow::Output conv_trans_layer(const tensorflow::Scope& scope,
				const tensorflow::Input& prev_layer,
				const std::string& name,
				const tensorflow::TensorShape& shape,
				const std::array<int, 4> strides = { 1, 1, 1, 1 },
				const std::string& padding = "SAME",
				const bool has_bias = false) const;

			tensorflow::Output residual_block(const tensorflow::Scope& scope,
				const tensorflow::Input& prev_layer,
				const std::string& name, const int in_channel,
				const int neck_channel, const int out_channel,
				const tensorflow::Input& trunk) const;

			/* not tested
			//create single column ad concat with original_tensor
			tensorflow::Output ConcatColumnWithValue(const tensorflow::Scope& scope,
				const tensorflow::Input& tensor,
				int value,bool concat_right) const;

			//get N dimension of tensor shape
			tensorflow::Output GetTensorShapeN(const tensorflow::Scope& scope,
				const tensorflow::Input& tensor,
				int dim) const;
				*/
			// check tensorflow scope status
			void check_status(const tensorflow::Scope& scope) const;
		private:
			//pointer to opened HDF file without ownership
			H5::H5File * const hdf_file_;
		};
	}
}
