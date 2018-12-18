#include "tensorflow_model/TfH5ModelLoader.h"

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
#include "tensorflow_model/TfBlocks.h"

using namespace tiny_face_model::internal;
namespace tiny_face_model
{
	
	namespace internal
	{
		//hdf 5 file records identifiers
		constexpr char* CLUSTERS_DATASET_NAME = "clusters";
		constexpr char* CHANNELS_MEAN_DATASET_NAME = "averages";

		//placeholders map keys
		constexpr char* PLACEHOLDER_IMAGE_KEY = "image_placeholder";
		constexpr char* PLACEHOLDER_SCALE_SMALL_PRESENTER_KEY = "scale_small_placeholder";
		constexpr char* PLACEHOLDER_SCALE_BIG_PRESENTER_KEY = "scale_big_placeholder";
		constexpr char* PLACEHOLDER_SCALE_KEY = "scale";
		constexpr char* PLACEHOLDER_BBOXES_KEY = "bboxes";
		constexpr char* PLACEHOLDER_BBOXES_HEIGHT_KEY = "bboxes_height";

		//outputs map keys
		constexpr char * OUTPUT_SCORES_KEY = "scores_output_node";
		constexpr char * OUTPUT_BBOXES_KEY = "stacked_bboxes_output_node";
		constexpr char * OUTPUT_FILTERED_BBOXES_KEY = "filtered_bboxes_output_node";
		constexpr char * OUTPUT_RESETTED_BBOXES_KEY = "resetted_bboxes_output_node";

	}  // namespace internal

	TfH5ModelLoader::TfH5ModelLoader() :
		file_(nullptr), scope_(tf::Scope::NewRootScope())
	{
		//computes two array (for small and big tids respectively)
		//in the model parsing step they will mark face_classes indexes and reset them to zero
		compute_ignored_tids();

		//create placeholders for model
		create_placeholders();
		
	}
	
	void TfH5ModelLoader::create_placeholders()
	{
		//create image placeholder with shape [1,-1,-1,3]: batch size = 1, width/height are undefined
		//and 3 color channels.		
		map_placeholders_.emplace(internal::PLACEHOLDER_IMAGE_KEY,
			tf::ops::Placeholder(scope_.WithOpName(internal::PLACEHOLDER_IMAGE_KEY),
				tf::DataType::DT_FLOAT, tf::ops::Placeholder::Attrs().Shape({ 1, -1, -1, 3 })));
							

		//create "bool" (DT_FLOAT) placeholder with shape [1] which is used for selecting one or
		//another tf executionbranch 		
		map_placeholders_.emplace(internal::PLACEHOLDER_SCALE_SMALL_PRESENTER_KEY,
			tf::ops::Placeholder(scope_.WithOpName(internal::PLACEHOLDER_SCALE_SMALL_PRESENTER_KEY),
				tf::DataType::DT_FLOAT, tf::ops::Placeholder::Attrs().Shape({ 1 })));

		//create "bool" (DT_FLOAT) placeholder with shape [1] which is used for selecting one or
		//another tf executionbranch 
		map_placeholders_.emplace(internal::PLACEHOLDER_SCALE_BIG_PRESENTER_KEY,
			tf::ops::Placeholder(scope_.WithOpName(internal::PLACEHOLDER_SCALE_BIG_PRESENTER_KEY),
				tf::DataType::DT_FLOAT, tf::ops::Placeholder::Attrs().Shape({ 1 })));

		//create placeholder for scale value
		map_placeholders_.emplace(internal::PLACEHOLDER_SCALE_KEY,
			tf::ops::Placeholder(scope_.WithOpName(internal::PLACEHOLDER_SCALE_KEY),
				tf::DataType::DT_FLOAT, tf::ops::Placeholder::Attrs().Shape({ 1 })));

		//create placeholder for bboxes tensor
		map_placeholders_.emplace(internal::PLACEHOLDER_BBOXES_KEY,
			tf::ops::Placeholder(scope_.WithOpName(internal::PLACEHOLDER_BBOXES_KEY),
				tf::DataType::DT_FLOAT, tf::ops::Placeholder::Attrs().Shape({-1,4})));

		//create placeholder for bboxes tensor height
		//easy way to determine how much values are in the PLACEHOLDER_BBOXES_KEY tensor
		map_placeholders_.emplace(internal::PLACEHOLDER_BBOXES_HEIGHT_KEY,
			tf::ops::Placeholder(scope_.WithOpName(internal::PLACEHOLDER_BBOXES_HEIGHT_KEY),
				tf::DataType::DT_INT32, tf::ops::Placeholder::Attrs().Shape({})));
	}

	void TfH5ModelLoader::compute_ignored_tids()
	{
		//initial tids all 1
		ignored_tids_small_scales_.fill(1.0);
		ignored_tids_big_scales_.fill(1.0);

		//tid indexes which should be zeroed.
		std::array<int, 17> tids_small_scales_indexes = { 0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
		std::array<int, 10> tids_big_scales_indexes = { 0, 1, 2, 3, 12, 13, 14, 15, 16, 17 };

		std::for_each(tids_small_scales_indexes.begin(), tids_small_scales_indexes.end(),
			[&](const auto& index) { ignored_tids_small_scales_[index] = 0.0; });
		

		std::for_each(tids_big_scales_indexes.begin(), tids_big_scales_indexes.end(),
			[&](const auto& index) { ignored_tids_big_scales_[index] = 0.0; });
	}


	//create model and keep important outputs in the map 
	void TfH5ModelLoader::compose_wanted_model_outputs(const internal::TfBlocks& blocks)
	{
		
		map_desired_model_outputs_.emplace(internal::OUTPUT_SCORES_KEY,
			tiny_face_model(scope_, blocks,
				map_placeholders_.at(internal::PLACEHOLDER_IMAGE_KEY)));


		map_desired_model_outputs_.emplace(internal::OUTPUT_BBOXES_KEY,
			calculate_face_bboxes_model(scope_,
				blocks,
				map_desired_model_outputs_.at(internal::OUTPUT_SCORES_KEY),
				map_placeholders_.at(internal::PLACEHOLDER_SCALE_SMALL_PRESENTER_KEY),
				map_placeholders_.at(internal::PLACEHOLDER_SCALE_BIG_PRESENTER_KEY),
				map_placeholders_.at(internal::PLACEHOLDER_SCALE_KEY)));

		map_desired_model_outputs_.emplace(internal::OUTPUT_FILTERED_BBOXES_KEY,
			filter_bboxes(scope_, blocks,
				map_placeholders_.at(internal::PLACEHOLDER_BBOXES_KEY),
				map_placeholders_.at(internal::PLACEHOLDER_BBOXES_HEIGHT_KEY)));
		
		map_desired_model_outputs_.emplace(internal::OUTPUT_RESETTED_BBOXES_KEY,
			reset_bboxes(scope_, blocks,
				map_placeholders_.at(internal::PLACEHOLDER_BBOXES_KEY)));
		/*
		map_desired_model_outputs_.emplace(internal::OUTPUT_RESETTED_BBOXES_KEY,
			reset_bboxes(scope_, blocks,
				map_desired_model_outputs_.at(internal::OUTPUT_BBOXES_KEY)));
				*/

		
	}

	void TfH5ModelLoader::create_model(
		const std::string& model_filename,
		Attrs model_parameters)
	{
		model_parameters_ = model_parameters;
		file_= std::unique_ptr<H5::H5File>(new H5::H5File(model_filename, H5F_ACC_RDONLY));
		TfBlocks blocks = internal::TfBlocks(file_.get());

		load_clusters();
		load_channels_mean();
		compose_wanted_model_outputs(blocks);							
	}

	void TfH5ModelLoader::calculate_bboxes(const tf::Tensor& image_tensor,
		bool scale_small_used,float scale)
	{				
				
		std::cout << "\n" << "Session.Run started" << "\n";
		//inputs
		auto& image_placeholder = map_placeholders_.at(internal::PLACEHOLDER_IMAGE_KEY);		
		auto& scale_small_placeholder = map_placeholders_.at(internal::PLACEHOLDER_SCALE_SMALL_PRESENTER_KEY);
		auto& scale_big_placeholder = map_placeholders_.at(internal::PLACEHOLDER_SCALE_BIG_PRESENTER_KEY);
		auto& scale_placeholder = map_placeholders_.at(internal::PLACEHOLDER_SCALE_KEY);
		
		//outputs
		auto& faces_bboxes_output = map_desired_model_outputs_.at(internal::OUTPUT_BBOXES_KEY);
		//auto& scores_output = map_desired_model_outputs_.at(internal::OUTPUT_SCORES_KEY);
		
		float scale_small_used_value = scale_small_used ? 1.0f : 0.0f;
		float scale_big_used_value = scale_small_used ? 0.0f : 1.0f;
		
		std::vector<tf::Tensor> outputs;
		
		auto status = session_->Run(
			{   //input values for placeholders
				{ image_placeholder,image_tensor},
				{ scale_small_placeholder,scale_small_used_value },
				{ scale_big_placeholder,scale_big_used_value },
				{ scale_placeholder,scale },
			},  //we want value produced by last model node
				{ faces_bboxes_output/*,scores_output*/}, &outputs);
			
		bboxes_tensor_height_ =  static_cast<int>(outputs[0].dim_size(0));
		bboxes_ = outputs[0];		
		std::cout << status.error_message() << std::endl;
		std::cout << "\n" <<"Session.Run ended" << "\n";
		
		return;
	}

	tf::Tensor TfH5ModelLoader::get_filtered_bboxes()
	{
		//inputs
		auto& bboxes_placeholder = map_placeholders_.at(internal::PLACEHOLDER_BBOXES_KEY);
		auto& bboxes_height_placeholder = map_placeholders_.at(internal::PLACEHOLDER_BBOXES_HEIGHT_KEY);

		//outputs
		auto& filtered_bboxes_output = map_desired_model_outputs_.at(internal::OUTPUT_FILTERED_BBOXES_KEY);

		std::vector<tf::Tensor> outputs;

		//tf_helpers::print_tensor(bboxes_);
		auto status = session_->Run(
			{   //input values for placeholders
				{ bboxes_placeholder,bboxes_},
				{ bboxes_height_placeholder,bboxes_tensor_height_},
			},  //we want value produced by last model node
			{ filtered_bboxes_output }, &outputs);
		return outputs[0];
	}

	//clear values from tensor bboxes
	void TfH5ModelLoader::reset_model_state()
	{
		//input
		auto& bboxes_placeholder = map_placeholders_.at(internal::PLACEHOLDER_BBOXES_KEY);	

		//output
		auto& resetted_bboxes_output = map_desired_model_outputs_.at(internal::OUTPUT_RESETTED_BBOXES_KEY);
		
		//we have only one stateful tensor - bboxes_
		std::vector<tf::Tensor> outputs;

		//tf_helpers::print_tensor(bboxes_);
		auto status = session_->Run(
			{   //input values for placeholders
				{ bboxes_placeholder,bboxes_},			
			},  //we want value produced by last model node
			{ resetted_bboxes_output }, &outputs);
		//tf_helpers::print_tensor(outputs[0]);
		return;
	}


#pragma region model

	tf::Output TfH5ModelLoader::tiny_face_model(const tf::Scope & scope,
		const internal::TfBlocks& tf_blocks,
		const tf::ops::Placeholder& image_data)
	{
		auto img_layer = tf::ops::Pad(scope, image_data, { {0, 0},{3, 3},{3, 3},{0, 0} });
		
		//conv = self.conv_block(img, 'conv1', shape = [7, 7, 3, 64], strides = [1, 2, 2, 1], padding = "VALID", add_relu = True)
		auto conv_1 = tf_blocks.conv_block(scope, tf::Input(img_layer), "conv1", tf::TensorShape({ 7,7,3,64 }),
			{ 1,2,2,1 }, "VALID", false, true);		
		auto pool_1 = tf::ops::MaxPool(scope, conv_1, { 1, 3, 3, 1 }, { 1, 2, 2, 1 }, "SAME");
		
		//res2a_branch1 = self.conv_block(pool1, 'res2a_branch1', shape = [1, 1, 64, 256], padding = "VALID", add_relu = False)
		auto res2a_branch1 = tf_blocks.conv_block(scope, pool_1, "res2a_branch1", tf::TensorShape({ 1,1,64,256 }),
			{ 1, 1, 1, 1 }, "VALID", false, false);
		
		//res2a = self.residual_block(pool1, 'res2a', 64, 64, 256, res2a_branch1)
		auto res2a = tf_blocks.residual_block(scope, pool_1, "res2a", 64, 64, 256, res2a_branch1);
		//res2b = self.residual_block(res2a, 'res2b', 256, 64, 256, res2a)
		auto res2b = tf_blocks.residual_block(scope, res2a, "res2b", 256, 64, 256, res2a);
		//res2c = self.residual_block(res2b, 'res2c', 256, 64, 256, res2b)
		auto res2c = tf_blocks.residual_block(scope, res2b, "res2c", 256, 64, 256, res2b);
		
		
		//res3a_branch1 = self.conv_block(res2c, 'res3a_branch1', shape = [1, 1, 256, 512], strides = [1, 2, 2, 1], padding = "VALID", add_relu = False)
		auto res3a_branch1 = tf_blocks.conv_block(scope, res2c, "res3a_branch1", tf::TensorShape({ 1,1,256,512 }),
			{ 1, 2, 2, 1 }, "VALID", false, false);
		
		//res3a = self.residual_block(res2c, 'res3a', 256, 128, 512, res3a_branch1)
		auto res3a = tf_blocks.residual_block(scope, res2c, "res3a", 256, 128, 512, res3a_branch1);
			
		//res3b1 = self.residual_block(res3a, 'res3b1', 512, 128, 512, res3a)
		auto res3b1 = tf_blocks.residual_block(scope, res3a, "res3b1", 512, 128, 512, res3a);
		
		//res3b2 = self.residual_block(res3b1, 'res3b2', 512, 128, 512, res3b1)
		auto res3b2 = tf_blocks.residual_block(scope, res3b1, "res3b2", 512, 128, 512, res3b1);
		//res3b3 = self.residual_block(res3b2, 'res3b3', 512, 128, 512, res3b2)
		auto res3b3 = tf_blocks.residual_block(scope, res3b2, "res3b3", 512, 128, 512, res3b2);

		//res4a_branch1 = self.conv_block(res3b3, 'res4a_branch1', shape = [1, 1, 512, 1024], strides = [1, 2, 2, 1], padding = "VALID", add_relu = False)
		auto res4a_branch1 = tf_blocks.conv_block(scope, res3b3, "res4a_branch1", tf::TensorShape({ 1, 1, 512, 1024 }),
			{ 1, 2, 2, 1 }, "VALID", false, false);
		//res4a = self.residual_block(res3b3, 'res4a', 512, 256, 1024, res4a_branch1)
		auto res4a = tf_blocks.residual_block(scope, res3b3, "res4a", 512, 256, 1024, res4a_branch1);

		auto res4b = res4a;

		//for i in range(1, 23) :
		//	res4b = self.residual_block(res4b, 'res4b' + str(i), 1024, 256, 1024, res4b)
		for (int i = 1; i < 23; i++)
			res4b = tf_blocks.residual_block(scope, res4b, "res4b" + std::to_string(i), 1024, 256, 1024, res4b);

		//score_res4 = self.conv_block(res4b, 'score_res4', shape = [1, 1, 1024, 125], padding = "VALID",
		//	has_bias = True, add_relu = False, add_bn = False)
		auto score_res4 = tf_blocks.conv_block(scope, res4b, "score_res4", tf::TensorShape({ 1, 1, 1024, 125 }),
			{ 1, 1, 1, 1 }, "VALID", true, false, false);
		
		//score4 = self.conv_trans_layer(score_res4, 'score4', shape = [4, 4, 125, 125], strides = [1, 2, 2, 1], padding = "SAME")
		auto score4 = tf_blocks.conv_trans_layer(scope, score_res4, "score4", tf::TensorShape({ 4, 4, 125, 125 }),
			{ 1, 2, 2, 1 }, "SAME");
		
		tf_blocks.check_status(scope);
		//score_res3 = self.conv_block(res3b3, 'score_res3', shape = [1, 1, 512, 125], padding = "VALID",
		//	has_bias = True, add_bn = False, add_relu = False)
		auto score_res3 = tf_blocks.conv_block(scope, res3b3, "score_res3", tf::TensorShape({ 1, 1, 512, 125 }),
			{ 1, 1, 1, 1 }, "VALID", true, false, false);
		
		//bs, height, width = tf.split(tf.shape(score4), num_or_size_splits = 4)[0:3]
		auto list = tf::ops::Split(scope, 0, tf::ops::Shape(scope, score4), 4); //1
		auto bs = list[0];
		auto height = list[1];
		auto width = list[2];
		
		//_size = tf.convert_to_tensor([height[0], width[0]])
		auto height_0 = tf::ops::Slice(scope, height, { 0 }, { 1 });
		
		auto width_0 = tf::ops::Slice(scope, width, { 0 }, { 1 });
		
		//1-d tensors so 0-axis should be used
		auto size = tf::ops::Concat(scope, { tf::Input(height_0),tf::Input(width_0) }, 0 );
		
		//_offsets = tf.zeros([bs[0], 2])
		auto offsets_0 = tf::ops::Slice(scope, bs, { 0 },{1});
		
		//1-d tensors so 0-axis should be used
		//notice {2}, instead of 2 -> {} mean array so its shape matches shape of offset_0
		auto offset_dims = tf::ops::Concat(scope, { tf::Input(offsets_0),tf::Input({2}) }, 0);
		
		//type of tensor is inferred from fill value. 
		auto offsets = tf::ops::Fill(scope, offset_dims, 0.0f);
		
		//score_res3c = tf.image.extract_glimpse(score_res3, _size, _offsets, centered = True, normalized = False, name = 'score_res3c')
		//todo: add name to scope?
		auto score_res3c = tf::ops::ExtractGlimpse(scope, score_res3, size, offsets, tf::ops::ExtractGlimpse::Attrs().Normalized(false));
		auto score_final = tf::ops::Add(scope, score4, score_res3c);
	
		return score_final;
	}

	tf::Output TfH5ModelLoader::calculate_face_bboxes_model(const tf::Scope & scope,
		const internal::TfBlocks& tf_blocks,
		const tf::Input& score_final_tf,
		const tensorflow::ops::Placeholder& scale_small_presenter,
		const tensorflow::ops::Placeholder& scale_big_presenter,
		const tensorflow::ops::Placeholder& scale
		)
	{		
		// collect scores
		// score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
		auto score_cls_tf = tf::ops::Slice(scope, score_final_tf, { 0,0,0,0 }, { -1,-1,-1,25 });
		auto score_reg_tf = tf::ops::Slice(scope, score_final_tf, { 0,0,0,25 }, { -1,-1,-1,125-25 });

		//std::array<int> mask_big = { 0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
		//prob_cls_tf = expit(score_cls_tf)
		auto prob_cls_tf = tf::ops::Sigmoid(scope, score_cls_tf);
		auto prob_cls_tf_shape = tf::ops::Shape(scope, prob_cls_tf);
		/*******************************************************************************************/
		
		//create small_ignored_tids tensor and broadcast it to prob_cls_tf shape
		auto small_scales_filter_tensor = tf_helpers::tf_stl_helper<tf::DataType::DT_FLOAT>
			::create_load_tensor({ 25 }, ignored_tids_small_scales_);
		tf_blocks.check_status(scope);
		auto bc_small_scales_tensor = tf::ops::BroadcastTo(scope, small_scales_filter_tensor, prob_cls_tf_shape);

		//create big_ignored_tids tensor and broadcast it to prob_cls_tf shape
		auto big_scales_filter_tensor = tf_helpers::tf_stl_helper<tf::DataType::DT_FLOAT>
			::create_load_tensor({ 25 }, ignored_tids_big_scales_);

		auto bc_big_scales_tensor = tf::ops::BroadcastTo(scope, big_scales_filter_tensor, prob_cls_tf_shape);

		//self-made refselect. need two signal inputs, scale_small and scale_big (placeholders)
		// 1) scales -> two tensors, broadcasted to prob_clf shape. 
		// 2) mul bc_*_scaled_tensors with respective scales tensors tensors.
		// 3) add one tensor to another (as one of them will have zeroes, it doesnt change anything)

		//dont need broadcast, multiply already support it.
		//auto bc_small_scale_bool_tensor = tf::ops::BroadcastTo(scope, scale_small_presenter, prob_cls_tf_shape);
		//auto bc_big_scale_bool_tensor = tf::ops::BroadcastTo(scope, scale_big_presenter, prob_cls_tf_shape);
		
		auto small_filtered_mul = tf::ops::Multiply(scope, bc_small_scales_tensor,scale_small_presenter);		
		auto big_filtered_mul = tf::ops::Multiply(scope, bc_big_scales_tensor, scale_big_presenter);
		
		//here we have two tensors one of which guarantee is full of zeros depend on scale mode.
		//so we can safely add them one with another and get "select" behavior without select
		auto selected_scale_tensor = tf::ops::Add(scope, small_filtered_mul, big_filtered_mul);

		//prob_cls_tf[0, :, : , ignoredTids] = 0.0
		auto filtered_prob_cls_tf = tf::ops::Multiply(scope, prob_cls_tf, selected_scale_tensor);				
		
		auto boxes = calculate_bboxes(scope, tf_blocks, filtered_prob_cls_tf,
			score_cls_tf, score_reg_tf,scale);
				
		return boxes;
	}

	tensorflow::Output TfH5ModelLoader::calculate_bboxes(
		const tensorflow::Scope & scope,
		const internal::TfBlocks & tf_blocks,
		const tensorflow::Input & cls_scores,
		const tensorflow::Input & score_cls_tf,
		const tensorflow::Input & score_reg_tf,
		const tensorflow::ops::Placeholder& scale)
	{
		//calculating bounding boxes
		//# threshold for detection
		//	_, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)
		auto prob_cls_greater_predicats = tf::ops::Greater(scope, cls_scores,
			{ model_parameters_.prob_threshold_ });
		tf_blocks.check_status(scope);

		//tensor with coordinates of true values
		auto prob_cls_indices = tf::ops::Cast(scope,
			tf::ops::Where(scope, prob_cls_greater_predicats), tf::DataType::DT_INT32);


		auto prob_cls_indices_slices = tf::ops::Split(scope, 1, prob_cls_indices, 4);
		auto not_used_slice = prob_cls_indices_slices[0];
		auto fy = prob_cls_indices_slices[1];

		auto fx = prob_cls_indices_slices[2];
		auto fc = prob_cls_indices_slices[3];


		/*# interpret heatmap into bounding boxes
			cy = fy * 8 - 1
			cx = fx * 8 - 1
		*/
		auto cy = tf::ops::Cast(scope, tf::ops::Sub(scope, tf::ops::Mul(scope, fy, 8), 1), tf::DT_FLOAT);
		auto cx = tf::ops::Cast(scope, tf::ops::Sub(scope, tf::ops::Mul(scope, fx, 8), 1), tf::DT_FLOAT);


		//add clusters to tf graph as constant tensor
		auto double_clusters_tensor = tf_blocks.tensor_on_cpu<tf::DataType::DT_DOUBLE>(CLUSTERS_DATASET_NAME);
		auto clusters_tensor = tf::ops::Cast(scope, double_clusters_tensor, tf::DataType::DT_FLOAT);

		//create tensor with dimensions of fc and init it with all zeroes
		auto fc_0 = tf::ops::Fill(scope, tf::ops::Shape(scope, fc), 0);

		//create tensor with dimensions of fc and init it with all ones
		auto fc_1 = tf::ops::Fill(scope, tf::ops::Shape(scope, fc), 1);

		tf_blocks.check_status(scope);
		//create tensor with dimensions of fc and init it with all ones
		auto fc_2 = tf::ops::Fill(scope, tf::ops::Shape(scope, fc), 2);

		//create tensor with dimensions of fc and init it with all threes
		auto fc_3 = tf::ops::Fill(scope, tf::ops::Shape(scope, fc), 3);
		tf_blocks.check_status(scope);

		auto fc_0_stacked = tf::ops::Concat(scope, { tf::Input(fc),fc_0 }, 1);
		auto clusters_fc_0 = tf::ops::GatherNd(scope, clusters_tensor,
			tf::ops::Cast(scope, fc_0_stacked, tf::DataType::DT_INT32));//);		


		auto fc_1_stacked = tf::ops::Concat(scope, { tf::Input(fc),fc_1 }, 1);
		auto clusters_fc_1 = tf::ops::GatherNd(scope, clusters_tensor, fc_1_stacked);


		auto fc_2_stacked = tf::ops::Concat(scope, { tf::Input(fc),fc_2 }, 1);
		auto clusters_fc_2 = tf::ops::GatherNd(scope, clusters_tensor, fc_2_stacked);

		auto fc_3_stacked = tf::ops::Concat(scope, { tf::Input(fc),fc_3 }, 1);
		auto clusters_fc_3 = tf::ops::GatherNd(scope, clusters_tensor, fc_3_stacked);

		auto ch = tf::ops::Add(scope, tf::ops::Sub(scope, clusters_fc_3, clusters_fc_1), 1.0f);
		auto cw = tf::ops::Add(scope, tf::ops::Sub(scope, clusters_fc_2, clusters_fc_0), 1.0f);


		auto clusters_shape = tf::ops::Shape(scope, clusters_tensor);
		auto nt = tf::ops::GatherNd(scope, clusters_shape, { {0} });



		/*	# extract bounding box refinement
				Nt = clusters.shape[0]
				tx = score_reg_tf[0, :, : , 0 : Nt]
				ty = score_reg_tf[0, :, : , Nt : 2 * Nt]
				tw = score_reg_tf[0, :, : , 2 * Nt : 3 * Nt]
				th = score_reg_tf[0, :, : , 3 * Nt : 4 * Nt]*/

				//tx = score_reg_tf[0, :, : , 0 : Nt]

		auto tx = tf::ops::Slice(scope, score_reg_tf,
			tf::Input({ 0,0,0,0 }),
			tf::ops::Concat(scope, { tf::Input({ 1,-1,-1}) ,nt }, 0));

		//slice length is the same for every tX and equal to nt
		auto t_slices_length = tf::ops::Concat(scope, { { 1,-1,-1 }, nt }, 0);

		//ty = score_reg_tf[0, :, : , Nt : 2 * Nt]
		auto ty_start_indexes = tf::ops::Concat(scope, { tf::Input({ 0,0,0 }), nt }, 0);
		auto ty = tf::ops::Slice(scope, score_reg_tf, ty_start_indexes, t_slices_length);

		//tw = score_reg_tf[0, :, : , 2 * Nt : 3 * Nt]
		auto tw_start_indexes = tf::ops::Concat(scope, { tf::Input({ 0,0,0 }), tf::ops::Multiply(scope, nt, { 2 }) }, 0);
		auto tw = tf::ops::Slice(scope, score_reg_tf, tw_start_indexes, t_slices_length);

		//th = score_reg_tf[0, :, : , 3 * Nt : 4 * Nt]
		auto th_start_indexes = tf::ops::Concat(scope, { tf::Input({ 0,0,0 }), tf::ops::Multiply(scope, nt, { 3 }) }, 0);
		auto th = tf::ops::Slice(scope, score_reg_tf, th_start_indexes, t_slices_length);



		/*
		# refine bounding boxes
		dcx = cw * tx[fy, fx, fc]
		dcy = ch * ty[fy, fx, fc]
		rcx = cx + dcx
		rcy = cy + dcy
		rcw = cw * np.exp(tw[fy, fx, fc])
		rch = ch * np.exp(th[fy, fx, fc])
		*/

		//0_fy_fx_fc indices
		auto zero_fy_fx_fc_indices = tf::ops::Concat(scope,
			{ tf::Input(tf::ops::Concat(scope,
			{ tf::Input(tf::ops::Concat(scope, {tf::Input(fc_0), fy},1)),
			  fx} , 1)),
			  fc }, 1);

		//dcx = cw * tx[fy, fx, fc]
		auto tx_gathered = tf::ops::GatherNd(scope, tf::Input(tx), zero_fy_fx_fc_indices);
		auto dcx = tf::ops::Multiply(scope, tx_gathered, cw);

		//dcy = ch * ty[fy, fx, fc]
		auto ty_gathered = tf::ops::GatherNd(scope, tf::Input(ty), zero_fy_fx_fc_indices);
		auto dcy = tf::ops::Multiply(scope, ch, ty_gathered);

		//(x,1) ->(x). necessary to prevent 1d->2d broadcasting in ops below.
		auto cx_reshaped = tf::ops::Reshape(scope, cx, { -1 });
		auto cy_reshaped = tf::ops::Reshape(scope, cy, { -1 });

		//rcx = cx + dcx
		auto rcx = tf::ops::Add(scope, cx_reshaped, dcx);//dcx //float

		//rcy = cy + dcy
		auto rcy = tf::ops::Add(scope, cy_reshaped, dcy); //float

		//rcw = cw * np.exp(tw[fy, fx, fc])
		auto tw_gathered = tf::ops::GatherNd(scope, tf::Input(tw), zero_fy_fx_fc_indices);
		auto tw_gathered_exp = tf::ops::Exp(scope, tw_gathered);
		auto rcw = tf::ops::Multiply(scope, cw, tw_gathered_exp);

		//rch = ch * np.exp(th[fy, fx, fc])
		auto th_gathered = tf::ops::GatherNd(scope, tf::Input(th), zero_fy_fx_fc_indices);
		auto th_gathered_exp = tf::ops::Exp(scope, th_gathered);
		auto rch = tf::ops::Multiply(scope, ch, th_gathered_exp);

		//scores = score_cls_tf[0, fy, fx, fc]
		auto scores = tf::ops::GatherNd(scope, score_cls_tf, zero_fy_fx_fc_indices);

		//tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
		auto rcw_div_2 = tf::ops::Div(scope, rcw, 2.0f);
		auto rch_div_2 = tf::ops::Div(scope, rch, 2.0f);

		auto left_w = tf::ops::Sub(scope, rcx, rcw_div_2);

		auto top_h = tf::ops::Sub(scope, rcy, rch_div_2);
		auto right_w = tf::ops::Add(scope, rcx, rcw_div_2);
		auto bottom_h = tf::ops::Add(scope, rcy, rch_div_2);
		auto bboxes = tf::ops::Stack(scope, { tf::Input(left_w),top_h,right_w,bottom_h });

		//tmp_bboxes = np.vstack((tmp_bboxes / s, scores))			
		//reshape scores [-1] -> [1,-1] and concat to the bottom
		auto bboxes_c = tf::ops::Concat(scope, { tf::Input(tf::ops::Div(scope,bboxes,scale)),
			tf::ops::Reshape(scope,scores,{1,-1}) }, 0);

		auto bboxes_transposed = tf::ops::Transpose(scope, bboxes_c, { 1,0 });

		//create state variables
		//variable for keeping stacked bboxes from different scale model runs
		auto bboxes_stack = tf::ops::Variable(scope, { 0,5 }, tf::DataType::DT_FLOAT);
		bboxes_var_ = bboxes_stack;

		tf_blocks.check_status(scope);
		return stack_bboxes(scope, tf_blocks, bboxes_stack, bboxes_transposed);
	
	}

	//add single bbox_tensor to stack of bboxes (each image scale has separate bbox tensor)
	tensorflow::Output TfH5ModelLoader::stack_bboxes(const tensorflow::Scope & scope,
		const internal::TfBlocks& tf_blocks,
		const tensorflow::Input stacked_bboxes,
		const tensorflow::Input new_bbox)
	{
		auto stack = tf::ops::Concat(scope, { stacked_bboxes,new_bbox }, 0);
		tf_blocks.check_status(scope);
		//update stacked_bboxes variable

		auto updated_stacked_bboxes = tf::ops::Assign(scope,
			stacked_bboxes,
			stack, tf::ops::Assign::Attrs().ValidateShape(false));

		return updated_stacked_bboxes;
	}

	tensorflow::Output TfH5ModelLoader::filter_bboxes(const tensorflow::Scope& scope,
		const internal::TfBlocks& tf_blocks,
		const tensorflow::Input stacked_bboxes,
		const tensorflow::Input bbox_count)
	{		

		auto bboxes_0_3 = tf::ops::Slice(scope, stacked_bboxes, { 0,0 }, { -1,4 });
		auto bboxes_4 = tf::ops::Slice(scope, stacked_bboxes, { 0,4 }, { -1,1 });
		auto flattened_bboxes_4 = tf::ops::Reshape(scope, bboxes_4, { -1 });

		//determine whose bbox tensor rows contain faces
		auto bboxes_indexes = tf::ops::NonMaxSuppression(scope, 
			bboxes_0_3,
			flattened_bboxes_4,
			tf::Input(bbox_count),
			tf::ops::NonMaxSuppression::Attrs().IouThreshold(model_parameters_.nms_threshold_));
		tf_blocks.check_status(scope);
		
		//1d array -> 2d array with single column. need for GatherNd
		auto transposed_indices = tf::ops::Reshape(scope, bboxes_indexes, { -1,1 });
				
		auto indexed_bboxes = tf::ops::GatherNd(scope, stacked_bboxes,
			transposed_indices);
		tf_blocks.check_status(scope);

		return indexed_bboxes;
	}
	

	//reset bboxes rows to zero
	tensorflow::Output TfH5ModelLoader::reset_bboxes(const tensorflow::Scope & scope,
		const internal::TfBlocks& blocks,
		const tensorflow::Input )
	{
		tf::ops::Variable stacked_bboxes = bboxes_var_.value();
		
		//create empty variable
		auto clean_bboxes = tf::ops::Variable(scope, { 0,5 }, tf::DataType::DT_FLOAT);
		//reset stacked_bboxes variable by assigning empty variable to it.
		return tf::ops::Assign(scope,
			stacked_bboxes,
			clean_bboxes, tf::ops::Assign::Attrs().ValidateShape(false));
	}
#pragma endregion
	void TfH5ModelLoader::load_clusters()
	{		
		internal::HDF5DatasetWrapper dataset_wrapper(file_.get(), internal::CLUSTERS_DATASET_NAME);		
		clusters_ = tf_helpers::vector_hdf_helper<double>::load_small_2d_array(dataset_wrapper.get_dims(), dataset_wrapper.get_dataset());				
	}

	void TfH5ModelLoader::load_channels_mean()
	{				
		internal::HDF5DatasetWrapper dataset_wrapper(file_.get(), internal::CHANNELS_MEAN_DATASET_NAME);
		assert(dataset_wrapper.get_dims().size() == 1);		
		image_channels_mean_ = tf_helpers::vector_hdf_helper<float>::load_1d_vector(dataset_wrapper.get_dims()[0], dataset_wrapper.get_dataset());
	}
	

	TfH5ModelLoader::~TfH5ModelLoader()
	{
		if (file_ != nullptr)
			file_->close();
	}
	const std::vector<std::vector<double>>& TfH5ModelLoader::get_clusters() const
	{
		return clusters_;
	}
	const std::vector<float>& TfH5ModelLoader::get_channels_mean() const
	{
		return image_channels_mean_;
	}
}
/*
prob_cls_tf.shape : [1,33,45,25]
*/