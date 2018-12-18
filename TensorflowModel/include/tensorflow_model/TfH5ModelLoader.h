#pragma once
#pragma warning(push, 0)
#include <string>
#include <memory>
#include <optional>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/cc/client/client_session.h"

#include <H5Cpp.h>
#pragma warning(pop)

#include "tensorflow_model/TfBlocks.h"

namespace tiny_face_model
{
	//!! as of
	//!https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/tutorials/example_trainer.cc
	// method CreateGraphDef
	// we shouldnt bother about tensors/ops destruction, when they are added to a
	// scope, it rule their lifetime.
	class TfH5ModelLoader 
	{
	public:

		//model hyperparameters
		struct Attrs
		{
			Attrs ProbTreshhold(float prob_treshhold)
			{
				Attrs ret = *this;
				ret.prob_threshold_ = prob_treshhold;
				return ret;
			}

			Attrs NMSTreshhold(float nms_treshhold)
			{
				Attrs ret = *this;
				ret.nms_threshold_ = nms_treshhold;
				return ret;
			}

			//The threshold of detection confidence.
			float prob_threshold_ = 0.5f;
			//The overlap threshold of non maximum suppression
			float nms_threshold_ = 0.1f;
		};

		TfH5ModelLoader();
		void compute_ignored_tids();
		void compose_wanted_model_outputs(const internal::TfBlocks& blocks);
		void create_model(
			const std::string& model_filename, Attrs model_parameters = Attrs());

		void start_new_session()
		{
			session_ = std::make_unique<tensorflow::ClientSession>(scope_);
		}
		void stop_delete_session()
		{
			session_.reset();
		}
		//calculate bboxes for image and keep them inside model tensor
		void calculate_bboxes(
			const tensorflow::Tensor& image_tensor, bool scale_small_used,float scale);

		

		//reset stateful ref tensors 
		void reset_model_state();
		//
		tensorflow::Tensor get_filtered_bboxes();
		~TfH5ModelLoader();

		// 2d image area clusters
		const std::vector<std::vector<double> >& get_clusters() const;
		// 1d array[0..2] with b,g,r channels means
		const std::vector<float>& get_channels_mean() const;

		static Attrs ProbTreshhold(float prob_treshhold) {
			return Attrs().ProbTreshhold(prob_treshhold);
		}
		static Attrs NMSTreshhold(float nms_treshhold) {
			return Attrs().NMSTreshhold(nms_treshhold);
		}

	private:
		TfH5ModelLoader(const TfH5ModelLoader&) = delete;
		TfH5ModelLoader operator=(const TfH5ModelLoader&) = delete;

		

		void load_clusters();
		void load_channels_mean();

		void create_placeholders();

		tensorflow::Output tiny_face_model(const tensorflow::Scope& scope,
			const internal::TfBlocks& tf_blocks,
			const tensorflow::ops::Placeholder& image_data);

		// generate face rects based on tiny_faqce_model final output score tensor
		tensorflow::Output calculate_face_bboxes_model(const tensorflow::Scope& scope,
			const internal::TfBlocks& tf_blocks,
			const tensorflow::Input& score_final_tf, 
			const tensorflow::ops::Placeholder& scale_small_presenter,
			const tensorflow::ops::Placeholder& scale_big_presenter,
			const tensorflow::ops::Placeholder& scale);

		tensorflow::Output calculate_bboxes(const tensorflow::Scope& scope,
			const internal::TfBlocks& tf_blocks,
			const tensorflow::Input& cls_scores,
			const tensorflow::Input & score_cls_tf,
			const tensorflow::Input& score_reg_tf,
			const tensorflow::ops::Placeholder& scale);

		//use NonMaxSupression over stacked bboxes
		tensorflow::Output filter_bboxes(const tensorflow::Scope& scope,
			const internal::TfBlocks& tf_blocks,
			const tensorflow::Input stacked_bboxes,
			const tensorflow::Input bbox_count
			);

		//add bbox tensor to bbox_stack variable tensor
		//(we collecting bboxes calculated from different image scales)
		tensorflow::Output stack_bboxes(const tensorflow::Scope& scope,
			const internal::TfBlocks& tf_blocks,
			const tensorflow::Input stacked_bboxes,
			const tensorflow::Input new_bbox
		);

		tensorflow::Output reset_bboxes(const tensorflow::Scope & scope,
			const internal::TfBlocks & tf_blocks,
			const tensorflow::Input stacked_bboxes);

		// filter arrays with 25 1|0 values.
		std::array<float, 25> ignored_tids_big_scales_, ignored_tids_small_scales_;		
		std::unique_ptr<H5::H5File> file_;						
		tensorflow::Scope scope_;
		std::unordered_map<std::string, tensorflow::ops::Placeholder> map_placeholders_;
		std::unordered_map<std::string, tensorflow::Output> map_desired_model_outputs_;

		std::vector<std::vector<double> > clusters_;
		std::vector<float> image_channels_mean_;
		Attrs model_parameters_;
		std::unique_ptr<tensorflow::ClientSession> session_;
		
		int bboxes_tensor_height_;
		tensorflow::Tensor bboxes_;
		std::optional<tensorflow::ops::Variable> bboxes_var_;
	};
}  // namespace tiny_face_model
