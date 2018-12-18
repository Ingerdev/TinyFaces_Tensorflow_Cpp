#include "tensorflow_model/ModelRunner.h"
#include "tensorflow_model/Exceptions.h"
#pragma warning(push, 0) 
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma warning(pop) 
#include "tensorflow_model/tf_helpers.h"
#include "tensorflow_model/TensorPredictionHelper.h"

namespace tiny_face_model
{
	namespace internal
	{
		ModelRunner::ModelRunner(TfH5ModelLoader & model_loader) :model_loader_(model_loader),
			image_parser_(ImageParser(model_loader_.get_channels_mean(), model_loader_.get_clusters()))
		{
			//todo: rework session work to separate object with RAII semantic
			model_loader_.start_new_session();
		}

		std::vector<Prediction> ModelRunner::generate_predictions(const std::string & image_name)
		{
			//load file as cv::Mat and detect faces on cv::mat
			auto image_data = cv::imread(image_name.c_str(), cv::IMREAD_COLOR); // Read the file
			if (image_data.empty())                      // Check for invalid input
			{
				throw image_load_exception("Could not open or find the image", image_name);
			}

			return generate_predictions(image_data);
		}

		std::vector<Prediction> ModelRunner::generate_predictions(const cv::Mat & image_data)
		{
			//calculate desired scales for this image
			auto scales = image_parser_.calculate_scales(image_data);

			//convert image to proper format and substract channels means
			auto preprocessed_image = image_parser_.parse_image(image_data);
			
			
			for (auto& scale : scales)
			{				
				auto image_filled_tensor = image_parser_.create_mat_tensor_on_cpu(preprocessed_image, scale);
				//model face detection part requires knowledge of image rescaling value.
				//two modes, small and big.
				bool scale_small_used = (scale <= 1.0);
				//feed resized image to the model
				model_loader_.calculate_bboxes(
					image_filled_tensor,
					scale_small_used,
					scale );			
			}


			const auto filtered_bboxes = model_loader_.get_filtered_bboxes();
			
			//reset model state (while calculating bboxes from set of scaled images model accummulate
			//some values). We need to reset them after all scales was processed.			
			
			//tf_helpers::print_tensor(filtered_bboxes);

			
			//ostream-like data hijacker.
			TensorPredictionHelper<double> print_helper;
									
			//dump tensor trough "print" procedure using operator<<
			tf_helpers::internal::print_tensor_impl_dynamic(print_helper,filtered_bboxes);
			
			auto predictions = print_helper.get_predictions();
			
			model_loader_.reset_model_state();
			return predictions;
		}


		ModelRunner::~ModelRunner()
		{
			model_loader_.stop_delete_session();
		}
		
	}
}
