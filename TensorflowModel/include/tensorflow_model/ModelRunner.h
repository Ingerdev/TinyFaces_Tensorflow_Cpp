#pragma once
#include "TfH5ModelLoader.h"
#include "videorender/Prediction.h"
#include "ImageParser.h"

namespace tiny_face_model
{
	namespace internal
	{
		using namespace videorender;
		//this class uses tf model and parse predictions.
		class ModelRunner
		{
		public:
			ModelRunner(TfH5ModelLoader & model_loader);
			std::vector<Prediction> generate_predictions(const std::string& image_name);
			std::vector<Prediction> generate_predictions(const cv::Mat& image_data);
			~ModelRunner();

		private:
			std::vector<Prediction> calculate_bounding_boxes(const tf::Tensor& predictions) const;
			TfH5ModelLoader & model_loader_;
			ImageParser image_parser_;

		};
	}
}
