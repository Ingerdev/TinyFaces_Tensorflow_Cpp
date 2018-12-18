#pragma once
#pragma warning(push, 0) 
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "tensorflow/core/framework/tensor.h"
#pragma warning(pop) 
#include "Exceptions.h"

namespace tiny_face_model
{
	namespace internal
	{
		namespace tf = tensorflow;
		class ImageParser
		{
		public:
			/*
			image_channel_substract_values - array with 3 values (one per bgr channel)
			whose would be substracted from each image pixel (kinda mean values)

			image_scales - set of pyramid image scales
			*/
			ImageParser(const std::vector<float>& image_channel_substract_values,
				const std::vector<std::vector<double>> clusters);


			cv::Mat parse_image(const std::string& image_path);
			cv::Mat parse_image(const cv::Mat & image_data);
			//convert image to CV_32FC3 and substract channels mean.

			tf::Tensor create_mat_tensor_on_cpu(const cv::Mat & image_data, float scale);
			std::vector<float> calculate_scales(const cv::Mat & image_data);

			~ImageParser() {}
		private:
			ImageParser() = delete;

			cv::Mat normalize_mat(const cv::Mat & image_data);

			//generate _clusters_normal_idx_w/_clusters_normal_idx_h arrays
			//(determine clusters height/width)
			void parse_clusters();

			//generate list of floats between begin to end with step step
			//note that begin always less than end
			static std::vector<float> frange(float begin, float end, float step);

			//image channel means (seems they are deducted from trained data so they arent stricly 128)
			std::vector<float> image_channel_substract_values_;

			//25x5 image areas [[left,top,right,bottom,??(unknown value between 0 and 1.0)]...]
			std::vector<std::vector<double>> clusters_;

			// width and height of image areas
			//std::vector<float> _clusters_normal_idx_w;
			//std::vector<float> _clusters_normal_idx_h;

			//precalculated maximums (python arrays _clusters_normal_xxx can be effectively converted to just two numbers)
			//as only two result numbers are used
			double clusters_w_max_;
			double clusters_h_max_;

			//CV_32FC3
			cv::Mat image_data_;
		};
	}
}