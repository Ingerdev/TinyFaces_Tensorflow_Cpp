#include "tensorflow_model/ImageParser.h"
#pragma warning(push, 0) 
#include <math.h>
#include <algorithm>
#pragma warning(pop) 
namespace tiny_face_model
{
	namespace internal
	{
		ImageParser::ImageParser(const std::vector<float>& image_channel_substract_values, const std::vector<std::vector<double>> clusters) :
			image_channel_substract_values_(image_channel_substract_values),
			clusters_(clusters)
		{
			parse_clusters();
		}

		cv::Mat ImageParser::parse_image(const std::string & image_path)
		{
			//load file as cv::Mat and detect faces on cv::mat
			auto image_data = cv::imread(image_path.c_str(), cv::IMREAD_COLOR); // Read the file
			if (image_data.empty())                      // Check for invalid input
			{
				throw image_load_exception("Could not open or find the image", image_path);
			}

			return parse_image(image_data);

		}
		cv::Mat ImageParser::parse_image(const cv::Mat & image_data)
		{
			//convert image to uniform format (CV_32FC3) and substract means from bgr channels.
			auto mat = normalize_mat(image_data);
			return mat;
		}

		std::vector<float> ImageParser::calculate_scales(const cv::Mat & image_data)
		{
			float width = static_cast<float>(image_data.size().width);
			float height = static_cast<float>(image_data.size().height);


			auto max_w = clusters_w_max_ / width;
			auto max_h = clusters_h_max_ / height;

			//std::cout << max_w << " " << max_h << std::endl;
			float min_scale = static_cast<float>(std::min(
				std::floor(std::log2(max_w)),
				std::floor(std::log2(max_h))));

			//maximum dim length of image
			const int MAX_INPUT_DIM = 5000;

			//max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
			float max_scale = std::min(1.0f, -std::log2(std::max(width, height) / MAX_INPUT_DIM));

			//std::cout << min_scale << " " << max_scale << std::endl;
			std::vector<float> scales_down = frange(min_scale, 0, 1);
			std::vector<float> scales_up = frange(0.5, max_scale, 0.5);

			//copy scales arrays to single array
			std::vector<float> scales_pow;
			copy(scales_down.begin(), scales_down.end(), back_inserter(scales_pow));
			copy(scales_up.begin(), scales_up.end(), back_inserter(scales_pow));

			//calculate 2^scale for each element in scales_pow array
			std::for_each(scales_pow.begin(), scales_pow.end(), [](float &n) { n = static_cast<float>(std::pow(2, n)); });
			//std::for_each(scales_pow.begin(), scales_pow.end(), [](float &n) { std::cout << n << " "; });
			return scales_pow;
		}

		//change image data format to CV_32FC3 (32bit float with 3 channels)
		//and substract channel-wise means
		cv::Mat ImageParser::normalize_mat(const cv::Mat & image_data)
		{
			cv::Mat converted_data = image_data;
			//change image format to tensorflow-compatible
			if (image_data.type() != CV_32FC3)
				image_data.convertTo(converted_data, CV_32FC3);

			//convert int-based pixel channel to float and substract channels mean
			// 0..255 -> -~128.0..~128.0

			cv::Scalar channels_almost_mean = cv::Scalar(
				image_channel_substract_values_[0],
				image_channel_substract_values_[1],
				image_channel_substract_values_[2]);


			converted_data = converted_data - channels_almost_mean;
			return converted_data;
		}

		//assuming image_data is already normalized
		tf::Tensor ImageParser::create_mat_tensor_on_cpu(const cv::Mat & image_data, float scale)
		{
			int resized_width = (int)(std::ceil(image_data.size().width*scale));
			int resized_height = (int)(std::ceil(image_data.size().height*scale));

			tf::Tensor input_tensor(tensorflow::DT_FLOAT,
				tf::TensorShape({ 1,resized_height,resized_width,3 }));

			// get pointer to memory for that Tensor
			float *p = input_tensor.flat<float>().data();

			//note that allocated memory is still owned by input_tensor 
			//so resized_image destructor wont free "p"'s memory.
			cv::Mat resized_image(resized_height, resized_width, CV_32FC3, p);

			//resize image.
			//cv::resize(image_data, resized_image, cv::Size(), scale, scale, cv::INTER_LINEAR);

			//this type of resize call uses preallocated dst memory
			cv::resize(image_data, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);

			return input_tensor;
		}

		void ImageParser::parse_clusters()
		{
			//not used anymore after using modern aggregates "clusters_(w|h)_max"
			/*
			_clusters_normal_idx_w.clear();
			_clusters_normal_idx_h.clear();
			//clusters has shape [25,5] ,
			//each cluster has shape [5]
			//first 4 floats are [left,top,right,bottom]
			for (auto& cluster : _clusters)
			{
				_clusters_normal_idx_w.push_back(cluster[2] - cluster[0] + 1);
				_clusters_normal_idx_h.push_back(cluster[3] - cluster[1] + 1);
			}
			*/
			//getting max of width and height of clusters
			//where 5th element of cluster array equal 1.0
			auto w_unset = true;
			auto h_unset = true;
			for (auto& cluster : clusters_)
			{
				if (cluster[4] != 1.0)
					continue;

				auto w_cluster = cluster[2] - cluster[0] + 1;
				auto h_cluster = cluster[3] - cluster[1] + 1;
				//init w_max with first matched value
				if (w_unset)
				{
					w_unset = false;
					clusters_w_max_ = w_cluster;
				}

				//init h_max with first matched value
				if (h_unset)
				{
					h_unset = false;
					clusters_h_max_ = h_cluster;
				}

				//check for new w_max
				if (w_cluster > clusters_w_max_)
					clusters_w_max_ = w_cluster;

				//check for new h_max
				if (h_cluster > clusters_h_max_)
					clusters_h_max_ = h_cluster;

			}
			//todo:: raise exception if w_unset or h_unset are true
		}

		std::vector<float> ImageParser::frange(float begin, float end, float step)
		{
			std::vector<float> float_range;

			int max_count = int(std::ceil((end - begin) / step));
			float current = begin;

			for (int k = 0; k <= max_count; k++)
			{
				float_range.push_back(current);
				current += step;
			}

			return float_range;
		}
	}
}