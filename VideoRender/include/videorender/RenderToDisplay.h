#pragma once
#pragma warning(push, 0) 
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma warning(pop) 

#include "Prediction.h"
namespace videorender
{
	class RenderToDisplay
	{
	public:
		RenderToDisplay();
		void process_faces(const std::vector<Prediction>& faces);
		void process_frame(const cv::Mat& frame_image) const;
	private:
		void draw_face_rect(cv::Mat& frame_image, const Prediction& prediction) const;

		std::vector<Prediction> predictions_;
		const std::string window_name_ = "render_window";
	};
}

