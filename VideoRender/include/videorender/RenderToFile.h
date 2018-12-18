#pragma once
#pragma warning(push, 0) 
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma warning(pop) 

namespace videorender
{
	namespace internal
	{
		class RenderToFile
		{
		public:
			RenderToFile();
			~RenderToFile();

			void process_image(cv::Mat& image);
		};
	}
}
