#pragma once
#pragma warning(push, 0) 
#include <string_view>
#include <vector>
#include <list>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma warning(pop) 

#include "SignalTypes.h"

namespace videorender
{	
	
	namespace internal
	{
		class VideoRenderImpl;
	}

	class VideoRender
	{

	public:
		VideoRender();
		void register_image_callback(frame_callback_t frame_callback);		
		void start_render(const std::string_view filename,float file_seek_percentage=0);
		~VideoRender();

	private:
		
		
		//unique_ptr "hides" VideorRenderImpl so we dont need to include its header
		std::unique_ptr<internal::VideoRenderImpl> render_impl_;
	};
}
