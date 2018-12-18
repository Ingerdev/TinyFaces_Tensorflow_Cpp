#pragma warning(push, 0) 
#pragma warning(pop) 
#include "videorender/VideoRender.h"
#include "videorender/VideoRenderImpl.h"
namespace videorender
{
	VideoRender::VideoRender():
		render_impl_(std::make_unique<internal::VideoRenderImpl>())
	{
		//(void)filename;
	}

	void VideoRender::start_render(const std::string_view filename,float file_seek_percentage)
	{
		render_impl_->start_render(filename, file_seek_percentage);
	}

	void VideoRender::register_image_callback(frame_callback_t frame_callback)
	{
		render_impl_.get()->register_image_callback(frame_callback);
	}
	
	VideoRender::~VideoRender()
	{		
	}
	
}