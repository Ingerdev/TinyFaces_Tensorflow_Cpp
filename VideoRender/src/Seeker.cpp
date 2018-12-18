#include "videorender/Seeker.h"
extern "C"
{
#include "libavutil/avutil.h"
}
#include <exception>
#include <cassert>
#include <iostream>
namespace videorender
{
	namespace internal
	{
		namespace seeker
		{
			void seek_percentile(AVFormatContext * input_stream, int stream_id, double percent)
			{
				assert(percent >= 0.0 && percent <= 1.0);				
				
				auto result = av_seek_frame(input_stream, stream_id, 
					input_stream->duration * percent / (AV_TIME_BASE  * av_q2d(input_stream->streams[stream_id]->time_base)),
					AVSEEK_FLAG_ANY);				
			}

			void seek_time(AVFormatContext * input_stream, int stream_id, std::chrono::duration<double> time)
			{
				throw std::exception("Not implemented");
			}
		}
	}
}