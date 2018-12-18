#include "ffmpeg.h"
#include <chrono>
namespace videorender
{
	namespace internal
	{
		namespace seeker
		{
			//seek file to percent (0..100%)
			void seek_percentile(AVFormatContext * input_stream, int stream_id, double percent);
			void seek_time(AVFormatContext * input_stream, int stream_id, std::chrono::duration<double> time);

		}
	}
}