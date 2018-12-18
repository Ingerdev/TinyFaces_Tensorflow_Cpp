#include "videorender/Delayer.h"
extern "C"
{
#include <libavutil/rational.h>
#include <libavutil/avutil.h>
}
#include <iostream>
#include <limits>
namespace videorender
{
	namespace internal
	{
		delayer::delayer(AVRational stream_time_base) :
			stream_time_base_(stream_time_base),
			start_time_(av_gettime_relative()),
			start_xts_(0)
		{					
		}

		void delayer::delay_frame(AVPacket* packet)
		{		
			static const int64_t seek_threshold_sec = 2; //seconds
			static const double seek_threshold_timebase_units = seek_threshold_sec /av_q2d(stream_time_base_);
		
			//get current packet's pts or dts(if pts is AV_NOPTS_VALUE)
			int64_t current_xts = (packet->pts != AV_NOPTS_VALUE) ? packet->pts : packet->dts;
			
			//update start values when difference between current and start xts is more than 2 seconds
			//so it can be seek operation done.
			if (current_xts - start_xts_ > seek_threshold_timebase_units)
			{
				start_xts_ = current_xts;
				start_time_ = av_gettime_relative();
				return;
			}

			//difference of current/first frames in microseconds
			auto diff_frame_time_mcs = convert_timebased_xts_to_microseconds(current_xts - start_xts_);						
			//get clock current/first time difference in microseconds
			auto diff_time_mcs = (av_gettime_relative() - start_time_);
		
			if (diff_frame_time_mcs < diff_time_mcs)
				return;

			av_usleep(static_cast<unsigned int>(diff_frame_time_mcs - diff_time_mcs));		
		}
		
		int64_t delayer::convert_timebased_xts_to_microseconds(int64_t timebased_xts)
		{
			static const int64_t microseconds = 100'0000;//1e+6 microseconds in second
			//av_q2d(stream_time_base_) is  frame rate per *second*		
			return static_cast<int64_t>(timebased_xts * microseconds*av_q2d(stream_time_base_));
		}
	
	}
}