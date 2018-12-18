extern "C"
{
#include <libavutil/time.h>
#include <libavutil/rational.h>
#include <libavcodec/avcodec.h>
}
namespace videorender
{
	namespace internal
	{
		//add pauses between frames.
		//delay calculate based on packet dts info
		//and stream's time_base
		class delayer
		{
		public:
			delayer(AVRational stream_time_base);
			void delay_frame(AVPacket* packet);
		private:
			int64_t convert_timebased_xts_to_microseconds(int64_t timebased_xts);
			const AVRational stream_time_base_;
			
			int64_t start_time_;
			int64_t start_xts_;
		};
	}
}