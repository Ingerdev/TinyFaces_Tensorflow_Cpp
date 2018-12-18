#pragma once
#pragma warning(push, 0) 
#pragma warning(pop) 
#include "BaseStreamDecoder.h"

#include "ffmpeg.h"
#include "SignalTypes.h"

namespace videorender
{
	namespace internal
	{
		class StreamVideoDecoder :
			public BaseStreamDecoder
		{
		public:
			
			StreamVideoDecoder();
			~StreamVideoDecoder() = default;

			virtual void decode_stream(AVFormatContext* input_stream, 
				AVCodecContext* input_codec, int stream_id,float file_seek_percentage = 0);
			
			//frame signal getter
			void register_image_callback(frame_callback_t frame_callback);

		private:
			int decode_packets(AVCodecContext * input_codec, AVFrame * frame, SwsContext* frame_converter, AVFrame * cv_adapted_frame);

			//some codecs keep last packets in buffer so we should try to read them 
			int decode_buffered_packets(AVCodecContext * input_codec, AVFrame * frame, SwsContext* frame_converter, AVFrame* cv_adapted_frame);
			//SwsContext * create_sws_context()

			//return true if loop should continue
			//return false when decoding should be stopped
			bool handle_decoded_frame(AVFrame * frame);

			AVFrame* create_cv_frame(AVCodecContext * input_codec, AVPixelFormat pixel_format);
			void encode_cv_adapted_frame(SwsContext * sws_cv, AVFrame * frame, AVFrame* cv_adapted_frame);
			SwsContext* create_sws_frame_converter(AVCodecContext * input_codec, AVPixelFormat new_pixel_format);

			cv::Size calculate_preffered_frame_size(const cv::Size& original_size) const;

			AVPixelFormat cv_pixel_format_;
			frame_callback_t frame_callback_;
			const int maximum_decoded_frame_dimension_ = 800;
		};
	}
}
