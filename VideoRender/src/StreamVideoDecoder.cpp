#pragma warning(push, 0) 
#include <memory>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#pragma warning(pop) 

#include "videorender/StreamVideoDecoder.h"
#include "videorender/Deleters.h"
#include "videorender/Delayer.h"
#include "videorender/Seeker.h"



namespace videorender
{
	namespace internal
	{
		StreamVideoDecoder::StreamVideoDecoder() :cv_pixel_format_(AV_PIX_FMT_BGR24)
		{
		}
			   
		void StreamVideoDecoder::register_image_callback(frame_callback_t frame_callback)
		{
			frame_callback_ = frame_callback;
		}

		void StreamVideoDecoder::decode_stream(AVFormatContext * input_stream, AVCodecContext * input_codec, int stream_id,float file_seek_percentage)
		{			
			int err = 0;
			seeker::seek_percentile(input_stream, stream_id, file_seek_percentage);
			//allocate frame that will hold decoded data
			std::unique_ptr<AVFrame, AVFrameDeleter> frame(av_frame_alloc());
			if (frame.get() == nullptr)
				throw VideoDecoderException(VideoDecoderError::CannotAllocateFrame);

			//create frame to cv::mat sws converter
			std::unique_ptr<SwsContext, SwsContextDeleter> sws_cv(create_sws_frame_converter(input_codec, cv_pixel_format_));
			if (sws_cv.get() == nullptr)
				throw VideoDecoderException(VideoDecoderError::CannotCreateSwsContext);

			std::unique_ptr<AVFrame, AVFrameDeleter> cv_frame(create_cv_frame(input_codec, cv_pixel_format_));
			if (cv_frame.get() == nullptr)
				throw VideoDecoderException(VideoDecoderError::CannotAllocateFrame);

			// Prepare the packet.
			std::unique_ptr<AVPacket, AVPacketDeleter> packet(av_packet_alloc());
			// Set default values.
			av_init_packet(packet.get());

			
			delayer delayer(input_stream->streams[stream_id]->time_base);
			while ((err = av_read_frame(input_stream, packet.get())) != AVERROR_EOF) {
				
				if (err != 0) {
					// Something went wrong.
					print_error("Read error.", err);
					break; // Don't return, so we can clean up nicely.
				}
				
				// Does the packet belong to the correct stream?
				if (packet.get()->stream_index != stream_id) {
					// Free the buffers used by the frame and reset all fields.
					av_packet_unref(packet.get());
					continue;
				}		

				//delay frame output to fit timebase fps
				delayer.delay_frame(packet.get());
				// We have a valid packet => send it to the decoder.
				if ((err = avcodec_send_packet(input_codec, packet.get())) == 0) {
					// The packet was sent successfully. We don't need it anymore.
					// => Free the buffers used by the frame and reset all fields.
					av_packet_unref(packet.get());
				}
				else {
					// Something went wrong.
					// EAGAIN is technically no error here but if it occurs we would need to buffer
					// the packet and send it again after receiving more frames. Thus we handle it as an error here.
					print_error("Send error.", err);
					break; // Don't return, so we can clean up nicely.
				}			
				// Receive and handle frames.
				// EAGAIN means we need to send before receiving again. So thats not an error.
				if ((err = decode_packets(input_codec, frame.get(), sws_cv.get(), cv_frame.get())) != AVERROR(EAGAIN)) {
					// Not EAGAIN => Something went wrong.
					print_error("Receive error.", err);
					break; // Don't return, so we can clean up nicely.
				}
			}
			//drain rest of packets
			if ((err = decode_buffered_packets(input_codec, frame.get(), sws_cv.get(), cv_frame.get())) != AVERROR(EAGAIN)) {
				// Not EAGAIN => Something went wrong.
				print_error("Receive error.", err);
			}
		}
		int StreamVideoDecoder::decode_packets(AVCodecContext * input_codec, AVFrame * frame, SwsContext* frame_converter, AVFrame* cv_adapted_frame)
		{
			int err = 0;
			// Read the packets from the decoder.
			// NOTE: Each packet may generate more than one frame, depending on the codec.
			while ((err = avcodec_receive_frame(input_codec, frame)) == 0) {
				// Let's handle the frame in a function.
				encode_cv_adapted_frame(frame_converter, frame, cv_adapted_frame);
			
				auto frame_handled = handle_decoded_frame(cv_adapted_frame);
					
				// Free any buffers and reset the fields to default values.
				av_frame_unref(frame);
				if (!frame_handled)
					break;
			}
			return err;
		}
		int StreamVideoDecoder::decode_buffered_packets(AVCodecContext * input_codec, AVFrame * frame, SwsContext* frame_converter, AVFrame* cv_adapted_frame)
		{
			int err = 0;
			// Some codecs may buffer frames. Sending NULL activates drain-mode.
			if ((err = avcodec_send_packet(input_codec, NULL)) == 0) {
				// Read the remaining packets from the decoder.
				err = decode_packets(input_codec, frame, frame_converter, cv_adapted_frame);

				if (err != AVERROR(EAGAIN) && err != AVERROR_EOF) {
					// Neither EAGAIN nor EOF => Something went wrong.
					print_error("Receive error.", err);
				}
			}
			else {
				// Something went wrong.
				print_error("Send error.", err);
			}

			return err;
		}
		void StreamVideoDecoder::encode_cv_adapted_frame(SwsContext * sws_cv, AVFrame * frame, AVFrame * cv_adapted_frame)
		{
			sws_scale(sws_cv, frame->data, frame->linesize, 0, frame->height,
				cv_adapted_frame->data, cv_adapted_frame->linesize);
		}

		AVFrame * StreamVideoDecoder::create_cv_frame(AVCodecContext * input_codec, AVPixelFormat pixel_format)
		{
			std::unique_ptr<AVFrame, AVFrameDeleter> frame(av_frame_alloc());
			frame.get()->format = pixel_format;
			frame.get()->width = input_codec->width;
			frame.get()->height = input_codec->height;

			//align - 32
			if (av_frame_get_buffer(frame.get(), 1))
				return nullptr;

			return frame.release();
		}

		SwsContext * StreamVideoDecoder::create_sws_frame_converter(AVCodecContext * input_codec, AVPixelFormat new_pixel_format)
		{
			return sws_getCachedContext(
				nullptr,
				input_codec->width,
				input_codec->height,
				input_codec->pix_fmt,
				input_codec->width,
				input_codec->height,
				new_pixel_format,
				SWS_BICUBIC,
				nullptr,
				nullptr,
				nullptr
			);
		}

		bool StreamVideoDecoder::handle_decoded_frame(AVFrame * frame)
		{
			//convert AVFrame to cv::Mat
			
			//looks very ineffective, possible reusing this dst_image instead of creating it every frame			
			cv::Mat src_image(frame->height, frame->width, CV_8UC3, frame->data[0], frame->linesize[0]);
			cv::Mat dst_image(maximum_decoded_frame_dimension_, maximum_decoded_frame_dimension_,CV_8UC3);
			
			//resize image to a maximum dim equal maximum_decoded_frame_dimension_ constant 
			auto new_size = calculate_preffered_frame_size(src_image.size());
		
			cv::resize(src_image, dst_image, new_size, 0, 0);
			return frame_callback_(dst_image);
		}
		cv::Size StreamVideoDecoder::calculate_preffered_frame_size(const cv::Size& original_size) const
		{
			
			auto calc_lambda = [&](int longest_dim,const cv::Size& original_size)
			{
				if (longest_dim < maximum_decoded_frame_dimension_)
					return original_size;

				double compression = static_cast<double>(maximum_decoded_frame_dimension_)/longest_dim ;
				return cv::Size(
					static_cast<int>(original_size.width*compression),
					static_cast<int>(original_size.height*compression));
			};

			if (original_size.width > original_size.height)
			{
				return calc_lambda(original_size.width,original_size);
			}
			else
			{
				return calc_lambda(original_size.height, original_size);			
			}
		}
	}
}