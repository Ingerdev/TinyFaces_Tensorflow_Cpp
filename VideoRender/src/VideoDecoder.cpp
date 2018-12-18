#pragma warning(push, 0) 
#include <iostream>
#include <memory>
#include <functional>
#pragma warning(pop) 

#include "videorender/VideoDecoder.h"
#include "videorender/Deleters.h"
namespace videorender
{
	namespace internal
	{

		VideoDecoder::VideoDecoder(StreamDecodersManager& manager, BaseStreamsSelector& streams_selector) :
			_av_init(AVInitSingleton::Instance()),
			_stream_decoders_manager(manager),
			_streams_selector(streams_selector)
		{

		}

		//c++0x11 feature - calling another constructor
		//preventing initializers duplicate
		VideoDecoder::VideoDecoder(StreamDecodersManager& manager, BaseStreamsSelector& streams_selector, 
			const std::string_view filename) :
			VideoDecoder(manager, streams_selector)
		{
			open(filename);
		}

		void VideoDecoder::open(const std::string_view  filename,float file_seek_percentage)
		{
			parse_video_from_file(filename, file_seek_percentage);
		}

		void VideoDecoder::parse_video_from_file(const std::string_view filename,float file_seek_percentage)
		{
			// open input file context		
			AVFormatContext* temp_context{ nullptr };
			int ret = avformat_open_input(&temp_context, std::string(filename).c_str(), nullptr, nullptr);
			if (ret < 0) {
				std::cerr << "fail to avforamt_open_input(\"" << filename << "\"): ret=" << ret;
				throw VideoDecoderException(VideoDecoderError::CannotOpenFile);
			}

			_inctx.reset(temp_context);

			// retrive input stream information
			ret = avformat_find_stream_info(_inctx.get(), nullptr);
			if (ret < 0) {
				std::cerr << "fail to avformat_find_stream_info: ret=" << ret;
				throw VideoDecoderException(VideoDecoderError::CannotFindStreamInfo);
			}

			decode_streams(file_seek_percentage);

			//init_scaler();
			return;
		}

		void VideoDecoder::init_scaler()
		{
			int ret = 0;

			// initialize sample scaler
			const int dst_width = _vstrm->codecpar->width;
			const int dst_height = _vstrm->codecpar->height;

			const AVPixelFormat dst_pix_fmt = AV_PIX_FMT_BGR24;

			_swsctx.reset(sws_getCachedContext(
				nullptr, _vstrm->codecpar->width, _vstrm->codecpar->height, (AVPixelFormat)_vstrm->codecpar->format,
				dst_width, dst_height, dst_pix_fmt, SWS_BICUBIC, nullptr, nullptr, nullptr));

			if (!_swsctx) {
				std::cerr << "fail to sws_getCachedContext";
				throw VideoDecoderException(VideoDecoderError::CannotCreateSwsContext);
			}

			_frame.reset(av_frame_alloc());

			_framebuf.resize(av_image_get_buffer_size(dst_pix_fmt, dst_width, dst_height, 1));

			ret = av_image_fill_arrays(_frame->data,
				_frame->linesize,
				_framebuf.data(), dst_pix_fmt, dst_width, dst_height, 1);

			if (ret < 0) {
				std::cerr << "fail to fill frame data: ret=" << ret;
				throw VideoDecoderException(VideoDecoderError::CannotFillFramedata);
			}
		}

		void VideoDecoder::decode_streams(float file_seek_percentage)
		{
			// get requested stream types and count
			auto streams = _streams_selector.SelectStreams(_inctx.get());
			for (auto stream_id : streams)
			{
				//getting stream type
				AVStream* stream = _inctx->streams[stream_id];
				switch (stream->codecpar->codec_type)
				{
				case AVMediaType::AVMEDIA_TYPE_VIDEO:
					decode_video_stream(stream_id, file_seek_percentage);
					break;
					//todo:implement decoding methods for every AVMediaType types
				case AVMediaType::AVMEDIA_TYPE_AUDIO:
				case AVMediaType::AVMEDIA_TYPE_NB:
				default:
					break;
				}
			}
		}

		void VideoDecoder::decode_video_stream(int stream_id,float file_seek_percentage)
		{

			AVStream* stream = _inctx->streams[stream_id];

			// open video decoder contetx
			AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);


			std::unique_ptr<AVCodecContext, AVCodecContextDeleter> codecCtx;
			codecCtx.reset(avcodec_alloc_context3(codec));

			int ret = avcodec_parameters_to_context(codecCtx.get(), stream->codecpar);
			if (ret < 0) {
				std::cerr << "Failed to copy decoder parameters to input decoder context ret=" << ret;
				throw VideoDecoderException(VideoDecoderError::CannotCopyDecoderParameters);
			}
#if 0
			avcodec_parameters_to_context(codecCtx.get(),
				_vstrm->codecpar);
#endif

			ret = avcodec_open2(codecCtx.get(), codec, NULL);

			if (ret < 0) {
				std::cerr << "fail to avcodec_open2: ret=" << ret;
				throw VideoDecoderException(VideoDecoderError::CannotOpenFile);
			}
			_stream_decoders_manager.decode_stream(_inctx.get(), codecCtx.get(), stream_id,
				stream->codecpar->codec_type, file_seek_percentage);
		}
#if 0
		int VideoDecoder::process_frame(void * ctx, AVFrame * frame)
		{

			struct SwsContext * img_convert_ctx;
			img_convert_ctx = sws_getCachedContext(NULL, pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
			sws_scale(img_convert_ctx, ((AVPicture*)pFrame)->data, ((AVPicture*)pFrame)->linesize, 0, pCodecCtx->height, ((AVPicture *)pFrameRGB)->data, ((AVPicture *)pFrameRGB)->linesize);

			cv::Mat img(pFrame->height, pFrame->width, CV_8UC3, pFrameRGB->data[0]); //dst->data[0]);
			cv::imshow("display", img);
			cvWaitKey(1);

			av_free_packet(&packet);
			sws_freeContext(img_convert_ctx);

			return 0;

		}
#endif
	}
}