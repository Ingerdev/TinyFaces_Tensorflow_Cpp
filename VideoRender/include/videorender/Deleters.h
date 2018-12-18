#pragma once
#pragma warning(push, 0) 
#pragma warning(pop) 
#include "ffmpeg.h"
namespace videorender
{
	namespace internal
	{
		struct AVCodecContextDeleter {
			void operator()(AVCodecContext* avCodecContext) const {
				if (avCodecContext->codec_id >= 0) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55, 53, 0)
					avcodec_free_context(&avCodecContext);
#else
					avcodec_close(avCodecContext);
					av_freep(&avCodecContext);
#endif
				}
			};
		};
		struct AVFormatContextDeleter {
			void operator()(AVFormatContext* avFormatContext) const {
				// Close an opened input AVFormatContext
				avformat_close_input(&avFormatContext);
			}
		};
		struct AVPacketDeleter {
			void operator()(AVPacket* avPacket) const {
				// Wipe the packet & unreferences the buffer referenced by the packet
				av_packet_unref(avPacket);
			}
		};
		struct AVFrameDeleter {
			void operator()(AVFrame* avFrame) const {
				av_frame_free(&avFrame);
			}
		};
		struct SwsContextDeleter {
			void operator()(SwsContext* swsContext) const {
				// Close an opened input SwsContext
				sws_freeContext(swsContext);
			}
		};
	}
}
