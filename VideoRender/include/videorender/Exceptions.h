#pragma once
#pragma warning(push, 0) 
#include <exception>
#pragma warning(pop) 
namespace videorender
{
	namespace internal
	{
		enum class VideoDecoderError
		{
			CannotOpenFile,
			CannotFindProperCodec,
			CannotFindBestStream,
			CannotFindStreamInfo,
			CannotCreateSwsContext,
			CannotFillFramedata,
			CannotAllocateFrame,
			CannotCopyDecoderParameters,
			UnknownError,


		};
		class VideoDecoderException : public std::exception
		{
		public:
			VideoDecoderException(VideoDecoderError e) :Error(e) {}

			virtual char const * what() const { return "Something bad happend."; }
			VideoDecoderError Error;
		};
	}
}

