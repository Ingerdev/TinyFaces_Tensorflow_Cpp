#pragma warning(push, 0) 
#pragma warning(pop) 
#include "videorender/BaseStreamDecoder.h"

namespace videorender
{
	namespace internal
	{

		BaseStreamDecoder::BaseStreamDecoder()
		{
		}


		BaseStreamDecoder::~BaseStreamDecoder()
		{
		}

		int BaseStreamDecoder::print_error(const char* prefix, int errorCode)
		{
			if (errorCode == 0) {
				return 0;
			}
			else {
				const size_t bufsize = 64;
				char buf[bufsize];

				if (av_strerror(errorCode, buf, bufsize) != 0) {
					strcpy_s(buf, "UNKNOWN_ERROR");
				}
				fprintf(stderr, "%s (%d: %s)\n", prefix, errorCode, buf);
				return errorCode;
			}
		}
	}
}