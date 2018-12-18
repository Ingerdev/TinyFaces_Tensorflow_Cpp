#pragma once
#pragma warning(push, 0) 
#include <exception>
#include <string>
#pragma warning(pop) 

namespace tiny_face_model
{
	class image_load_exception :public std::exception
	{
		std::string message_;
		std::string image_name_;
	public:
		image_load_exception(const std::string & msg,
			const std::string & image_name) :std::exception(msg.c_str()),
			message_(msg),
			image_name_(image_name)
		{
		}
		const std::string& message() const noexcept { return message_; } 
		const std::string& image_name() const noexcept { return image_name_; }

	};
}

