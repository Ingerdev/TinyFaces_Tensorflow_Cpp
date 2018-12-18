#pragma once
#pragma warning(push, 0) 
#include <fstream>
#include <string>
#pragma warning(pop) 

//small RAII wrapper for "C" FILE* descriptor

class CFileHandleWrapper
{
public:

	CFileHandleWrapper(const std::string& file_path, const std::string& open_mode) :file_(nullptr),_opened(false)
	{
		file_ = fopen(file_path.c_str(), open_mode.c_str()); // non-Windows use "r" //"rb"
		
		//check file handle and set internal state variable if file was opened
		if (nullptr != file_)
			_opened = true;
	}

	~CFileHandleWrapper()
	{
		if (file_ != nullptr)
			fclose(file_);
	}

	bool is_opened()
	{
		return _opened;
	}

	FILE* get()
	{
		return file_;
	}

private:
	FILE* file_;
	bool _opened;

};

