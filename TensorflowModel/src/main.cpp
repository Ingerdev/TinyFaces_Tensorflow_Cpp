
#pragma warning(push, 0) 
#include <memory>
#include <iostream>
#include <filesystem> // Microsoft-specific implementation header file name  
#pragma warning(pop) 

#include "tensorflow_model/TfH5ModelLoader.h"
#include "tensorflow_model/ModelRunner.h"
#include "tensorflow_model/TestOps.h"
#include "tensorflow_model/tf_types_converter.h"
#include "videorender/VideoRender.h"

std::string get_current_directory()
{
	auto path = std::filesystem::current_path();
	return path.generic_string();
}

/*
int main()
{	
	//tf_helpers::test();
	auto path = std::filesystem::current_path().generic_string();
	auto loader = std::make_unique<tiny_face_model::TfH5ModelLoader>();
	loader->create_model(path +R"(\..\data\model\weights.h5)");

	auto runner = tiny_face_model::internal::ModelRunner(*loader);
	auto predictions = runner.generate_predictions(path + R"(\..\data\images\Chapaev.avi_snapshot_00.27.10_[2018.01.18_16.39.20].jpg)");
	//predictions now contain set of Prediction objects.
	return 0;
}
*/
int main()
{
	auto render = videorender::VideoRender();
	render.start_render(R"(E:\Programs\test\chapaev.avi)");
}