#pragma once
#include <tensorflow_model/TfH5ModelLoader.h>
#include <tensorflow_model/ModelRunner.h>

#include <VideoRender/VideoRender.h>
#include <VideoRender/SignalTypes.h>
#include <videorender/Prediction.h>
#include <videorender/SignalTypes.h>
#include <videorender/RenderToDisplay.h>

#include <mutex>
#include <thread>

#include <boost/fiber/all.hpp>
#include <chrono>
#include <string_view>
#include <filesystem>


namespace app
{
	using image_value_t = cv::Mat;
	using predict_value_t = std::vector<videorender::Prediction>;

	using image_channel_t = boost::fibers::buffered_channel<image_value_t>;
	using predict_channel_t = boost::fibers::buffered_channel<predict_value_t>;

	//flag channel type. should be typeless (message existence is a flag itself)
	using flag_t = bool;
	using signal_channel_f = boost::fibers::buffered_channel<flag_t>;

	class frame_spawner
	{
	private:
		image_channel_t& output_image_channel_;
		videorender::frame_callback_t frame_ready_;
		videorender::VideoRender render_;

	public:
		frame_spawner(image_channel_t& image_channel) :output_image_channel_(image_channel)
		{

		};
		void run(std::string_view file_name,float file_seek_percentage)
		{
			videorender::VideoRender render;
			render.register_image_callback(std::bind(&frame_spawner::frame_ready, &*this, std::placeholders::_1));
			
			//decoder's loop
			render.start_render(file_name, file_seek_percentage);
		}

	private:

		//frame event handler
		bool frame_ready(const cv::Mat& frame)
		{
			//if channel is full - dont wait
			auto status = output_image_channel_.try_push(frame);
			
			//return false if our channel is closed
			//what means "terminate" video decoding
			return (status != boost::fibers::channel_op_status::closed);
			//return true;
		}
	};

	class face_finder
	{
	private:
		image_channel_t& input_image_channel_;
		predict_channel_t& output_predictions_channel_;
		float faceprob_lower_limit_;
	public:
		face_finder(image_channel_t& image_channel, predict_channel_t& predictions_channel,float faceprob_lower_limit) :
			input_image_channel_(image_channel),
			output_predictions_channel_(predictions_channel),
			faceprob_lower_limit_(faceprob_lower_limit)
		{}
		void run()
		{
			//create face recognizer
			auto path = std::filesystem::current_path().generic_string();
			auto loader = std::make_unique<tiny_face_model::TfH5ModelLoader>();
			loader->create_model(path + R"(\..\data\model\weights.h5)");

			auto runner = tiny_face_model::internal::ModelRunner(*loader);

			//thread loop
			while (true)
			{
				//read image from image_channel
				cv::Mat image;
				if (input_image_channel_.pop(image) == boost::fibers::channel_op_status::closed)
					break;

				//calculate faces
				auto predictions = runner.generate_predictions(image);
				std::copy(predictions.begin(), predictions.end(), std::ostream_iterator<videorender::Prediction>(std::cout, " "));
				
				//remove from predictions every one with probability lower than lower limit.
				predictions.erase(
					std::remove_if(predictions.begin(),predictions.end(),
					[&](videorender::Prediction& pred) {return pred.probability < faceprob_lower_limit_; }),
						predictions.end());

				//write predictions to predictions_channel
				if (output_predictions_channel_.push(predictions) == boost::fibers::channel_op_status::closed)
					break;
			}
			
		}
	};
	class frame_render
	{
		image_channel_t& input_image_channel_;
		predict_channel_t& input_predictions_channel_;
	public:
		frame_render(image_channel_t& image_channel, predict_channel_t& predict_channel):
			input_image_channel_(image_channel),
			input_predictions_channel_(predict_channel)
		{
		}
		void run()
		{
			videorender::RenderToDisplay renderer;
			while (true)
			{				
				cv::Mat image;
				if (input_image_channel_.pop(image) == boost::fibers::channel_op_status::closed)
					break;

				renderer.process_frame(image);

				predict_value_t predictions;
				auto status = input_predictions_channel_.try_pop(predictions);
				if (status == boost::fibers::channel_op_status::closed)
					break;
				if (status == boost::fibers::channel_op_status::success)					
					renderer.process_faces(predictions);

			}

		}
	};
	//takes input channel and provide its output replicas
	//so it can also be named "multiplexor"
	template <class ChannelT>
	class replicator
	{
	public:
		using channel_type = ChannelT;
		using value_type = typename channel_type::value_type;
		replicator(channel_type& channel): input_channel(channel)
		{			
		}

		channel_type& add_output_channel(size_t capacity)
		{			
			out_channels.emplace_back(capacity);
			return out_channels.back();
		}

		void run()
		{
			value_type value;
			while (true)
			{
				auto status = input_channel.pop(value);
				if (status == boost::fibers::channel_op_status::closed)
					break;
				
				//copy input value to each of output channels
				for (auto& channel : out_channels)
				{
					channel.try_push(value);
				}
			}
		}
	private:
		channel_type& input_channel;
		std::list<channel_type> out_channels;
	};
	class Communicator
	{
	public:
		Communicator(std::string_view file_name,float file_seek_percentage,float faceprob_lower_limit)
		{
			try
			{
				video_run(file_name, file_seek_percentage, faceprob_lower_limit);
				//image_run(file_name);
			}
			catch (std::exception& ex)
			{
				std::cout << "exception: " << ex.what();
			}
			
		}
	private:
		void image_run(std::string_view file_name, float faceprob_lower_limit)
		{
			//data channels
			constexpr size_t channel_capacity = 1 << 1;
			image_channel_t image_channel(channel_capacity);
			predict_channel_t predict_channel(channel_capacity);

			auto path = std::filesystem::current_path().generic_string();
			auto image = cv::imread(path + R"(\..\data\images\Chapaev.avi_snapshot_00.27.10_[2018.01.18_16.39.20].jpg)", CV_LOAD_IMAGE_COLOR);
			//create channels' users
			face_finder face_finder(image_channel, predict_channel, faceprob_lower_limit);
			//create working threads			
			auto face_finder_thread = create_threaded_fiber(std::bind(&face_finder::run, &face_finder));
			
			image_channel.push(image);
			//wait all threads to end
			face_finder_thread.join();
		}
		void video_run(std::string_view file_name,float file_seek_percentage,float faceprob_lower_limit)
		{
			//data channels
			constexpr size_t channel_capacity = 1 << 1;
			image_channel_t image_channel(channel_capacity);
			predict_channel_t predict_channel(channel_capacity);

			//multiplex image_channel
			auto repl_image_channels = replicator(image_channel);
			auto& first_repl_image_channel = repl_image_channels.add_output_channel(channel_capacity);
			auto& second_repl_image_channel = repl_image_channels.add_output_channel(channel_capacity);

			//create channels' users
			frame_spawner frame_spawner(image_channel);
			frame_render frame_renderer(first_repl_image_channel, predict_channel);
			face_finder face_finder(second_repl_image_channel, predict_channel, faceprob_lower_limit);

			//create working threads
			auto frame_renderer_thread = create_threaded_fiber(std::bind(&frame_render::run, &frame_renderer));
			auto image_replicator_thread = create_threaded_fiber(
				std::bind(&decltype(repl_image_channels)::run, &repl_image_channels));

			auto frame_spawner_thread = create_threaded_fiber(std::bind(&frame_spawner::run, &frame_spawner,
				file_name, file_seek_percentage));
			auto face_finder_thread = create_threaded_fiber(std::bind(&face_finder::run, &face_finder));

			//wait all threads to end
			frame_spawner_thread.join();
			frame_renderer_thread.join();
			face_finder_thread.join();
			image_replicator_thread.join();

			//while (cvWaitKey(0) != 27);
		}
		template <typename Fn>
		std::thread create_threaded_fiber(Fn& f)
		{
			return std::thread([&]()
			{
				boost::fibers::fiber fiber(f);
				fiber.join();
			});
		};
	};
}