#include <boost/fiber/all.hpp>
#include <chrono>
namespace app
{
	namespace test
	{
		using value_t = int64_t;

		using buff_channel_t = boost::fibers::buffered_channel<value_t>;
		using unbuff_channel_t = boost::fibers::unbuffered_channel<value_t>;

		//flag channel type. shoult be typeless (message existence is a flag itself)
		using flag_t = bool;
		using buff_channel_f = boost::fibers::buffered_channel<flag_t>;
		using unbuff_channel_f = boost::fibers::unbuffered_channel<flag_t>;

		
		class spawner
		{
			buff_channel_t &value_output_;

		public:
			spawner(buff_channel_t& value_output):value_output_(value_output)
			{}
			void run()
			{
				int64_t counter{ 0 };
				while (true)
				{
					//send value and leave cycle if channel is closed
					if (boost::fibers::channel_op_status::closed ==  value_output_.try_push(counter))
						break;

					std::cout << "spawn " << counter <<"             ";
					//sleep for a bit;
					using namespace std::chrono;
					std::this_thread::sleep_for(0.2s);
					
					++counter;
				}
			}
		};

		class calculator
		{
		private:
			buff_channel_f &ready_output_;
			buff_channel_t &value_input_;
		public:
			calculator( buff_channel_t& value_input_channel,
						buff_channel_f& ready_output_channel):
				value_input_(value_input_channel),
				ready_output_(ready_output_channel) {}
			
			void run()
			{
				value_t value;
				while (true)
				{
					//send message "we are ready"
					ready_output_.push(flag_t());
					//synchronized call, wait till success or close happens.
					auto result = value_input_.pop(value);
					if (boost::fibers::channel_op_status::closed == result)
					{
						std::cout << "calc break\n";
						break;
					}
					
					std::cout << "calc " << value <<"\n";
					using namespace std::chrono;
					boost::this_fiber::sleep_for(5s);
					
				}
			}
		};

		//wait console input which mean halt of the program
		class console_waiter
		{
		private:
			buff_channel_f& _flag_output;
		public:
			console_waiter(buff_channel_f& channel) :_flag_output(channel) {};

			void run()
			{
				//read string from console
				std::string input;
				std::cin >> input;

				//signal output flag and return;
				_flag_output.push(flag_t());				
			}
		};

		//spawn other classes and wait
		class controller
		{
		private:
			value_t storage_;
			bool value_updated_;
			
			buff_channel_t& spawner_value_output_;
			
			buff_channel_f& calculator_ready_output_;
			buff_channel_t& calculator_value_input_;

			buff_channel_f& console_waiter_output_;


		public:
			controller( buff_channel_t& spawner_value_channel,
						buff_channel_t& calculator_value_channel,
						buff_channel_f&calculator_ready_channel,					    
						buff_channel_f&console_waiter_channel):
				spawner_value_output_(spawner_value_channel),				
				calculator_value_input_(calculator_value_channel),
				calculator_ready_output_(calculator_ready_channel),
				console_waiter_output_(console_waiter_channel)
			{

			}
			void run()
			{
				while (true)
				{
					//check console input
					flag_t unused;
					if (console_waiter_output_.try_pop(unused) == boost::fibers::channel_op_status::success)
					{
						close_channels();
						break;
					}


					//read all valuee from spawner channel and keep the last one in storage_
					if (read_update_from_spawner())
						value_updated_ = true;
					
					if (!value_updated_)
						continue;

					//check if calculator is ready and write value.				
					if (calculator_ready_output_.try_pop(unused) == boost::fibers::channel_op_status::success)
					{
						calculator_value_input_.push(storage_);
						value_updated_ = false;
					}
					
				}
			}

		private:
			
			//if true new value has been red
			bool read_update_from_spawner()
			{
				//read every value spawner messaged to us
				bool has_value = false;
				while (spawner_value_output_.try_pop(storage_) == boost::fibers::channel_op_status::success)
				{
					has_value = true;
				}
				return has_value;
			}

			void close_channels()
			{
				spawner_value_output_.close();
				calculator_ready_output_.close();
				calculator_value_input_.close();				
			}
		};

		
	}

	class runner
	{
	public: 

		void run()
		{
			try
			{
				std::cout << "runner.run\n";
				test::buff_channel_f console_waiter_channel(1<<1);
				std::cout << "console_waiter channel created\n";
				test::buff_channel_t spawner_value_channel(1<<5);
				std::cout << "spawner value channel created\n";
				test::buff_channel_t calculator_value_channel(1<<1);
				test::buff_channel_f calculator_ready_channel(1<<1);

				std::cout << "channels created\n";

				test::spawner spawner(spawner_value_channel);
				test::calculator calculator(calculator_value_channel, calculator_ready_channel);
				test::console_waiter waiter(console_waiter_channel);

				test::controller controller(spawner_value_channel,
					calculator_value_channel,
					calculator_ready_channel,
					console_waiter_channel);

				std::cout << "classes created\n";

				auto spawner_thread = create_threaded_fiber(std::bind(&test::spawner::run, spawner));
				auto console_waiter_thread = create_threaded_fiber(std::bind(&test::console_waiter::run, waiter));
				auto calculator_thread = create_threaded_fiber(std::bind(&test::calculator::run, calculator));
				auto controller_thread = create_threaded_fiber(std::bind(&test::controller::run, controller));

				std::cout << "fibers created\n";
				spawner_thread.join();
				console_waiter_thread.join();
				calculator_thread.join();
				controller_thread.join();
			}
			catch (const std::exception& e)
			{
				std::cout << "exception: " << e.what() << "\n";
			}
		}
	private:
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