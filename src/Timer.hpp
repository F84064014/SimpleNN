#pragma once

#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

class Timer
{

public:

	void start()
	{
		m_start_time = high_resolution_clock::now();
	}

	double end()
	{
		high_resolution_clock::time_point end_time = high_resolution_clock::now();
		duration<double, std::milli> ms_double = end_time - m_start_time;
		m_total_time += ms_double.count();
		m_cnt += 1;
		return ms_double.count();
	}

	double get_total_time()
	{
		return m_total_time;
	}

	double get_average_time()
	{
		if (m_cnt == 0) return -1;
		return m_total_time / m_cnt;
	}

private:
	high_resolution_clock::time_point m_start_time;
	double m_total_time = 0;
	int m_cnt = 0; // count how many times end() called
};
