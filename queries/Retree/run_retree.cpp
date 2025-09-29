#include "duckdb.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <regex>

std::string LOAD_PATH = "/volumn/Retree_exp/queries/Retree/common/";
std::string SQL_PATH = "/volumn/Retree_exp/queries/Retree/workloads/";
std::string MODEL_PATH = "/volumn/Retree_exp/workloads/";

struct Config
{
	std::string workload = "nyc-taxi-green-dec-2016";
	std::string model = "nyc-taxi-green-dec-2016_d11_l1491_n2981_20250112085333";
	std::string model_type = "rf";
	std::string scale = "1G";
	std::string thread = "4";
	int times = 6;
	int optimization_level = 3;
	int debug = 0;
};

Config parse_args(int argc, char *argv[])
{
	Config config;
	int opt;
	while ((opt = getopt(argc, argv, "t:w:o:m:s:n:d:")) != -1)
	{
		switch (opt)
		{
		case 'w':
			config.workload = optarg;
			break;
		case 'm':
			config.model = optarg;
			break;
		case 's':
			config.scale = optarg;
			break;
		case 't':
			config.thread = optarg;
			break;
		case 'o':
			config.optimization_level = atoi(optarg);
			break;
		case 'd':
			config.debug = atoi(optarg);
			break;
		default:
			std::cerr << "Usage: " << argv[0]
					  << " [-w workloads] [-m model] [-s scale] [-t threads] [-o optimization_level] [-d debug]\n";
			exit(EXIT_FAILURE);
		}
	}
	return config;
}

std::string read_file(const std::string &filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Unable to open file: " + filename);
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

std::vector<std::string> read_predicates(const std::string &filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Unable to open predicate file: " + filename);
	}
	std::vector<std::string> predicates;
	std::string line;
	while (std::getline(file, line))
	{
		predicates.push_back(line);
	}
	return predicates;
}

std::string replacePlaceholder(std::string str, const std::string &from, const std::string &to)
{
	size_t start_pos = str.find(from);
	if (start_pos != std::string::npos)
	{
		str.replace(start_pos, from.length(), to);
	}
	return str;
}

void run(const Config &config)
{
	std::string sql_path = SQL_PATH + config.workload + "/";
	std::ofstream outputfile;
	outputfile.open(sql_path + "output.csv", std::ios::app);

	duckdb::DBConfig db_config;
	db_config.options.allow_unsigned_extensions = true;
	db_config.options.allow_extensions_metadata_mismatch = true;
	duckdb::DuckDB db(nullptr, &db_config);
	duckdb::Connection con(db);

	con.Query("PRAGMA disable_verification;");
	con.Query("set allow_extensions_metadata_mismatch=true;");
	con.Query(read_file(LOAD_PATH + "load_inference_function.sql"));

	switch (config.optimization_level)
	{
	case 1:
		break;
	case 2:
		con.Query(read_file(LOAD_PATH + "load_convert_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_prune_rule.sql"));
		break;
	case 3: // merge
	case 5: // one boundary
		con.Query(read_file(LOAD_PATH + "load_convert_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_prune_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_merge_rule.sql"));
		break;
	case 4:
		con.Query(read_file(LOAD_PATH + "load_convert_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_prune_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_naive_merge_rule.sql"));
		break;
	case 6:
		con.Query(read_file(LOAD_PATH + "load_retree_rules_1.sql"));
		break;
	case 7:
		con.Query(read_file(LOAD_PATH + "load_retree_rules_2.sql"));
		break;
	case 8: // clf2reg
		con.Query(read_file(LOAD_PATH + "load_convert_rule.sql"));
		break;
	default:
		break;
	}

	std::vector<std::string> predicates;
	if (config.model_type == "rf")
	{
		predicates = read_predicates(sql_path + "predicates.txt");
	}
	else
	{
		predicates = read_predicates(sql_path + "predicates-dt.txt");
	}

	std::string data = replacePlaceholder(read_file(sql_path + "load_data.sql"), "?", config.scale);
	data = replacePlaceholder(data, "?", config.scale);
	data = replacePlaceholder(data, "?", config.scale);
	
	std::string threads = replacePlaceholder("set threads = ?;", "?", config.thread);

	con.Query(data);
	con.Query(threads);
	std::vector<double> records;
	int count = config.times;
	for (const auto &predicate : predicates)
	{
		std::string sql = read_file(sql_path + "query.sql");
		if (config.optimization_level == 1)
		{
			sql = replacePlaceholder(sql, "predict", "forge");
		}
		sql = replacePlaceholder(sql, "?", config.model);
		sql = replacePlaceholder(sql, "?", predicate);

		records.clear();
		count = config.times;
		while (count--)
		{
			auto start = std::chrono::high_resolution_clock::now();
			con.Query(sql);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> duration = end - start;
			records.push_back(duration.count());
		}

		auto maxminvals =
			*std::max_element(records.begin(), records.end()) + *std::min_element(records.begin(), records.end());
		double sum = std::accumulate(records.begin(), records.end(), 0.0) - maxminvals;
		double average = sum / (records.size() - 2);

		outputfile << config.workload << "," << config.model << "," << config.model_type << "," << predicate << "," << config.scale << ","
				   << config.thread << "," << config.optimization_level << "," << average << "\n";
		std::cout << config.workload << "," << config.model << "," << config.model_type << "," << predicate << "," << config.scale << ","
				  << config.thread << "," << config.optimization_level << "," << average << "\n";
		// for test use
		// if (config.optimization_level <= 3)
		// 	break;
		if (config.optimization_level <= 1)
			break;
	}
	outputfile.close();
}

void debug(const Config &config)
{
	duckdb::DBConfig db_config;
	db_config.options.allow_unsigned_extensions = true;
	db_config.options.allow_extensions_metadata_mismatch = true;
	duckdb::DuckDB db(nullptr, &db_config);
	duckdb::Connection con(db);

	con.Query("PRAGMA disable_verification;");
	con.Query("set allow_extensions_metadata_mismatch=true;");

	con.Query(read_file(LOAD_PATH + "load_inference_function.sql"));

	switch (config.optimization_level)
	{
	case 1:
		break;
	case 2:
		con.Query(read_file(LOAD_PATH + "load_convert_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_prune_rule.sql"));
		break;
	case 3: // merge
	case 5: // one boundary
		con.Query(read_file(LOAD_PATH + "load_convert_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_prune_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_merge_rule.sql"));
		break;
	case 4:
		con.Query(read_file(LOAD_PATH + "load_convert_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_prune_rule.sql"));
		con.Query(read_file(LOAD_PATH + "load_naive_merge_rule.sql"));
		break;
	case 6:
		con.Query(read_file(LOAD_PATH + "load_retree_rules_1.sql"));
		break;
	case 7:
		con.Query(read_file(LOAD_PATH + "load_retree_rules_2.sql"));
		break;
	case 8: // clf2reg
		con.Query(read_file(LOAD_PATH + "load_convert_rule.sql"));
		break;
	default:
		break;
	}

	std::string sql_path = SQL_PATH + config.workload + "/";
	std::vector<std::string> predicates;
	if (config.model_type == "rf")
	{
		predicates = read_predicates(sql_path + "predicates.txt");
	}
	else
	{
		predicates = read_predicates(sql_path + "predicates-dt.txt");
	}

	std::ofstream outputfile;
	outputfile.open(sql_path + "output-debug.csv", std::ios::app);

	std::string threads = replacePlaceholder("set threads = ?;", "?", config.thread);
	con.Query(threads);

	auto result = con.Query(replacePlaceholder(read_file(sql_path + "load_data.sql"), "?", config.scale));
	outputfile << result->ToString() << "\n";
	for (const auto &predicate : predicates)
	{
		std::string sql = read_file(sql_path + "query.sql");
		sql = replacePlaceholder(sql, "?", config.model);
		sql = replacePlaceholder(sql, "?", predicate);

		auto start = std::chrono::high_resolution_clock::now();
		auto result = con.Query(sql);
		outputfile << result->ToString() << "\n";
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> duration = end - start;
		auto average = duration.count();

		outputfile << config.workload << "," << config.model << "," << config.model_type << "," << predicate << "," << config.scale << ","
				   << config.thread << "," << config.optimization_level << "," << average << "\n";
		std::cout << config.workload << "," << config.model << "," << config.model_type << "," << predicate << "," << config.scale << ","
				  << config.thread << "," << config.optimization_level << "," << average << "\n";

		break;
	}
	outputfile.close();
}

int main(int argc, char *argv[])
{
	Config config = parse_args(argc, argv);

	std::regex rf_pattern("t100");
	if (regex_search(config.model, rf_pattern))
	{
		config.model_type = "rf";
		config.thread = "4";
	}
	else
	{
		config.model_type = "dt";
		config.thread = "1";
	}

	config.debug == 0 ? run(config) : debug(config);
	return 0;
}
