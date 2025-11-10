#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include<iostream>
#include<string>
#include<vector>
#include <sstream>
#include "ollama.hpp"
#include "Log.cpp"
#include "main.cpp"

using namespace std;
vector<string> parseLine(string line);

TEST_CASE("line is parsed correctly", "[parsing]") {
    string line = "this is a test";
    vector<string> result = parseLine(line);
	REQUIRE(result[0] == "this");
	REQUIRE(result[1] == "is");
	REQUIRE(result[2] == "a");
	REQUIRE(result[3] == "test");

    string line = "group \"the part in quotes\"";
    vector<string> result = parseLine(line);
	REQUIRE(result[0] == "group");
	REQUIRE(result[1] == "the part in quotes");
}

TEST_CASE("Accepts and understands prompts", "[VLM]"){
    string response = generate(parseLine("generate \"this is a test. respond with \'hello world\'\""));
    //convert to lowercase
    transform(response.begin(), response.end(), response.begin(), ::tolower);
    REQUIRE(response.find("hello world") != string::npos);

    //test image
    string response = generate(parseLine("generate \"What shape is this\" testImages/circle.png"));
    //convert to lowercase
    transform(response.begin(), response.end(), response.begin(), ::tolower);
    REQUIRE(response.find("circle") != string::npos);

    //test two images
    string response = generate(parseLine("generate \"What are these two shapes\" testImages/circle.png testImages/triangle.png"));
    transform(response.begin(), response.end(), response.begin(), ::tolower);
    REQUIRE(response.find("circle") != string::npos && response.find("triangle") != string::npos);
}
