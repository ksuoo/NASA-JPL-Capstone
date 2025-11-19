#include <iostream>
#include <mutex>
#include <vector>
#include <string>
#include "Log.hpp"
using namespace std;

// Initialize static members
Log* Log::instancePtr = nullptr;
mutex Log::mtx;
Log::Log()
{
    log = new std::vector<std::string>(); 
}

Log::~Log()
{
}

Log* Log::getInstance() {
    if (instancePtr == nullptr) {
        lock_guard<mutex> lock(mtx);
        if (instancePtr == nullptr) {
            instancePtr = new Log();
        }
    }
    return instancePtr;
}

void Log::insert(string message){
    log->push_back(message);
}

vector<string> * Log::getStringVector(){
    return log;
}

string Log::getLastMessage(){
    return log->back();
}

