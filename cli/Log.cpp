#include <iostream>
#include <mutex>
#include <vector>
#include <string>
using namespace std;


class Log
{
private:
    Log();
    vector<string> * log;
    // Static pointer to the Singleton instance
    static Log* instancePtr;

    // Mutex to ensure thread safety
    static mutex mtx;

public:
    static Log* getInstance();
    void insert(string message);
    vector<string> * getStringVector();
    string getLastMessage();
    ~Log();
};

// Initialize static members
Log* Log::instancePtr = nullptr;
mutex Log::mtx;
Log::Log()
{
    log = new vector<string>;
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

