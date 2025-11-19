#include <iostream>
#include <mutex>
#include <vector>
#include <string>
#ifndef LOG_H
#define LOG_H

class Log
{
private:
    Log();
    std::vector<std::string> * log;
    // Static pointer to the Singleton instance
    static Log* instancePtr;

    // Mutex to ensure thread safety
    static std::mutex mtx;

public:
    static Log* getInstance();
    void insert(std::string message);
    std::vector<std::string>* getStringVector();
    std::string getLastMessage();
    ~Log();
};

#endif