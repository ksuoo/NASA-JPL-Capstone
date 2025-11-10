#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "ollama.hpp"
#include "Log.cpp"
using namespace std;

vector<string> parseLine(string line){
    string arg;
    string segment;
    stringstream ss(line);
    vector<string> argv;
    if(line.length()== 0 ) return argv;

    //segment line into where quotes separate the command
    while(getline(ss, segment, '"')){
        //erase an extra space if exists
        if (segment[0] == ' '){segment.erase(0,1); }
        
        //cout<< segment << endl;
        stringstream ss2(segment);

        //separate words where there aren't quotes
        while(getline(ss2, arg, ' ')) {
            argv.push_back(arg);
        }

        //keep quoted sections grouped
        if(getline(ss, segment, '"')) {
            argv.push_back(segment);
        }
    }
    for(string x:argv){
        
    }
    return argv;
}

string generate(vector<string> argv){
    stringstream ss;
    string prompt = argv[1];
    int numArgs = argv.size();
    const string errMsg = "Incorrect usage of generate. Ex: generate \"hello world\"";
    //checks if command is valid
    if(numArgs<2){return errMsg;}
    //generates response
    if(argv.size()>2){
            ollama::options options;
            vector<ollama::image> attatchments;

            for(int i = 2; i<argv.size(); i++){
                cout<< "loading: " << argv[i]<<endl;
                try{
                    attatchments.push_back(ollama::image::from_file(argv[i]));
                }catch(...){
                    return "Failed to load image";
                }
            }

            ollama::images promptImages = attatchments;
            ss << ollama::generate("gemma3:4b", prompt, options, promptImages) << endl;
    }else{
        cout << "generating: " << prompt << endl;
        ss << ollama::generate("gemma3:4b", prompt) << endl;
    }
    return ss.str();
}

int save(vector<string> argv){
    string filename = "response.txt";
    if(argv.size() == 2) {
        filename = argv[1];
    }
    string file_content = Log::getInstance()->getLastMessage();
    ofstream out(filename);
    out << file_content;
    return 0;
}

string eval(vector<string> argv){
    if(argv.size()==0) return "";
    string command = argv[0];
    string output;
    if(command == "generate"){

        output = generate(argv);
        cout << output << endl;
        return output;
    }
    if(command == "save"){
        save(argv);
        return "saved";
    }

    output = "invalid command";
    cout << output << endl;
    return output;
}


/*int main() {
    ollama::options options;
    ollama::setReadTimeout(10800);
    /*vector<ollama::image> img;
    img.push_back(ollama::image::from_file("earth.jpg"));
    img.push_back(ollama::image::from_file("mars.png"));
    ollama::images promptImages = img;
    cout << ollama::generate("gemma3:4b", "what are these planets", options, promptImages);*//*
    std::cout << "Ollama_CLI>";
    string command;
    while (true)
    {
        getline(cin,command);
        vector<string> argv = parseLine(command);
        string response = eval(argv);
        
        if(argv.size()>0 && argv[0] == "quit"){break;}

        //cout << argv[2] << endl;
        Log::getInstance()->insert("User: " + command);
        Log::getInstance()->insert(response);
        std::cout << "Ollama_CLI>";
    }
    return 0;
}*/

