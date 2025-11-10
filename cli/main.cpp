#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <unistd.h>
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

 map<string, string> extractFlags(vector<string> argv){
    // we can start with --model, --temperature, and --num_predict
    map<string, string> flags;
    ollama::options options;

    const vector<string> validFlags = {
        "temperature",
        "num_predict",
        "num_ctx",
        "model"
    };

    for(int i = 0; i < argv.size(); i++){
        string current_arg = argv[i];

        if(current_arg.rfind("--", 0) == 0) {
            string flag = current_arg.substr(2);

            // simple check to see if the flag is valid
            if (find(validFlags.begin(), validFlags.end(), flag) != validFlags.end()){
                //find the corresponding value for the flag
                if (i + 1 < argv.size() && argv[i + 1].rfind("--", 0) != 0) {
                    flags[flag] = argv[i + 1];
                    //skip iterating through the value we already extracted 
                    i++;
                }
            }
            else{
                cout << "Unrecognized flag '--" << flag << "' will be ignored" << endl;
            }
        }

    }

    return flags;

}

bool endsWith(const string& fullString, const string& ending){
    if (ending.size() > fullString.size())
        return false;
    return fullString.compare(fullString.size() - ending.size(), ending.size(), ending) == 0;

}

string generate(vector<string> argv){
    stringstream ss;
    string prompt = argv[1];
    string model = "gemma3:4b";
    int numArgs = argv.size();
    const string errMsg = "Incorrect usage of generate. Ex: generate \"hello world\"";
    //checks if command is valid
    if(numArgs<2){return errMsg;}
    //generates response


    map<string, string> flags = extractFlags(argv);
    //set model if provided
    if (flags.find("model") != flags.end()){
        model = flags["model"];
    }

    if(argv.size()>2){

            ollama::options options;
            vector<ollama::image> attachments;

            // sample flags, we can add more and probably do this in a function as we go when we want to deal with more optimization stuff
            // temperature basically controls how creative the model's output is
            // num_predict can be tuned to mess around with the length of the model's output
            // num_ctx can adjust the context window size
            // prompt flag?
            if (flags.find("temperature") != flags.end()){
                options["temperature"] = stof(flags["temperature"]);
            }
            if (flags.find("num_predict") != flags.end()){
                options["num_predict"] = stoi(flags["num_predict"]);
            }
            if (flags.find("num_ctx") != flags.end()){
                options["num_ctx"] = stoi(flags["num_ctx"]);
            }

            for(int i = 2; i<argv.size(); i++){
                // doing this to account for flag usage
                if (endsWith(argv[i], ".png") || endsWith(argv[i], ".jpg") || endsWith(argv[i], ".jpeg")){
                    cout<< "loading: " << argv[i]<<endl;
                    try{
                        attachments.push_back(ollama::image::from_file(argv[i]));
                    }catch(...){
                        return "Failed to load image";
                    }
                }
                
            }

            ollama::images promptImages = attachments;
            
            ss << ollama::generate(model, prompt, options, promptImages) << endl;
    }else{
        cout << "generating: " << prompt << endl;
        ss << ollama::generate(model, prompt) << endl;
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

