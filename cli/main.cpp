#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "ollama.hpp"

using namespace std;

vector<string> parseLine(string line){
    string arg;
    string segment;
    stringstream ss(line);
    vector<string> argv;

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



void  eval(vector<string> argv){
    stringstream ss;
    if(argv[0] == "generate"){
        //checks if command is valid
        //if(argv.size()<2){return;}
        //if(!(argv[1][0] == '"' && argv[1].back() == '"')){return;}
        //generates response
        if(argv.size()>2){
            ollama::options options;
            vector<ollama::image> img;
            for(int i = 2; i<argv.size(); i++){
                cout<< "loading: " << argv[i]<<endl;
                img.push_back(ollama::image::from_file(argv[i]));
            }
            ollama::images promptImages = img;
            cout << ollama::generate("gemma3:4b", argv[1], options, promptImages) << endl;
        }else{
            cout << "generating: " << argv[1] << endl;
            cout << ollama::generate("gemma3:4b", argv[1]) << endl;
        }

    }
    return;
}


int main() {
    ollama::options options;
    /*vector<ollama::image> img;
    img.push_back(ollama::image::from_file("earth.jpg"));
    img.push_back(ollama::image::from_file("mars.png"));
    ollama::images promptImages = img;
    cout << ollama::generate("gemma3:4b", "what are these planets", options, promptImages);*/
    std::cout << "Ollama_CLI>";
    string command;
    while (true)
    {
        getline(cin,command);
        vector<string> argv = parseLine(command);

        if(argv[0] == "quit"){
            break;
        }
        //cout << argv[2] << endl;
        eval(argv);
        std::cout << "Ollama_CLI>";
    }
    return 0;
}

