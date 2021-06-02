#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main(){
    fstream newfile;
    newfile.open("test.txt",ios::in); //open a file to perform read operation using file object
    if (newfile.is_open()){   //checking whether the file is open
        float tp;
        while(getline(newfile, tp)) //read data from file object and put it into string.
            cout << "line: " << tp << "\n"; //print the data of the string
    }     
    return 0;
}