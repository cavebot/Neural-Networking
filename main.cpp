#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>


#include "neural_network.h"
using namespace std;


int main()
{

	

	//srand(time(NULL));

	//Produce random training data file (training the network to produce logical XOR)
	ofstream TrainingData("trainingData.txt");
	//TrainingData << "2" << "	" << "4" << "	" << "1" << endl;
	for (int i = 0; i < 5000; i++)
	{
		int input1 = (int)(2.0 * rand() / double(RAND_MAX));
		int input2 = (int)(2.0 * rand() / double(RAND_MAX));
		int output = input1 ^ input2;
		TrainingData << input1 << "	" << input2 << "	" << output << endl;

	}
	TrainingData.close();


	
	

	//Read training data file and line by line and pass into network and backpro
	string myText;
	ifstream Data("trainingData.txt");

	double input1, input2, target1;

	//create 2-4-1 network
	vector<int> topology;
	topology.push_back(2);
	topology.push_back(2);
	topology.push_back(1);

	NeuralNetwork network(topology);

	vector<double> inputVals, targetVals, resultVals;

	while (Data >> input1 >> input2 >> target1)
	{

		inputVals.clear();
		targetVals.clear();
		resultVals.clear();
		inputVals.push_back(input1);
		inputVals.push_back(input2);
		targetVals.push_back(target1);
		
		network.ForwardPropagate(inputVals);
		network.BackPropagate(targetVals);
		network.GetResults(resultVals);

		cout << input1 << " " << input2 << " " << target1 << " " << resultVals[0] << endl;

	}
	


	Data.close();
	


	
	


	cout << resultVals[0] << endl;
	cout << "reached here " << endl;

	


	return 0;
}

