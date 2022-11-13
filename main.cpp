#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "neural_network.h"
using namespace std;


int main()
{

	vector<int> topology;

	topology.push_back(2);
	topology.push_back(4);
	topology.push_back(1);
	NeuralNetwork network(topology);

	vector<double> inputVals, targetVals, resultVals;

	inputVals.push_back(1.0);
	inputVals.push_back(0.0);


	targetVals.push_back(3.0);

	network.ForwardPropagate(inputVals);

	network.GetResults(resultVals);

	network.BackPropagate(targetVals);

	cout << resultVals[0] << endl;
	cout << "reached here " << endl;

	


	return 0;
}

