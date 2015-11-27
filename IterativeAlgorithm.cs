using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenMap
{
    class IterativeAlgorithm : Algorithm
    {
        public IterativeAlgorithm(int latticeSize, int featuresCnt) : 
            base(latticeSize, featuresCnt, 
                 epochsNumber: 1000, learningRate: 0.1, tau2: 1000, trainingError: 1e-3) {}
        
        public void run(ref List<List<double>> inputData, ref List<List<int>> outputVectors)
        {
            int inputDataLen = inputData.Count();
            List<List<double>> winnerWeights = initWinnerWeights(inputDataLen);

            List<int>[,] labels;
            Utility.initLabels(out labels, latticeSize_);

            int time = 0;
            double currError = 1e3;
            double prevError = 0.0;
            
            while (Math.Abs(currError - prevError) > trainingError_)
            {
                Utility.clearLabels(ref labels, latticeSize_);
                for (int inputInd = 0; inputInd < inputData.Count(); ++inputInd)
                {
                    List<double> singleObservation = inputData[inputInd];
                    
                    double[,] distances = Utility.calcProximityMetrics(ref singleObservation, ref neurons_);
                    Tuple<int, int> neuronWinnerCoord = Utility.findClosestNeuron(ref distances, latticeSize_);                    
                    winnerWeights[inputInd] = neurons_.get(neuronWinnerCoord).weights_;              
                          
                    updateWeights(ref singleObservation, ref neuronWinnerCoord, time);
                    labels[neuronWinnerCoord.Item1, neuronWinnerCoord.Item2].Add(inputInd);
                }
                prevError = currError;
                currError = Utility.calculateTrainingError(ref winnerWeights, ref inputData);
                ++time;
            }
        }
        
        private void updateWeights(ref List<double> input, ref Tuple<int, int> neuronWinnerCoord, int time)
        {
            int latticeSize = neurons_.getSize();

            for (int rowInd = 0; rowInd < latticeSize; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize; ++colInd)
                {
                    List<double> neuronWeights = neurons_.get(rowInd, colInd).weights_;
                    List<double> diff = Utility.vectorDifference(ref input, ref neuronWeights);
                    
                    double coef = calcUpdateCoef(neuronWinnerCoord, rowInd, colInd, time);
                    
                    diff = diff.Select(x => x * (-coef)).ToList();
                    List<double> updatedWeights = Utility.vectorDifference(ref neuronWeights, ref diff); //value dependens on argument order
                    
                    neurons_.get(rowInd, colInd).weights_ = updatedWeights;
                }
            }
        }
    }
}
