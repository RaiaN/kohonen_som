using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenMap
{
    class NeuronDict : SortedDictionary<Tuple<int, int>, List<int>>
    { }
    
    class BatchAlgorithm : Algorithm
    {
        public BatchAlgorithm(int latticeSize, int featuresCnt) : 
            base(latticeSize, featuresCnt, 
                 epochsNumber: 1000, learningRate: 0.1, tau2: 1000, trainingError: 1e-3) { }
        
        public void run(List<List<double>> inputData, List<List<int>> outputVectors)
        {
            int inputDataLen = inputData.Count();
            List<List<double>> winnerWeights = initWinnerWeights(inputDataLen);

            List<int>[,] labels;
            Utility.initLabels(out labels, latticeSize_);
            
            double[,] latticeDist = precalcLatticeDist();

            int time = 0;
            double currError = 1e3;
            double prevError = 0.0;            
            while (Math.Abs(currError - prevError) > trainingError_)
            {
                Utility.clearLabels(labels, latticeSize_);
                for (int inputInd = 0; inputInd < inputData.Count(); ++inputInd)
                {
                    List<double> singleObservation = inputData[inputInd];
                    double[,] distances = Utility.calcProximityMetrics(singleObservation, neurons_);                    
                    Tuple<int, int> neuronWinnerCoord = Utility.findClosestNeuron(distances, latticeSize_);                    
                    winnerWeights[inputInd] = neurons_.get(neuronWinnerCoord).weights_;
                    labels[neuronWinnerCoord.Item1, neuronWinnerCoord.Item2].Add(inputInd);
                }
                updateWeights(labels, inputData, latticeDist, time);
                
                prevError = currError;
                currError = Utility.calculateTrainingError(winnerWeights, inputData);
                ++time;
            }

            NeuronDict aliveNeurons = chooseAliveNeurons(labels);
            spreadOverPopularNeurons(inputData, labels, aliveNeurons, inputData);
            showInfo(aliveNeurons, outputVectors);           
            
            //store utilities
            //storeLatticeWeights("ba_weights_" + time.ToString() + ".txt");
            //currError = Utility.calculateTrainingError(ref winnerWeights, ref inputData);
            //Console.WriteLine("Training error AFTER exlusion dead nodes: " + currError);
            //Console.WriteLine("Epochs passed: " + time);
        }
        
        private double[,] precalcLatticeDist()
        {
            int neuronsNumber = latticeSize_ * latticeSize_;
            double[,] latticeDist = new double[neuronsNumber, neuronsNumber];

            for (int n1 = 0; n1 < neuronsNumber; ++n1)
            {
                for (int n2 = n1; n2 < neuronsNumber; ++n2)
                {
                    Tuple<int, int> x = Utility.neuronIndToCoord(n1, latticeSize_);
                    Tuple<int, int> y = Utility.neuronIndToCoord(n2, latticeSize_);
                    latticeDist[n1, n2] = Utility.calcLatticeDist(x, y);
                    latticeDist[n2, n1] = latticeDist[n1, n2];
                }
            }
            return latticeDist;
        }
        
        private List<double> calcMeanVector(List<int> labels, List<List<double>> inputData)
        {
            int labelsLen = labels.Count();
            List<double> meanVector = new List<double>(new double[featuresCnt_]);
            if (labelsLen != 0)
            {
                foreach (int label in labels)
                {
                    meanVector = meanVector.Zip(inputData[label], (x, y) => x + y).ToList();
                }
                meanVector = meanVector.Select(x => x / (1.0 * labelsLen)).ToList();
                return meanVector;
            }
            return meanVector;
        }
        
        private List<Tuple<int, int>> getNeighbourhood(double[,] latticeDist, int neuronNumber, double delta)
        {
            int neuronsNumber = latticeSize_ * latticeSize_;
            List<Tuple<int, int>> nb = new List<Tuple<int, int>>();
            for (int ni = 0; ni < neuronsNumber; ++ni)
            {
                if (latticeDist[neuronNumber, ni] <= delta)
                {
                    nb.Add(Utility.neuronIndToCoord(ni, latticeSize_));
                }                
            }
            return nb;
        }
        
        private List<double> calculateUpdatedWeights(double[,] latticeDist, 
                                                     List<int>[,] labels, 
                                                     List<List<double>> inputData, 
                                                     int neuronNumber, Tuple<int, int> currNeuronCoord, int time)
        {
            List<double> result = new List<double>(new double[featuresCnt_]);
            double denominator = 0.0;

            double delta = width_ * Math.Exp(-time / tau1_);            
            List<Tuple<int, int>> nb = getNeighbourhood(latticeDist, neuronNumber, delta);
            
            double denominatorPart = 0.0;
            foreach (Tuple<int, int> n in nb)
            {
                int x = n.Item1;
                int y = n.Item2;
                int nInd = x * latticeSize_ + y;
                List<int> currLabels = labels[x, y];
                
                double updateCoef = calcUpdateCoef(Utility.neuronIndToCoord(nInd, latticeSize_), currNeuronCoord, time) * currLabels.Count();                
                List<double> currNumerator = calcMeanVector(currLabels, inputData).Select(val => val * updateCoef).ToList();

                result = result.Zip(currNumerator, (v1, v2) => v1 + v2).ToList();
                denominatorPart += updateCoef;
            }
            denominator += denominatorPart;
            
            if (Math.Abs(denominator) > 0)
            {
                result = result.Select(x => x / (denominator * 1.0)).ToList();
            }
            return result;
        }
        
        private void updateWeights(List<int>[,] labels, 
                                   List<List<double>> inputData, 
                                   double[,] latticeDist,
                                   int time)
        {           
            for (int rowInd = 0; rowInd < latticeSize_; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize_; ++colInd)
                {
                    int neuronNumber = rowInd * latticeSize_ + colInd;
                    Tuple<int, int> currNeuronCoord = Utility.neuronIndToCoord(neuronNumber, latticeSize_);                    
                    neurons_.get(rowInd, colInd).weights_ = calculateUpdatedWeights(latticeDist, labels, inputData, 
                                                                                    neuronNumber, currNeuronCoord, time);
                }
            }
        }

        private NeuronDict chooseAliveNeurons(List<int>[,] labels)
        {
            NeuronDict aliveNeurons = new NeuronDict();
            for (int x = 0; x < latticeSize_; ++x)
            {
                for (int y = 0; y < latticeSize_; ++y)
                {
                    List<int> ls = labels[x, y];
                    int labelsLen = ls.Count();

                    if (labelsLen > 2)
                    {
                        aliveNeurons[new Tuple<int, int>(x, y)] = labels[x, y].Select(val => val).ToList();
                        labels[x, y].Clear();
                    }
                }
            }
            return aliveNeurons;
        }

        private void spreadOverPopularNeurons(List<List<double>> inputData, 
                                              List<int>[,] labels, 
                                              NeuronDict aliveNeurons,
                                              List<List<double>> winnerWeights)
        {
            for (int x = 0; x < latticeSize_; ++x)
            {
                for (int y = 0; y < latticeSize_; ++y)
                {
                    List<int> ls = labels[x, y];
                    int labelsLen = ls.Count();

                    if (labelsLen > 0 && labelsLen <= 2) //just check "introvert" neurons
                    {
                        foreach (int label in ls)
                        {
                            List<double> singleObservation = inputData[label];
                            double[,] distances = Utility.calcProximityMetrics(singleObservation, neurons_);
                            Tuple<int, int> closestPopularNeuron = Utility.findClosestPopularNeuron(distances, 
                                                                                                    aliveNeurons, 
                                                                                                    latticeSize_);
                            aliveNeurons[closestPopularNeuron].Add(label);
                            winnerWeights[label] = neurons_.get(closestPopularNeuron).weights_;
                        }
                    }
                }
            }
        }

        private void showInfo(NeuronDict aliveNeurons, List<List<int>> outputVectors)
        {
            List<KeyValuePair<Tuple<int, int>, List<int>>> temp = aliveNeurons.ToList();
            temp.Sort((x, y) =>
            {
                return -1 * x.Value.Count().CompareTo(y.Value.Count());
            });

            Console.WriteLine("Top 3 neuron weights by popularity:");
            for (int pind = 0; pind < 3; ++pind)
            {
                Console.WriteLine(String.Join(" ", neurons_.get(temp[pind].Key).weights_));

                int classesCount = outputVectors[pind].Count();
                List<double> classes = new List<double>(new double[classesCount]);
                foreach (int inputInd in temp[pind].Value)
                {
                    classes = classes.Zip(outputVectors[inputInd], (x, y) => (x + y)).ToList();
                }
                classes = classes.Select(x => 100.0 * x / (1.0 * temp[pind].Value.Count())).ToList();

                Console.WriteLine("Class representation: " + String.Join("% ", classes) + "%\n");
            }
        }
    }
}
