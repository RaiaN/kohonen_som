using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenMap
{
    //Итеративный тип обучения
    class IterativeAlgorithm : Algorithm
    {
        public IterativeAlgorithm(int latticeSize, int featuresCnt) : 
            base(latticeSize, featuresCnt, 
                 epochsNumber: 1000, learningRate: 0.1, tau2: 1000, trainingError: 1e-3) {}

        //основная процедура
        public void run(ref List<List<double>> inputData, ref List<List<int>> outputVectors)
        {
            int inputDataLen = inputData.Count();
            List<List<double>> winnerWeights = initWinnerWeights(inputDataLen);

            List<int>[,] labels;
            Utility.initLabels(out labels, latticeSize_);

            int time = 0;
            double currError = 1e3;
            double prevError = 0.0;

            //итерационный процесс
            while (Math.Abs(currError - prevError) > trainingError_)
            {
                Utility.clearLabels(ref labels, latticeSize_);
                for (int inputInd = 0; inputInd < inputData.Count(); ++inputInd)
                {
                    List<double> singleObservation = inputData[inputInd];

                    //находим нейрона-победителя по метрике
                    double[,] distances = Utility.calcProximityMetrics(ref singleObservation, ref neurons_);
                    Tuple<int, int> neuronWinnerCoord = Utility.findClosestNeuron(ref distances, latticeSize_);
                    
                    //Сохраняем веса для удобства вычисления ошибки представления
                    winnerWeights[inputInd] = neurons_.get(neuronWinnerCoord).weights_;

                    //обновляем веса
                    updateWeights(ref singleObservation, ref neuronWinnerCoord, time);       
                    
                    //относим наблюдение к нейрону-победителю
                    labels[neuronWinnerCoord.Item1, neuronWinnerCoord.Item2].Add(inputInd);
                }
                prevError = currError;
                currError = Utility.calculateTrainingError(ref winnerWeights, ref inputData);
                //Console.WriteLine("Delta error: " + Math.Abs(currError - prevError).ToString());
                ++time;
            }
            //storeLatticeWeights("ia_weights_" + time.ToString() + ".txt");
            //Console.WriteLine("Training error: " + currError);
            //Console.WriteLine("Epochs passed: " + time);
        }

        //обновляем веса
        private void updateWeights(ref List<double> input, ref Tuple<int, int> neuronWinnerCoord, int time)
        {
            int latticeSize = neurons_.getSize();

            for (int rowInd = 0; rowInd < latticeSize; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize; ++colInd)
                {
                    List<double> neuronWeights = neurons_.get(rowInd, colInd).weights_;
                    List<double> diff = Utility.vectorDifference(ref input, ref neuronWeights);

                    //вычисляем коэффициент обновления 
                    double coef = calcUpdateCoef(neuronWinnerCoord, rowInd, colInd, time);

                    //по формуле обновляем сами веса
                    diff = diff.Select(x => x * (-coef)).ToList();
                    List<double> updatedWeights = Utility.vectorDifference(ref neuronWeights, ref diff); //value dependens on argument order

                    //сохраняем обновленные веса
                    neurons_.get(rowInd, colInd).weights_ = updatedWeights;
                }
            }
        }
    }
}
