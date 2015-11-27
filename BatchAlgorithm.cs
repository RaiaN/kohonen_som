using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenMap
{
    //Пакетный тип обучения
    class BatchAlgorithm : Algorithm
    {
        public BatchAlgorithm(int latticeSize, int featuresCnt) : 
            base(latticeSize, featuresCnt, 
                 epochsNumber:1000, learningRate:0.1, tau2:1000, trainingError:1e-3) { }

        //основная процедура
        public void run(ref List<List<double>> inputData, ref List<List<int>> outputVectors)
        {
            int inputDataLen = inputData.Count();
            List<List<double>> winnerWeights = initWinnerWeights(inputDataLen);

            List<int>[,] labels;
            Utility.initLabels(out labels, latticeSize_);

            //предподсчёт дистанций между всеми парами нейронов
            double[,] latticeDist = calcLatticeDist();

            int time = 0;
            double currError = 1e3;
            double prevError = 0.0;

            //Пока изменение ошибки больше trainingError итерации продолжаются
            while (Math.Abs(currError - prevError) > trainingError_)
            {
                Utility.clearLabels(ref labels, latticeSize_);
                //находим нейрона-победителя для каждого наблюдения
                for (int inputInd = 0; inputInd < inputData.Count(); ++inputInd)
                {
                    List<double> singleObservation = inputData[inputInd];
                    //считаем расстояния от всех нейронов до налблюдения
                    double[,] distances = Utility.calcProximityMetrics(ref singleObservation, ref neurons_);

                    //используем расстояния, чтобы найти нейрона-победителя
                    Tuple<int, int> neuronWinnerCoord = Utility.findClosestNeuron(ref distances, latticeSize_);

                    //сохраняем информацию о том, к какому нейрону относится наблюдение, и веса нейрона-победителя
                    winnerWeights[inputInd] = neurons_.get(neuronWinnerCoord).weights_;
                    labels[neuronWinnerCoord.Item1, neuronWinnerCoord.Item2].Add(inputInd);
                }
                //обновляем веса
                updateWeights(ref labels, ref inputData, ref latticeDist, time);

                //считаем ошибку представления
                prevError = currError;
                currError = Utility.calculateTrainingError(ref winnerWeights, ref inputData);
                //Console.WriteLine("Delta error: " + Math.Abs(currError - prevError).ToString());
                ++time;
            }

            //Console.WriteLine("Training error BEFORE exlusion dead nodes: " + currError);

            //берем из всех нейронов после обучения только живые
            SortedDictionary<Tuple<int, int>, List<int>> aliveNeurons = new SortedDictionary<Tuple<int, int>, List<int>>();
            for (int x = 0; x < latticeSize_; ++x)
            {
                for (int y = 0; y < latticeSize_; ++y)
                {
                    List<int> ls = labels[x, y];
                    int labelsLen = ls.Count();

                    if (labelsLen > 2)
                    {
                        //сохраняем живой нейрон в отдельную структуру
                        aliveNeurons[new Tuple<int, int>(x, y)] = labels[x, y].Select(val => val).ToList();
                        labels[x, y].Clear();
                    }
                }                    
            }

            //Наблюдения нейронов, которые содержат их всего 1-2, перекидываем на более популярные
            for (int x = 0; x < latticeSize_; ++x)
            {
                for (int y = 0; y < latticeSize_; ++y)
                {
                    List<int> ls = labels[x, y];
                    int labelsLen = ls.Count();

                    if (labelsLen > 0 && labelsLen <= 2)
                    {
                        foreach (int label in ls)
                        {
                            List<double> singleObservation = inputData[label];

                            //ищем живой нейрон
                            double[,] distances = Utility.calcProximityMetrics(ref singleObservation, ref neurons_);
                            Tuple<int, int> closestPopularNeuron = Utility.findClosestPopularNeuron(ref distances, ref aliveNeurons, latticeSize_);

                            //перебрасываем наблюдение к живому нейрону
                            aliveNeurons[closestPopularNeuron].Add(label);
                            //сохраняем веса
                            winnerWeights[label] = neurons_.get(closestPopularNeuron).weights_;
                        }          
                    }
                }
            }

            //сортируем нейроны по количеству наблюдений, присвоенных им
            List<KeyValuePair<Tuple<int, int>, List<int>>> temp = aliveNeurons.ToList();
            temp.Sort((x, y) =>
            {
                return -1 * x.Value.Count().CompareTo(y.Value.Count());
            });

            //выводим веса 3-х наиболее популярных нейронов, информацию о представлении классов в этих нейронах
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

            //можно посчитать дополнительно ошибку представления после распределения наблюдений по живым нейронам
            //storeLatticeWeights("ba_weights_" + time.ToString() + ".txt");
            //currError = Utility.calculateTrainingError(ref winnerWeights, ref inputData);
            //Console.WriteLine("Training error AFTER exlusion dead nodes: " + currError);
            //Console.WriteLine("Epochs passed: " + time);
        }

        //считаем все попарные расстояния между нейронами в карте Кохонена
        private double[,] calcLatticeDist()
        {
            int neuronsNumber = latticeSize_ * latticeSize_;
            double[,] latticeDist = new double[neuronsNumber, neuronsNumber];
            for (int n1 = 0; n1 < neuronsNumber; ++n1)
            {
                for (int n2 = n1; n2 < neuronsNumber; ++n2)
                {
                    Tuple<int, int> coord = new Tuple<int, int>((int)n1 / latticeSize_, n1 % latticeSize_);
                    latticeDist[n1, n2] = Utility.calcLatticeDist(coord, (int)n2 / latticeSize_, n2 % latticeSize_);
                    latticeDist[n2, n1] = latticeDist[n1, n2];
                }
            }
            return latticeDist;
        }

        //Вычисляем средний вектор по наблюдениям, отнесенных к текущему нейрону победителю
        private List<double> calcMeanVector(ref List<int> labels, ref List<List<double>> inputData)
        {
            int labelsLen = labels.Count();
            List<double> meanVector = new List<double>(new double[featuresCnt_]);
            meanVector = meanVector.Select(x => 0.0).ToList();
            if (labelsLen == 0)
            {
                return meanVector;
            }
          
            foreach (int label in labels)
            {
                meanVector = meanVector.Zip(inputData[label], (x, y) => x + y).ToList();
            }
            meanVector = meanVector.Select(x => x / (1.0 * labelsLen)).ToList();
            return meanVector;
        }

        //ищем соседей нейрона, веса которого обновляем
        private List<Tuple<int, int>> getNeighbourhood(ref double[,] latticeDist, int neuronNumber, double delta)
        {
            int neuronsNumber = latticeSize_ * latticeSize_;
            List<Tuple<int, int>> nb = new List<Tuple<int, int>>();
            for (int ni = 0; ni < neuronsNumber; ++ni)
            {
                if (latticeDist[neuronNumber, ni] <= delta)
                {
                    nb.Add(new Tuple<int, int>((int)ni / latticeSize_, ni % latticeSize_));
                }                
            }
            return nb;
        }

        //вычисляем новые веса
        private void calculateUpdatedWeights(ref double[,] latticeDist, 
                                             ref List<int>[,] labels, 
                                             ref List<List<double>> inputData, 
                                             int neuronNumber, int time, int rowInd, int colInd,
                                             ref List<double> numerator, ref double denominator)
        {
            double delta = width_ * Math.Exp(-time / tau1_);
            
            //ищем соседей
            List<Tuple<int, int>> nb = getNeighbourhood(ref latticeDist, neuronNumber, delta);

            //считаем числитель и знаменатель по формуле
            double denominatorPart = 0.0;
            foreach (Tuple<int, int> n in nb)
            {
                int x = n.Item1;
                int y = n.Item2;
                List<int> currLabels = labels[x, y];

                //вычисляем коэффициент перед "средним" наблюдением
                double updateCoef = calcUpdateCoef(new Tuple<int, int>(x, y), rowInd, colInd, time) * currLabels.Count();

                //вычисляем "среднее" наблюдение
                List<double> currNumerator = calcMeanVector(ref currLabels, ref inputData).Select(val => val * updateCoef).ToList();

                numerator = numerator.Zip(currNumerator, (v1, v2) => v1 + v2).ToList();
                denominatorPart += updateCoef;
            }
            denominator += denominatorPart;

            //если знаменатель 0, то веса не изменяются
            if (denominator != 0)
            {
                //сохраняем обновленные веса
                neurons_.get(rowInd, colInd).weights_ = numerator.Select(x => x / (denominatorPart * 1.0)).ToList();
            }
        }

        //обновить веса всех нейронов
        private void updateWeights(ref List<int>[,] labels, 
                                   ref List<List<double>> inputData, 
                                   ref double[,] latticeDist,
                                   int time)
        {
           
            for (int rowInd = 0; rowInd < latticeSize_; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize_; ++colInd)
                {                                        
                    int neuronNumber = rowInd * latticeSize_ + colInd;
                    List<double> numerator = new List<double>(new double[featuresCnt_]);
                    numerator = numerator.Select(x => 0.0).ToList();
                    double denominator = 0.0;

                    calculateUpdatedWeights(ref latticeDist, ref labels, ref inputData, neuronNumber, time, rowInd, colInd,
                                            ref numerator, ref denominator);
                }
            }
        }
    }
}
