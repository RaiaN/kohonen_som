using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System;

namespace KohonenMap
{
    //общий класс для удобства представления алгоритмов
    class Algorithm
    {
        //передаем в конструктор параметры алгоритма (всё по методичке)
        public Algorithm(int latticeSize, int featuresCnt, 
                         int epochsNumber = 1000, double learningRate = 0.01, int tau2 = 1000, double trainingError = 1e-3)
        {
            latticeSize_ = latticeSize;
            featuresCnt_ = featuresCnt;
            epochsNumber_ = epochsNumber;
            learningRate_ = learningRate;
            tau2_ = tau2;
            trainingError_ = trainingError;

            neurons_ = new Lattice(latticeSize, featuresCnt);
            width_ = neurons_.getWidth();
            tau1_ = epochsNumber_ / (width_ * 1.0);            
        }

        //инициализируем веса нейронов-победителей 
        protected List<List<double>> initWinnerWeights(int inputDataLen)
        {
            List<List<double>> winnerWeights = new List<List<double>>(inputDataLen);
            for (int inputInd = 0; inputInd < inputDataLen; ++inputInd)
            {
                winnerWeights.Add(new List<double>(new double[featuresCnt_]));
            }
            return winnerWeights;
        }

        //вычисляем коэффициент обновления 
        //1 - учитываем окрестность (уменьшается с течением времени) distCoef
        //2 - учитываем скорость обучения (learning rate), также уменьшается с течением времени time
        protected double calcUpdateCoef(Tuple<int, int> neuronWinnerCoord, int rowInd, int colInd, int time)
        {
            double latticeDist = Utility.calcLatticeDist(neuronWinnerCoord, rowInd, colInd);           
            double delta = width_ * Math.Exp(-time / tau1_);
            double distCoef = Math.Exp(-latticeDist * latticeDist / (2 * delta * delta));
            double learningRateCoef = learningRate_ * Math.Exp(-time / (tau2_ * 1.0));
            return learningRateCoef * distCoef;
        }

        //сохраняем веса в файл
        protected void storeLatticeWeights(string filename)
        {
            var culture = CultureInfo.InvariantCulture;
            using (System.IO.StreamWriter sw = new System.IO.StreamWriter("weights/" + filename))
            {
                for (int x = 0; x < latticeSize_; ++x)
                {
                    for (int y = 0; y < latticeSize_; ++y)
                    {
                        List<string> weights = neurons_.get(x, y).weights_.Select(weight => weight.ToString(culture)).ToList();
                        sw.WriteLine(string.Join(" ", weights));
                    }
                }
            }
        }

        protected int epochsNumber_;
        protected double learningRate_;
        protected double trainingError_;
        protected double tau1_;
        protected double tau2_;
        protected int width_;

        protected int latticeSize_;
        protected int featuresCnt_;
        protected Lattice neurons_;        
    }
}
