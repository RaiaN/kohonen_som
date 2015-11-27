using System;

namespace KohonenMap
{
    //Класс для представления карты Кохонена
    class Lattice
    {
        //передаем размер сетки и число характеристик
        public Lattice(int size, int featuresCnt)
        {
            featuresCnt_ = featuresCnt;
            size_ = size;
            width_ = Math.Min(size_, size_) / 2;
            Random r = new Random();

            lattice_ = new Neuron[size_, size_];
            for (int rowInd = 0; rowInd < size_; ++rowInd)
            {
                for (int colInd = 0; colInd < size_; ++colInd)
                {
                    //заполняем карту нейронами
                    lattice_[rowInd, colInd] = new Neuron(featuresCnt, r);
                }
            }
        }

        public int getFeaturesCnt()
        {
            return featuresCnt_;
        }

        //размер карты (как ширина, так и высота)
        public int getSize()
        {
            return size_;
        }

        //начальный размер окрестности нейрона
        public int getWidth()
        {
            return width_;
        }

        //получить нейрон по его координатам
        public Neuron get(int x, int y)
        {
            return lattice_[x, y];
        }

        //получить нейрон по его координатам (перегрузка)
        public Neuron get(Tuple<int, int> coord)
        {
            return lattice_[coord.Item1, coord.Item2];
        }

        private int featuresCnt_;
        private int size_;
        private int width_;
        private Neuron[,] lattice_;
    }
}
