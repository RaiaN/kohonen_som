using System;

namespace KohonenMap
{
    class Lattice
    {
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
                    lattice_[rowInd, colInd] = new Neuron(featuresCnt, r);
                }
            }
        }

        public int getFeaturesCnt()
        {
            return featuresCnt_;
        }
        
        public int getSize()
        {
            return size_;
        }
        
        public int getWidth()
        {
            return width_;
        }
        
        public Neuron get(int x, int y)
        {
            return lattice_[x, y];
        }
        
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
