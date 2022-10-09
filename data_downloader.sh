mkdir -p datasets
cd datasets

# house price
wget -O house_data_precessed.csv https://cloud.tsinghua.edu.cn/f/5bdbc173cc7e4ebf8886/?dl=1 --no-check-certificate 

# CelebA
mkdir CelebA
cd CelebA 
wget -O test_40000_0.999_0.8_20000_0.01_0.2_0.8_0.999.csv https://cloud.tsinghua.edu.cn/f/8e09c33ae8bd450abb83/?dl=1 --no-check-certificate
wget -O train_40000_0.999_0.8_20000_0.01_0.2_0.8_0.999.csv https://cloud.tsinghua.edu.cn/f/eb856022aa184c43b189/?dl=1 --no-check-certificate
cd ..

# landcover
wget -O landcover.pkl https://cloud.tsinghua.edu.cn/f/4ed9cb7399cf47329c1e/?dl=1 --no-check-certificate 



