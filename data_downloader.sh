mkdir -p datasets
cd datasets

# house price
wget https://cloud.tsinghua.edu.cn/lib/bb2fe96f-0339-4c08-8885-01de9eb037a6/file/house_data_precessed.csv  --no-check-certificate 

# landcover
wget https://cloud.tsinghua.edu.cn/lib/bb2fe96f-0339-4c08-8885-01de9eb037a6/file/landcover_data.pkl --no-check-certificate 


# CelebA
mkdir CelebA
cd CelebA 
wget -O test_40000_0.999_0.8_20000_0.01_0.2_0.8_0.999.csv https://cloud.tsinghua.edu.cn/f/8e09c33ae8bd450abb83/?dl=1 --no-check-certificate
wget -O train_40000_0.999_0.8_20000_0.01_0.2_0.8_0.999.csv https://cloud.tsinghua.edu.cn/f/eb856022aa184c43b189/?dl=1 --no-check-certificate


