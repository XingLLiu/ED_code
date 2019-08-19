library(data.table)
library(ff)

BLOBS <- fread("/home/lebo/mybigdata/data/BLOBcsv1.csv", nrows=1)

BLOBS <- read.csv.ffdf(file="/home/lebo/mybigdata/data/BLOBcsv1.csv", nrows=10)
dim(BLOBS)
colnames(BLOBS)
BLOBS[1:2,]
