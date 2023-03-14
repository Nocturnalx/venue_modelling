nvcc digest.cu -o vmd -I/usr/include/cppconn -L/usr/lib -lmysqlcppconn

sudo mv vmd /bin/vmd