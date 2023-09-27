echo "installing npm packages" 
npm install --prefix ../ express multer mysqlmysql body-parser

echo ""
echo "setting up nginx"
sudo rm /etc/nginx/sites-enabled/default

sudo cp conf/node.conf /etc/nginx/sites-available
sudo ln -s /etc/nginx/sites-available/node.conf /etc/nginx/sites-enabled/

sudo nginx -t
echo ""
echo "restarting nginx"
sudo systemctl restart nginx

echo "opening ports"
sudo ufw allow 80
sudo ufw allow 443

echo ""
echo "creating /var/venue_modeling directory"
sudo mkdir -p ~/.venue_modelling/digest/in
sudo mkdir ~/.venue_modelling/digest/out
sudo mkdir ~/.var/venue_modelling/digest/temp

sudo cp uninstall.sh ~/.venue_modelling

echo ""
echo "compiling"
# nvcc digest/digest.cu -o digest/vmd -I/usr/include/cppconn -L/usr/lib -lmysqlcppconn
cmake ../digest
make

echo "adding vmd to bin"
sudo mv digest/build/vmd /bin/vmd
