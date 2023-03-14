echo "installing npm packages"
npm install express
npm install multer
npm install mysql
npm install body-parser

echo "setting up nginx"
sudo rm /etc/nginx/sites-enabled/default

sudo cp conf/node.conf /etc/nginx/sites-available
sudo ln -s /etc/nginx/sites-available/node.conf /etc/nginx/sites-enabled/

sudo nginx -t
echo "restarting nginx"
sudo systemctl restart nginx

sudo ufw allow 80
sudo ufw allow 443

echo "creating /etc/venue_modeling directory"
sudo mkdir -p /etc/venue_modelling/digest/in
sudo mkdir /etc/venue_modelling/digest/out
sudo mkdir /etc/venue_modelling/digest/temp

sudo cp uninstall.sh /etc/venue_modelling

echo "compiling"
nvcc digest/digest.cu -o digest/vmd -I/usr/include/cppconn -L/usr/lib -lmysqlcppconn
echo "vmd added to /bin/"
sudo mv digest/vmd /bin/vmd
