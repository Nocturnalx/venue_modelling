echo "deleting /etc/venue_modelling"
sudo rm -r /etc/venue_modelling

echo "deleting /bin/vmd"
sudo rm /bin/vmd

echo "reseting + restarting nginx"
sudo rm /etc/nginx/sites-enabled/node.conf
sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl restart nginx

echo "Remember to delete pm2 backgrond processes if not done already and remove firewall rules using ufw!"