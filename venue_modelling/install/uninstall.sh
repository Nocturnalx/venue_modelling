
echo ""
echo "deleting /bin/vmd"
sudo rm /bin/vmd

echo ""
echo "reseting + restarting nginx"
sudo rm /etc/nginx/sites-enabled/node.conf
sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl restart nginx

echo ""
echo "Remember to delete pm2 backgrond processes if not done already and remove firewall rules using ufw!"

echo "deleting /var/venue_modelling"
rm -r ~/.venue_modelling