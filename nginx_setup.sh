sudo rm /etc/nginx/sites-enabled/default
sudo cp node.conf /etc/nginx/sites-available
sudo ln -s /etc/nginx/sites-available/node.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo ufw allow 80
sudo ufw allow 443