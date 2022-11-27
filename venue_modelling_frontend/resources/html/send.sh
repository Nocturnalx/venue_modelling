read -p "send html file?"
scp $1 pi@192.168.0.10:~/Documents/repos/venue_modelling_frontend/resources/html
echo "sent to html!"