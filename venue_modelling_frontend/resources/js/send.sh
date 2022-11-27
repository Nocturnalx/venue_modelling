read -p "send js file?"
scp $1 pi@192.168.0.10:~/Documents/repos/venue_modelling_frontend/resources/js
echo "sent to js!"