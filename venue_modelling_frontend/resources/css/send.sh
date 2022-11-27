read -p "send css file?"
scp $1 pi@192.168.0.10:~/Documents/repos/venue_modelling_frontend/resources/css
echo "sent to css!"