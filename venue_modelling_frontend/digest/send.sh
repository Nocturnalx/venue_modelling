read -p "send cpp file?"
scp $1 pi@192.168.0.10:~/Documents/repos/venue_modelling_frontend/digest
echo "sent to digest!"
