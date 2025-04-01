#!/bin/bash

source /home/platepatrol/dev/bin/activate
cd /home/platepatrol/Desktop/PlatePatrol
sudo systemctl stop gpsd.socket
sudo systemctl stop gpsd.service
sudo systemctl disable gpsd.socket
sudo gpsd /dev/ttyACM0 -F /var/run/gpsd.sock
python3 endtoend.py
# sudo shutdown -h now