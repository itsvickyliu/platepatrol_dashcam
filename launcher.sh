#!/bin/bash

source /home/platepatrol/dev/bin/activate
cd /home/platepatrol/Desktop/PlatePatrol
sudo gpsd /dev/ttyACM0 -F /var/run/gpsd.sock
python3 endtoend.py
# sudo shutdown -h now