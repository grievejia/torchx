#!/usr/bin/env python3

import socket
import os

HOST = os.environ["VC_SH_0_HOSTS"]  # Standard loopback interface address (localhost)
PORT = 30001  # Port to listen on (non-privileged ports are > 1023)

print("listening on ", HOST,PORT)
print(socket.getfqdn(socket.gethostname()))

