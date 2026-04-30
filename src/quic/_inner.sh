#!/bin/bash
quiche-client --no-verify --wire-version 00000001 https://10.0.0.1/10MB.zip > /dev/null && echo DOWNLOAD_OK
