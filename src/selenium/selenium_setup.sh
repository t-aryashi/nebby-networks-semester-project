#!/bin/bash
# selenium_setup.sh — Install Selenium + Chrome for Nebby browser measurements
# Run once before using selenium_measure.py
# Paper reference: Nebby §3.5

echo "=== Installing Python dependencies ==="
pip install selenium webdriver-manager --break-system-packages

echo ""
echo "=== Installing Chrome ==="
# Check if Chrome already installed
if command -v google-chrome &> /dev/null; then
    echo "Chrome already installed: $(google-chrome --version)"
else
    # Download and install Chrome
    wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    sudo apt install -y ./google-chrome-stable_current_amd64.deb
    rm -f google-chrome-stable_current_amd64.deb
    echo "Chrome installed: $(google-chrome --version)"
fi

echo ""
echo "=== Installing ChromeDriver ==="
# webdriver-manager handles this automatically at runtime
# but we install it now to cache it
python3 -c "
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
driver_path = ChromeDriverManager().install()
print(f'ChromeDriver cached at: {driver_path}')
"

echo ""
echo "=== Verifying Mahimahi is available ==="
if command -v mm-delay &> /dev/null; then
    echo "Mahimahi: OK"
else
    echo "ERROR: Mahimahi not found. Install with: sudo apt install mahimahi"
    exit 1
fi

echo ""
echo "=== Setup complete ==="
echo "Run: python3 selenium_measure.py"