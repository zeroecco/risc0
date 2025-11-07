#!/bin/bash

# USB Device Monitor for Basys 3
# Monitors for new USB devices and helps identify the Basys 3 board

echo "🔍 Monitoring for USB device changes..."
echo "Connect/disconnect your Basys 3 board to see changes"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to check for USB devices
check_devices() {
    echo -e "${BLUE}=== USB Device Check ===${NC}"

    # Detect operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo -e "${YELLOW}🐧 Linux system detected${NC}"

        # Check serial devices
        echo -e "${YELLOW}Serial devices:${NC}"
        ls -la /dev/ttyUSB* /dev/ttyACM* /dev/ttyS* 2>/dev/null | head -10 || echo "  No additional serial devices found"

        # Check USB devices
        echo -e "${YELLOW}USB devices (FTDI/Digilent):${NC}"
        if command_exists lsusb; then
            lsusb | grep -i "ftdi\|digilent\|basys\|artix" || echo "  No FTDI/Digilent devices found"
        else
            echo "  lsusb not available"
        fi

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}🍎 macOS system detected${NC}"

        # Check serial devices
        echo -e "${YELLOW}Serial devices:${NC}"
        ls -la /dev/tty.* /dev/cu.* 2>/dev/null | grep -v "Bluetooth\|debug-console" || echo "  No additional serial devices found"

        # Check USB devices
        echo -e "${YELLOW}USB devices (FTDI/Digilent):${NC}"
        system_profiler SPUSBDataType | grep -A 3 -B 1 -i "ftdi\|digilent\|basys\|artix" || echo "  No FTDI/Digilent devices found"

    else
        echo -e "${YELLOW}💻 Generic system detected${NC}"

        # Check serial devices
        echo -e "${YELLOW}Serial devices:${NC}"
        ls -la /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -10 || echo "  No additional serial devices found"

        # Check USB devices
        echo -e "${YELLOW}USB devices:${NC}"
        if command_exists lsusb; then
            lsusb | grep -i "ftdi\|digilent\|basys\|artix" || echo "  No FTDI/Digilent devices found"
        else
            echo "  Use 'lsusb' to check USB devices"
        fi
    fi

    echo ""
}

# Initial check
check_devices

# Monitor for changes
while true; do
    sleep 2
    check_devices
done
