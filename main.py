"""
    Run with python 3:
    ```
        python3 main.py
    ```    
"""
from driver import Driver

# Start the driver and run the 
# preprocessing and feature extraction phases
driverObj = Driver(mode="training").run()