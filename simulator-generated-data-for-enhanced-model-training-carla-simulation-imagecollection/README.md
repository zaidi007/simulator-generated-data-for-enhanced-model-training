# CARLA Simulation Script

This script connects to the CARLA simulator, spawns a vehicle, attaches a camera sensor to it, and captures images during the simulation.

## Prerequisites

- **CARLA Simulator**: Ensure that the CARLA server is installed and running. You can download the latest version from the [CARLA releases page](https://github.com/carla-simulator/carla/releases).

- **Python**: Python 3.6 or later is recommended. Ensure that Python is installed and added to your system's PATH.

- **CARLA Python API**: Install the CARLA Python API client library. You can install it using `pip`:

  ```bash
  pip install carla
  ```

  For more details, refer to the [CARLA PyPI page](https://pypi.org/project/carla/).

- **Additional Dependencies**: Install the following Python packages:

  - `pygame`: Used for handling input events and rendering.

  Install `pygame` using `pip`:

  ```bash
  pip install pygame
  ```

## Usage

1. **Start the CARLA Server**: Launch the CARLA server by running the executable. For example:

   - On Linux:

     ```bash
     ./CarlaUE4.sh
     ```

   - On Windows:

     ```bash
     CarlaUE4.exe
     ```

   For detailed instructions, refer to the [CARLA Quick Start Guide](https://carla.readthedocs.io/en/latest/start_quickstart/).

2. **Run the Script**: Execute the Python script to connect to the CARLA server and start the simulation:

   ```bash
   python your_script_name.py
   ```

   Replace `your_script_name.py` with the actual name of your script file.

## Script Overview

The script performs the following steps:

1. **Connect to the CARLA Server**: Establishes a connection to the CARLA server.

2. **Set Up the Simulation Environment**: Configures the simulation world, including weather conditions and other parameters.

3. **Spawn a Vehicle**: Adds a vehicle actor to the simulation at a specified location.

4. **Attach a Camera Sensor**: Attaches a camera sensor to the vehicle to capture images during the simulation.

5. **Run the Simulation Loop**: Continuously advances the simulation, captures images from the camera sensor, and processes them as needed.

6. **Clean Up**: Destroys all actors and gracefully disconnects from the server upon completion or interruption.

## Notes

- Ensure that the CARLA server is running before executing the script.

- The script assumes that the CARLA server is running on the default IP address (`localhost`) and port (`2000`). If your server is running on a different IP or port, adjust the connection parameters in the script accordingly.

- For more information on the CARLA Python API and its functionalities, refer to the [CARLA Python API Tutorial](https://carla.readthedocs.io/en/latest/python_api_tutorial/).
 