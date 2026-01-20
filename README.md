# Pac‑Man Q‑Learning Deployment

This repository contains a trained Q‑learning implementation of Pac‑Man together with its saved state and performance metrics.  It includes everything needed to run further training or evaluation locally, as well as instructions for hosting the learned policy (`*.qpac`) and the accompanying metrics file (`*_metrics.csv`) on your own server or cloud storage.

## Contents

```
pacman_deployment/
├── PAC‑MAN_patched_v23_10_headless_pump_guard.py  # main game and Q‑learning code
├── Final Simulation.qpac                         # saved Q‑table state after training
├── Final Simulation_metrics.csv                  # metrics logged during training
└── README.md                                     # you are here
```

### PAC‑MAN_patched_v23_10_headless_pump_guard.py

This Python script implements a version of the classic Pac‑Man game with a Q‑learning agent.  It uses `pygame` for rendering, `tkinter` for the user interface, and `pickle`/`csv` for saving the agent’s learned Q‑table (`.qpac` files) and episode metrics.  When you run the script it presents a simple GUI:

* **New Simulation** – start training from scratch and set hyper‑parameters.
* **Old Simulation** – load a previously saved simulation (`*.qpac`) and continue training.
* **Print Details** – print a summary of a saved simulation’s contents.

The script automatically saves updated Q‑tables and writes metrics to a CSV file at the end of each episode if the `metrics_enabled` parameter is set to `True` (it is by default).

### Final Simulation.qpac

This file is a pickled Python dictionary containing the learned Q‑table and hyper‑parameters from a training run.  Loading this file allows the agent to continue learning from where it left off.  The `meta` subdictionary stores metadata such as the number of episodes already trained.

### Final Simulation_metrics.csv

During training the script logs per‑episode statistics (reward, steps, epsilon value, etc.) to a CSV file.  This file makes it easy to chart learning progress or debug training behaviour.

## Running Locally

1. **Install Python 3.9+** and ensure `pip` is available.
2. **Install dependencies.**  The game requires `pygame` for graphics and `tkinter` for the GUI.  On most Linux distributions `tkinter` is provided by the system package manager (e.g. `apt install python3‑tk`).  Use `pip` to install the remaining packages:

   ```bash
   pip install pygame pandas
   ```

3. **Run the game.**  Navigate into the `pacman_deployment` directory and execute:

   ```bash
   python3 PAC‑MAN_patched_v23_10_headless_pump_guard.py
   ```

   When prompted, you can create a new simulation or load the provided `Final Simulation.qpac` to continue training.  At the end of each episode the updated Q‑table and metrics will be saved alongside the original files.

4. **Headless training.**  If you want to train without opening a game window (for example on a server), set the `render` parameter to `0` when creating a new simulation.  You must also instruct SDL to use a dummy video driver so that `pygame` can run without a display.  Before running the script, set the environment variable:

   ```bash
   export SDL_VIDEODRIVER=dummy
   ```

## Deploying on a Website or Cloud Storage

The Q‑table (`*.qpac`) and metrics CSV (`*_metrics.csv`) are ordinary files that can be served from any web server or cloud storage provider.  After running training locally, simply upload these files to your chosen platform (for example GitHub, AWS S3, Google Drive or your personal web hosting).  Visitors can then download the files to continue training or analyse the metrics.

For automated deployments you can modify the Python script to save updated Q‑tables directly into a cloud bucket by replacing the `save_simulation_file` and `_append_metrics_row` functions with code that writes to your storage API.

## Notes

* The game uses `tkinter` for its GUI.  If you encounter `ModuleNotFoundError: No module named 'tkinter'` it means that the Tk libraries are missing on your system.  On Debian/Ubuntu run `sudo apt install python3‑tk` to install them.
* Training can take a long time.  The provided `Final Simulation.qpac` has been trained for roughly 5 000 episodes and demonstrates reasonable behaviour.  Further training may improve the agent’s performance.

If you plan to embed the game in a web page, consider rewriting the interface using a web framework (e.g. Flask or Streamlit) and exposing a REST API for uploading/downloading Q‑tables.  The core logic (the `Game` and `QLearningAgent` classes) can be reused without the `tkinter` components.