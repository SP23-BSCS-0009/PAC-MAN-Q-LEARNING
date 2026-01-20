# Pac-Man Q-Learning

This project contains a Pac‑Man game driven by a Q‑learning agent.  The agent learns how to play through reinforcement learning by updating a Q‑table as it explores the maze.

## Requirements

To run the game on your local machine you need:

* **Python 3.8 or higher.**  The code has been tested with Python 3.10–3.12.
* **Pygame.**  Install the community edition via `pip`:

  ```bash
  pip install pygame
  ```

* **Tkinter** for the graphical menu.  Tkinter is included with standard Python distributions on Windows and macOS.  On Linux you may need to install it separately, e.g. `sudo apt‑get install python3‑tk`.

No other third‑party libraries are required.

## Running the game

1. Clone or download this repository and open a terminal in the project folder.
2. Install Pygame as shown above and ensure Tkinter is available.
3. Launch the game with Python:

   ```bash
   python PAC-MAN.py
   ```

When the program starts it displays a window with a menu.  You can create a new simulation or load an existing `.qpac` file containing a previously trained Q‑table.  After each training episode the Q‑table is updated and saved back to disk.

If you wish to run headless training (without a window), edit the `DEFAULT_PARAMS` dictionary in `PAC-MAN.py` and set `"render": False` before starting the script.  You can also adjust other parameters (learning rate, discount factor, etc.) in this dictionary or via the in‑game GUI.

## Notes

* Q‑table files use the `.qpac` extension.  Keep them in the same directory as `PAC‑MAN.py` so the loader can find them.
* If you encounter a `ModuleNotFoundError` for `pygame`, double‑check that you installed Pygame in the same Python environment from which you are running the script.
* On some Linux systems you might need to install additional audio packages for Pygame to play sounds.

Enjoy training your own Pac‑Man agent!
