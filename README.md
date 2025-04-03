# NBA Game Predictor

A machine learning application that predicts NBA game outcomes using team statistics and performance data. The application features an interactive GUI with real-time predictions and comprehensive team statistics visualization.

## Features

- Real-time game predictions with win probability
- Comprehensive team statistics display
- Interactive head-to-head history visualization
  - Win distribution pie chart
  - Scoring trends line graph with team averages
- Recent form analysis
- Modern, responsive GUI with dark theme
- Scrollable interface for better usability

## Model Performance

- Training Accuracy: 99.59%
- Test Accuracy: 76.42%

Top predictive features:
1. Home Team Assists
2. Home Team Rebounds
3. Away Team Field Goal Percentage
4. Away Team Assists
5. Away Team Rebounds

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the NBA dataset (NBA-BoxScores-2023-2024.csv) in the project directory
   - The dataset should contain box score statistics for NBA games
   - Required columns: GAME_ID, Home Team, Away Team, Result, and various team statistics

## Usage

1. Run the application:
   ```bash
   python predictor.py
   ```
2. Select teams from the dropdown menus
3. View team statistics and head-to-head history
4. Get real-time predictions for the matchup

## Data Source

This application uses NBA box score data from the 2023-2024 season. The dataset includes:
- Game IDs and dates
- Home and away team statistics
- Game results
- Team performance metrics (FG%, 3P%, FT%, REB, AST, STL, BLK, TO)

## Mac Installation Notes

1. Python Requirements:
   - Python 3.8 or higher recommended
   - Tkinter support required for GUI

2. Install Python with Tkinter support:
   ```bash
   brew install python-tk
   ```

3. For better compatibility, ensure you have the latest version of Python:
   ```bash
   brew install python
   ```

4. If you see any permission errors when running the application:
   - Go to System Preferences > Security & Privacy > Privacy
   - Add your terminal application (Terminal, iTerm2, etc.) to the "Full Disk Access" list
   - Restart your terminal application after granting permissions

5. Virtual Environment (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Running on Mac

### Method 1: Using Terminal
1. Open Terminal
2. Navigate to the project directory:
   ```bash
   cd /path/to/nba_predictor
   ```
3. Activate the virtual environment (if you created one):
   ```bash
   source venv/bin/activate
   ```
4. Run the application:
   ```bash
   python predictor.py
   ```

### Method 2: Using VS Code
1. Open the project folder in VS Code
2. Open Terminal within VS Code (View > Terminal)
3. Activate the virtual environment (if you created one):
   ```bash
   source venv/bin/activate
   ```
4. Run the application:
   ```bash
   python predictor.py
   ```

### Troubleshooting
- If you see a "Permission denied" error, make the script executable:
  ```bash
  chmod +x predictor.py
  ```
- If the application window doesn't appear, try running it with the full Python path:
  ```bash
  /usr/local/bin/python3 predictor.py
  ```
- If you encounter any display issues, ensure you have the latest version of Tkinter:
  ```bash
  brew upgrade python-tk
  ``` 