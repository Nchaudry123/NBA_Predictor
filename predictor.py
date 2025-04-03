import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

class NBAPredictor(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Basic window setup
        self.title("NBA Game Predictor")
        self.geometry("1000x700")
        self.configure(bg='#1a1a1a')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Initialize variables
        self.thinking = False
        self.thinking_dots = 0
        
        # Configure styles
        self._configure_styles()
        
        # Setup UI components
        self._setup_scrollable_frame()
        self._setup_graphs()
        
        # Load data and create UI
        self.load_data()
        self.create_ui()
        
        # Bind mousewheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Common style configurations
        styles = {
            'TFrame': {'background': '#1a1a1a'},
            'TLabel': {'background': '#1a1a1a', 'foreground': '#ffffff', 'font': ('Helvetica Neue', 11)},
            'Header.TLabel': {'font': ('Helvetica Neue', 24, 'bold'), 'foreground': '#ffffff'},
            'Stats.TLabel': {'font': ('Helvetica Neue', 10), 'foreground': '#cccccc'},
            'TButton': {'font': ('Helvetica Neue', 11), 'padding': 5},
            'TLabelframe': {'background': '#1a1a1a', 'foreground': '#ffffff'},
            'TLabelframe.Label': {'background': '#1a1a1a', 'foreground': '#ffffff'},
            'TCombobox': {
                'fieldbackground': '#333333',
                'background': '#333333',
                'foreground': '#ffffff',
                'selectbackground': '#2ecc71',
                'selectforeground': '#ffffff',
                'arrowcolor': '#ffffff',
                'borderwidth': 1,
                'relief': 'solid',
                'padding': 5
            }
        }
        
        for widget, config in styles.items():
            style.configure(widget, **config)
        
        # Configure combobox listbox
        style.map('TCombobox',
                 fieldbackground=[('readonly', '#333333')],
                 selectbackground=[('readonly', '#2ecc71')],
                 selectforeground=[('readonly', '#ffffff')])

    def _setup_scrollable_frame(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.grid(row=0, column=0, sticky='nsew')
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.main_frame, bg='#1a1a1a', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollbar.grid(row=0, column=1, sticky='ns')

    def _setup_graphs(self):
        self.graphs_frame = ttk.Frame(self.scrollable_frame)
        self.graphs_frame.grid(row=4, column=0, columnspan=3, sticky='nsew', pady=15)
        self.graphs_frame.grid_rowconfigure(0, weight=1)
        self.graphs_frame.grid_columnconfigure(0, weight=1)
        self.graphs_frame.grid_columnconfigure(1, weight=1)
        
        # Create a frame for each graph with padding
        self.h2h_graph_frame = ttk.Frame(self.graphs_frame)
        self.h2h_graph_frame.grid(row=0, column=0, sticky='nsew', padx=10)
        self.h2h_graph_frame.grid_rowconfigure(0, weight=1)
        self.h2h_graph_frame.grid_columnconfigure(0, weight=1)
        
        self.form_graph_frame = ttk.Frame(self.graphs_frame)
        self.form_graph_frame.grid(row=0, column=1, sticky='nsew', padx=10)
        self.form_graph_frame.grid_rowconfigure(0, weight=1)
        self.form_graph_frame.grid_columnconfigure(0, weight=1)
        
        # Set up matplotlib with dark theme and high DPI
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.dpi': 120,
            'savefig.dpi': 120,
            'figure.figsize': [4, 3],
            'axes.facecolor': '#1a1a1a',
            'figure.facecolor': '#1a1a1a',
            'axes.edgecolor': '#ffffff',
            'axes.labelcolor': '#ffffff',
            'text.color': '#ffffff',
            'xtick.color': '#ffffff',
            'ytick.color': '#ffffff',
            'grid.color': '#333333',
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica Neue', 'Arial', 'sans-serif'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.edgecolor': '#ffffff',
            'legend.facecolor': '#1a1a1a',
            'legend.framealpha': 0.8
        })
        
        # Create figures with improved styling
        self.head_to_head_fig = Figure(figsize=(4, 3), facecolor='#1a1a1a', dpi=120)
        self.form_fig = Figure(figsize=(4, 3), facecolor='#1a1a1a', dpi=120)
        
        # Create canvases for plots with padding
        self.head_to_head_canvas = FigureCanvasTkAgg(self.head_to_head_fig, self.h2h_graph_frame)
        self.form_canvas = FigureCanvasTkAgg(self.form_fig, self.form_graph_frame)
        
        # Add titles to the graph frames
        ttk.Label(
            self.h2h_graph_frame, 
            text="Head-to-Head Record", 
            style='Header.TLabel',
            font=('Helvetica Neue', 14, 'bold')
        ).grid(row=0, column=0, sticky='n', pady=(0, 5))
        
        ttk.Label(
            self.form_graph_frame, 
            text="Scoring Trends", 
            style='Header.TLabel',
            font=('Helvetica Neue', 14, 'bold')
        ).grid(row=0, column=0, sticky='n', pady=(0, 5))
        
        # Place the canvases
        self.head_to_head_canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.form_canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def load_data(self):
        try:
            file_path = "NBA-BoxScores-2023-2024.csv"
            if not os.path.exists(file_path):
                messagebox.showerror("Error", "Could not find the NBA dataset in the current directory.")
                self.destroy()
                return
            
            # Load and preprocess data
            raw_data = pd.read_csv(file_path)
            
            # Group by game and team
            game_stats = raw_data.groupby(['GAME_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY']).agg({
                'PTS': 'sum',
                'FG_PCT': 'mean',
                'FG3_PCT': 'mean',
                'FT_PCT': 'mean',
                'REB': 'sum',
                'AST': 'sum',
                'STL': 'sum',
                'BLK': 'sum',
                'TO': 'sum'
            }).reset_index()
            
            # Create matches dataframe
            matches = []
            for game_id in game_stats['GAME_ID'].unique():
                game = game_stats[game_stats['GAME_ID'] == game_id]
                if len(game) == 2:
                    home_team, away_team = game.iloc[0], game.iloc[1]
                    matches.append({
                        'GAME_ID': game_id,
                        'Home Team': f"{home_team['TEAM_CITY']} {home_team['TEAM_ABBREVIATION']}",
                        'Away Team': f"{away_team['TEAM_CITY']} {away_team['TEAM_ABBREVIATION']}",
                        'Home Points': home_team['PTS'],
                        'Away Points': away_team['PTS'],
                        'Home FG%': home_team['FG_PCT'],
                        'Away FG%': away_team['FG_PCT'],
                        'Home 3P%': home_team['FG3_PCT'],
                        'Away 3P%': away_team['FG3_PCT'],
                        'Home FT%': home_team['FT_PCT'],
                        'Away FT%': away_team['FT_PCT'],
                        'Home REB': home_team['REB'],
                        'Away REB': away_team['REB'],
                        'Home AST': home_team['AST'],
                        'Away AST': away_team['AST'],
                        'Home STL': home_team['STL'],
                        'Away STL': away_team['STL'],
                        'Home BLK': home_team['BLK'],
                        'Away BLK': away_team['BLK'],
                        'Home TO': home_team['TO'],
                        'Away TO': away_team['TO']
                    })
            
            self.df = pd.DataFrame(matches)
            self.df['Result'] = np.where(self.df['Home Points'] > self.df['Away Points'], 'H', 'A')
            
            # Calculate team stats
            self.teams = sorted(pd.concat([self.df['Home Team'], self.df['Away Team']]).unique())
            self.team_stats = {
                team: self._calculate_team_stats(team) for team in self.teams
            }
            
            # Train model
            self._train_model()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.destroy()

    def _calculate_team_stats(self, team):
        home_games = self.df[self.df['Home Team'] == team]
        away_games = self.df[self.df['Away Team'] == team]
        
        return {
            'games_played': len(home_games) + len(away_games),
            'points_scored': (home_games['Home Points'].mean() + away_games['Away Points'].mean()) / 2,
            'points_allowed': (home_games['Away Points'].mean() + away_games['Home Points'].mean()) / 2,
            'fg_pct': (home_games['Home FG%'].mean() + away_games['Away FG%'].mean()) / 2,
            'fg3_pct': (home_games['Home 3P%'].mean() + away_games['Away 3P%'].mean()) / 2,
            'ft_pct': (home_games['Home FT%'].mean() + away_games['Away FT%'].mean()) / 2,
            'rebounds': (home_games['Home REB'].mean() + away_games['Away REB'].mean()) / 2,
            'assists': (home_games['Home AST'].mean() + away_games['Away AST'].mean()) / 2,
            'steals': (home_games['Home STL'].mean() + away_games['Away STL'].mean()) / 2,
            'blocks': (home_games['Home BLK'].mean() + away_games['Away BLK'].mean()) / 2,
            'turnovers': (home_games['Home TO'].mean() + away_games['Away TO'].mean()) / 2,
            'home_record': f"{len(home_games[home_games['Result'] == 'H'])}-{len(home_games[home_games['Result'] == 'A'])}",
            'away_record': f"{len(away_games[away_games['Result'] == 'A'])}-{len(away_games[away_games['Result'] == 'H'])}"
        }

    def _train_model(self):
        features = [
            'Home FG%', 'Away FG%', 'Home 3P%', 'Away 3P%',
            'Home FT%', 'Away FT%', 'Home REB', 'Away REB',
            'Home AST', 'Away AST', 'Home STL', 'Away STL',
            'Home BLK', 'Away BLK', 'Home TO', 'Away TO'
        ]
        
        X = self.df[features]
        y = self.df['Result']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Print model performance
        print(f"Training Accuracy: {self.model.score(X_train, y_train):.2%}")
        print(f"Test Accuracy: {self.model.score(X_test, y_test):.2%}")
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop Features:")
        print(feature_importance)

    def create_ui(self):
        # Configure scrollable frame grid
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame.grid_columnconfigure(2, weight=1)
        
        # Header
        header = ttk.Label(
            self.scrollable_frame,
            text="NBA Game Predictor",
            style='Header.TLabel'
        )
        header.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # Teams selection frame
        teams_frame = ttk.Frame(self.scrollable_frame)
        teams_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 15))
        teams_frame.grid_columnconfigure(0, weight=1)
        teams_frame.grid_columnconfigure(1, weight=0)
        teams_frame.grid_columnconfigure(2, weight=1)
        
        # Home team selection
        home_frame = ttk.Frame(teams_frame)
        home_frame.grid(row=0, column=0, sticky='ew', padx=10)
        home_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(home_frame, text="Home Team", style='TLabel').grid(row=0, column=0, sticky='w')
        self.home_team_var = tk.StringVar()
        self.home_team_cb = ttk.Combobox(
            home_frame,
            textvariable=self.home_team_var,
            values=self.teams,
            state='readonly'
        )
        self.home_team_cb.grid(row=1, column=0, sticky='ew', pady=5)
        self.home_team_cb.bind('<<ComboboxSelected>>', self.update_stats)
        
        # VS label
        ttk.Label(
            teams_frame,
            text="VS",
            style='Header.TLabel'
        ).grid(row=0, column=1, padx=15)
        
        # Away team selection
        away_frame = ttk.Frame(teams_frame)
        away_frame.grid(row=0, column=2, sticky='ew', padx=10)
        away_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(away_frame, text="Away Team", style='TLabel').grid(row=0, column=0, sticky='w')
        self.away_team_var = tk.StringVar()
        self.away_team_cb = ttk.Combobox(
            away_frame,
            textvariable=self.away_team_var,
            values=self.teams,
            state='readonly'
        )
        self.away_team_cb.grid(row=1, column=0, sticky='ew', pady=5)
        self.away_team_cb.bind('<<ComboboxSelected>>', self.update_stats)
        
        # Stats frame
        stats_frame = ttk.Frame(self.scrollable_frame)
        stats_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(0, 15))
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)
        
        # Home team stats
        self.home_stats_frame = ttk.LabelFrame(stats_frame, text="Home Team Stats")
        self.home_stats_frame.grid(row=0, column=0, sticky='nsew', padx=10)
        self.home_stats_frame.grid_columnconfigure(0, weight=1)
        
        self.home_stats_labels = {
            'record': ttk.Label(self.home_stats_frame, style='Stats.TLabel'),
            'points': ttk.Label(self.home_stats_frame, style='Stats.TLabel'),
            'shooting': ttk.Label(self.home_stats_frame, style='Stats.TLabel'),
            'rebounds': ttk.Label(self.home_stats_frame, style='Stats.TLabel'),
            'other': ttk.Label(self.home_stats_frame, style='Stats.TLabel')
        }
        for i, label in enumerate(self.home_stats_labels.values()):
            label.grid(row=i, column=0, sticky='w', pady=1)
        
        # Away team stats
        self.away_stats_frame = ttk.LabelFrame(stats_frame, text="Away Team Stats")
        self.away_stats_frame.grid(row=0, column=1, sticky='nsew', padx=10)
        self.away_stats_frame.grid_columnconfigure(0, weight=1)
        
        self.away_stats_labels = {
            'record': ttk.Label(self.away_stats_frame, style='Stats.TLabel'),
            'points': ttk.Label(self.away_stats_frame, style='Stats.TLabel'),
            'shooting': ttk.Label(self.away_stats_frame, style='Stats.TLabel'),
            'rebounds': ttk.Label(self.away_stats_frame, style='Stats.TLabel'),
            'other': ttk.Label(self.away_stats_frame, style='Stats.TLabel')
        }
        for i, label in enumerate(self.away_stats_labels.values()):
            label.grid(row=i, column=0, sticky='w', pady=1)
        
        # Head to head frame
        self.h2h_frame = ttk.LabelFrame(self.scrollable_frame, text="Head to Head History")
        self.h2h_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(0, 15), padx=10)
        self.h2h_frame.grid_columnconfigure(0, weight=1)
        
        self.h2h_text = tk.Text(
            self.h2h_frame,
            height=2,
            wrap=tk.WORD,
            font=('Helvetica Neue', 11),
            bg='#2a2a2a',
            fg='#ffffff',
            insertbackground='white'
        )
        self.h2h_text.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        
        # Prediction frame
        self.pred_frame = ttk.LabelFrame(self.scrollable_frame, text="Game Prediction")
        self.pred_frame.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(0, 15), padx=10)
        self.pred_frame.grid_columnconfigure(0, weight=1)
        
        # Main prediction result
        self.result_var = tk.StringVar()
        self.result_label = ttk.Label(
            self.pred_frame,
            textvariable=self.result_var,
            style='Header.TLabel'
        )
        self.result_label.grid(row=0, column=0, pady=5)
        
        # Win probabilities
        self.prob_labels = {
            'home': ttk.Label(self.pred_frame, style='Stats.TLabel'),
            'away': ttk.Label(self.pred_frame, style='Stats.TLabel')
        }
        for i, label in enumerate(self.prob_labels.values()):
            label.grid(row=i+1, column=0, pady=2)

    def update_stats(self, event=None):
        home_team = self.home_team_var.get()
        away_team = self.away_team_var.get()
        
        if home_team:
            stats = self.team_stats[home_team]
            self.home_stats_labels['record'].config(
                text=f"Home Record: {stats['home_record']}"
            )
            self.home_stats_labels['points'].config(
                text=f"PPG: {stats['points_scored']:.1f} | Allowed: {stats['points_allowed']:.1f}"
            )
            self.home_stats_labels['shooting'].config(
                text=f"FG: {stats['fg_pct']:.1%} | 3P: {stats['fg3_pct']:.1%} | FT: {stats['ft_pct']:.1%}"
            )
            self.home_stats_labels['rebounds'].config(
                text=f"Rebounds: {stats['rebounds']:.1f} | Assists: {stats['assists']:.1f}"
            )
            self.home_stats_labels['other'].config(
                text=f"Steals: {stats['steals']:.1f} | Blocks: {stats['blocks']:.1f} | TO: {stats['turnovers']:.1f}"
            )
        
        if away_team:
            stats = self.team_stats[away_team]
            self.away_stats_labels['record'].config(
                text=f"Away Record: {stats['away_record']}"
            )
            self.away_stats_labels['points'].config(
                text=f"PPG: {stats['points_scored']:.1f} | Allowed: {stats['points_allowed']:.1f}"
            )
            self.away_stats_labels['shooting'].config(
                text=f"FG: {stats['fg_pct']:.1%} | 3P: {stats['fg3_pct']:.1%} | FT: {stats['ft_pct']:.1%}"
            )
            self.away_stats_labels['rebounds'].config(
                text=f"Rebounds: {stats['rebounds']:.1f} | Assists: {stats['assists']:.1f}"
            )
            self.away_stats_labels['other'].config(
                text=f"Steals: {stats['steals']:.1f} | Blocks: {stats['blocks']:.1f} | TO: {stats['turnovers']:.1f}"
            )
        
        if home_team and away_team:
            self.update_h2h_history(home_team, away_team)
            self.update_head_to_head(home_team, away_team)
            self.update_form_plot(home_team, away_team)
            self.update_prediction(home_team, away_team)

    def update_h2h_history(self, home_team, away_team):
        self.h2h_text.delete(1.0, tk.END)
        h2h_matches = self.df[
            ((self.df['Home Team'] == home_team) & (self.df['Away Team'] == away_team)) |
            ((self.df['Home Team'] == away_team) & (self.df['Away Team'] == home_team))
        ].copy()
        
        if len(h2h_matches) > 0:
            self.h2h_text.insert(tk.END, "Previous Meetings This Season:\n\n")
            for _, game in h2h_matches.iterrows():
                result = f"{game['Home Team']} {game['Home Points']} - {game['Away Points']} {game['Away Team']}\n"
                self.h2h_text.insert(tk.END, result)
        else:
            self.h2h_text.insert(tk.END, "No previous meetings this season")

    def update_prediction(self, home_team, away_team):
        # Clear previous predictions
        self.result_var.set("Analyzing matchup...")
        for label in self.prob_labels.values():
            label.config(text="")
        
        # Start thinking animation (Make it look like a loading animation, even though it's not)
        self.thinking = True
        self.thinking_dots = 0
        self.after(100, self._update_thinking_animation)
        
        # Schedule the actual prediction
        self.after(2000, lambda: self._make_prediction(home_team, away_team))

    def _update_thinking_animation(self):
        if self.thinking:
            dots = "." * (self.thinking_dots % 4)
            self.result_var.set(f"Analyzing matchup{dots}")
            self.thinking_dots += 1
            self.after(500, self._update_thinking_animation)

    def _make_prediction(self, home_team, away_team):
        self.thinking = False
        
        # Get the last meeting between these teams
        h2h_matches = self.df[
            ((self.df['Home Team'] == home_team) & (self.df['Away Team'] == away_team)) |
            ((self.df['Home Team'] == away_team) & (self.df['Away Team'] == home_team))
        ]
        
        last_meeting = h2h_matches.iloc[-1] if len(h2h_matches) > 0 else None
        
        # Create prediction features
        features = {
            'Home FG%': self.team_stats[home_team]['fg_pct'],
            'Away FG%': self.team_stats[away_team]['fg_pct'],
            'Home 3P%': self.team_stats[home_team]['fg3_pct'],
            'Away 3P%': self.team_stats[away_team]['fg3_pct'],
            'Home FT%': self.team_stats[home_team]['ft_pct'],
            'Away FT%': self.team_stats[away_team]['ft_pct'],
            'Home REB': self.team_stats[home_team]['rebounds'],
            'Away REB': self.team_stats[away_team]['rebounds'],
            'Home AST': self.team_stats[home_team]['assists'],
            'Away AST': self.team_stats[away_team]['assists'],
            'Home STL': self.team_stats[home_team]['steals'],
            'Away STL': self.team_stats[away_team]['steals'],
            'Home BLK': self.team_stats[home_team]['blocks'],
            'Away BLK': self.team_stats[away_team]['blocks'],
            'Home TO': self.team_stats[home_team]['turnovers'],
            'Away TO': self.team_stats[away_team]['turnovers']
        }
        
        X_pred = pd.DataFrame([features])
        probs = self.model.predict_proba(X_pred)[0]
        
        # Update prediction display
        result = self.model.predict(X_pred)[0]
        if result == 'H':
            self.result_var.set(f"Prediction: {home_team} Win 🏆")
        else:
            self.result_var.set(f"Prediction: {away_team} Win 🏆")
        
        # Update probability labels
        self.after(300, lambda: self.prob_labels['home'].config(
            text=f"{home_team} Win: {probs[0]*100:.1f}%"
        ))
        self.after(600, lambda: self.prob_labels['away'].config(
            text=f"{away_team} Win: {probs[1]*100:.1f}%"
        ))

    def update_head_to_head(self, home_team, away_team):
        if not self.home_team_var.get() or not self.away_team_var.get():
            return
            
        # Get head-to-head games
        h2h_games = self.df[
            ((self.df['Home Team'] == self.home_team_var.get()) & (self.df['Away Team'] == self.away_team_var.get())) |
            ((self.df['Home Team'] == self.away_team_var.get()) & (self.df['Away Team'] == self.home_team_var.get()))
        ].sort_values('GAME_ID')
        
        if len(h2h_games) == 0:
            self.h2h_text.insert(tk.END, "No previous meetings")
            return
            
        # Calculate head-to-head record
        home_wins = len(h2h_games[
            ((h2h_games['Home Team'] == self.home_team_var.get()) & (h2h_games['Result'] == 'H')) |
            ((h2h_games['Away Team'] == self.home_team_var.get()) & (h2h_games['Result'] == 'A'))
        ])
        away_wins = len(h2h_games) - home_wins
        
        # Update head-to-head label with more detailed information
        self.h2h_text.insert(tk.END, f"Head-to-Head Record: {self.home_team_var.get()} {home_wins}-{away_wins} {self.away_team_var.get()}\n"
                 f"Total Games: {len(h2h_games)}")
        
        # Create pie chart with improved styling
        self.head_to_head_fig.clear()
        ax = self.head_to_head_fig.add_subplot(111)
        
        # Define colors and labels
        colors = ['#2ecc71', '#e74c3c']  # Green for home team, Red for away team
        labels = [f"{self.home_team_var.get()}\n{home_wins} wins", f"{self.away_team_var.get()}\n{away_wins} wins"]
        
        # Create pie chart with percentage labels and improved styling
        wedges, texts, autotexts = ax.pie(
            [home_wins, away_wins],
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'alpha': 0.8},
            textprops={'fontsize': 10, 'fontweight': 'bold'},
            explode=(0.05, 0) if home_wins > away_wins else (0, 0.05)
        )
        
        # Style the percentage labels
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=10)
        
        # Add title with improved styling
        ax.set_title('Win Distribution', pad=15, color='white', size=12, weight='bold')
        
        # Create line graph with improved styling
        self.form_fig.clear()
        ax = self.form_fig.add_subplot(111)
        
        # Prepare data for the line graph
        games = range(1, len(h2h_games) + 1)
        home_scores = []
        away_scores = []
        
        for _, game in h2h_games.iterrows():
            if game['Home Team'] == self.home_team_var.get():
                home_scores.append(game['Home Points'])
                away_scores.append(game['Away Points'])
            else:
                home_scores.append(game['Away Points'])
                away_scores.append(game['Home Points'])
        
        # Plot lines with improved styling
        ax.plot(games, home_scores, 'o-', color='#2ecc71', label=self.home_team_var.get(), 
                linewidth=2.5, markersize=8, markerfacecolor='white', markeredgecolor='#2ecc71')
        ax.plot(games, away_scores, 'o-', color='#e74c3c', label=self.away_team_var.get(), 
                linewidth=2.5, markersize=8, markerfacecolor='white', markeredgecolor='#e74c3c')
        
        # Style the graph
        ax.set_title('Points Per Game', pad=15, color='white', size=12, weight='bold')
        ax.set_xlabel('Game Number', color='white', size=10, labelpad=10)
        ax.set_ylabel('Points', color='white', size=10, labelpad=10)
        ax.tick_params(colors='white', labelsize=9, grid_color='#333333', grid_alpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend with improved styling
        legend = ax.legend(loc='upper right', fontsize=9, frameon=True, 
                          edgecolor='white', facecolor='#1a1a1a', framealpha=0.8)
        legend.get_frame().set_linewidth(1)
        
        # Add average lines with improved styling
        home_avg = sum(home_scores) / len(home_scores)
        away_avg = sum(away_scores) / len(away_scores)
        ax.axhline(y=home_avg, color='#2ecc71', linestyle='--', alpha=0.7, 
                   label=f'{self.home_team_var.get()} Avg: {home_avg:.1f}')
        ax.axhline(y=away_avg, color='#e74c3c', linestyle='--', alpha=0.7, 
                   label=f'{self.away_team_var.get()} Avg: {away_avg:.1f}')
        
        # Add data point annotations
        for i, (h_pts, a_pts) in enumerate(zip(home_scores, away_scores)):
            ax.annotate(f'{h_pts}', (games[i], h_pts), 
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', color='white', fontsize=8, fontweight='bold')
            ax.annotate(f'{a_pts}', (games[i], a_pts), 
                       textcoords="offset points", xytext=(0,-15), 
                       ha='center', color='white', fontsize=8, fontweight='bold')
        
        # Adjust layout with more padding
        self.head_to_head_fig.tight_layout(pad=2.0)
        self.form_fig.tight_layout(pad=2.0)
        
        # Update canvases
        self.head_to_head_canvas.draw()
        self.form_canvas.draw()

    def update_form_plot(self, home_team, away_team):
        self.form_fig.clear()
        ax = self.form_fig.add_subplot(111)
        
        # Get head-to-head matches
        h2h_matches = self.df[
            ((self.df['Home Team'] == home_team) & (self.df['Away Team'] == away_team)) |
            ((self.df['Home Team'] == away_team) & (self.df['Away Team'] == home_team))
        ].sort_values('GAME_ID')
        
        if len(h2h_matches) > 0:
            home_points = []
            away_points = []
            game_numbers = []
            
            for i, (_, game) in enumerate(h2h_matches.iterrows(), 1):
                if game['Home Team'] == home_team:
                    home_points.append(game['Home Points'])
                    away_points.append(game['Away Points'])
                else:
                    home_points.append(game['Away Points'])
                    away_points.append(game['Home Points'])
                game_numbers.append(i)
            
            # Plot points for both teams
            ax.plot(game_numbers, home_points, 'o-', color='#2ecc71', label=home_team)
            ax.plot(game_numbers, away_points, 'o-', color='#e74c3c', label=away_team)
            
            # Add point values as annotations
            for i, (h_pts, a_pts) in enumerate(zip(home_points, away_points)):
                ax.annotate(f'{h_pts}', (game_numbers[i], h_pts), 
                          textcoords="offset points", xytext=(0,10), ha='center')
                ax.annotate(f'{a_pts}', (game_numbers[i], a_pts), 
                          textcoords="offset points", xytext=(0,-15), ha='center')
            
            ax.set_xlabel('Head-to-Head Games')
            ax.set_ylabel('Points Scored')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title('Head-to-Head Scoring Trend')
        else:
            ax.text(0.5, 0.5, 'No previous meetings this season',
                   horizontalalignment='center', verticalalignment='center')
        
        self.form_canvas.draw()

if __name__ == "__main__":
    app = NBAPredictor()
    app.mainloop() 