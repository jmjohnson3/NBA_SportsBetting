#!/usr/bin/env python
import optuna
import asyncio
import logging
import math
import json
import hashlib
import os
import psycopg2
import psycopg2.extras
import io
import time
from datetime import datetime, timedelta, date, timezone
import pytz
import requests
import pandas as pd
from dateutil import parser
from requests.auth import HTTPBasicAuth
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
from contextlib import redirect_stdout
from rich.console import Console
from rich.table import Table
import smtplib
from email.message import EmailMessage
from jinja2 import Template
import aiohttp
from scipy.stats import norm
from sklearn.cluster import KMeans

# For scaling pipeline and ensemble (if desired)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

# Set pandas option to opt in to future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Set up Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_logging_callback(study, trial):
    logging.info(f"Trial {trial.number} finished with value {trial.value:.5f} and params {trial.params}")
    logging.info(f"Best trial so far: {study.best_trial.number} with value {study.best_value:.5f}")


from optuna.study import MaxTrialsCallback


# -------------------------------
# Constants and API Configuration
# -------------------------------
API_USERNAME = "4359aa1b-cc29-4647-a3e5-7314e2"
API_PASSWORD = "MYSPORTSFEEDS"
ODDS_API_KEY = "5b6f0290e265c3329b3ed27897d79eaf"
SPORT = "basketball_nba"
REGIONS = "us"
MARKETS = "player_points,player_assists,player_rebounds,player_threes"
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"
seasons = ["2023-2024", "2024-2025", "2025-2026"]
TODAY = datetime.now(pytz.timezone("US/Eastern")).date()
TARGET_DATE = TODAY
HARD_CODED_PG_USER = "josh"
HARD_CODED_PG_PASSWORD = "password"
HARD_CODED_PG_HOST = "localhost"
HARD_CODED_PG_PORT = "5432"
HARD_CODED_PG_DATABASE = "nba"
API_CACHE_TTL_MINUTES = 60
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1461420766108319858/LenBk50YR1eS1isFMSOzE8gMWgSgBTSYmU4Ac1unf2SOo_kPSGk71afBqbBiQDuUZwD3"

session = None
api_cache_initialized = False


def get_db_connection():
    try:
        conn = psycopg2.connect(
            user=HARD_CODED_PG_USER,
            password=HARD_CODED_PG_PASSWORD,
            host=HARD_CODED_PG_HOST,
            port=HARD_CODED_PG_PORT,
            dbname=HARD_CODED_PG_DATABASE,
            connect_timeout=5
        )
    except psycopg2.OperationalError as exc:
        logging.error("Postgres connection failed: %s", exc)
        raise
    conn.autocommit = True
    return conn


def init_api_cache() -> None:
    global api_cache_initialized
    if api_cache_initialized:
        return
    try:
        conn = get_db_connection()
    except psycopg2.OperationalError as exc:
        raise RuntimeError(
            "Postgres is required for API caching. Start the database or update the connection settings."
        ) from exc
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    id SERIAL PRIMARY KEY,
                    url TEXT NOT NULL,
                    response_json JSONB NOT NULL,
                    response_hash TEXT NOT NULL,
                    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_api_cache_url_fetched_at
                ON api_cache(url, fetched_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_api_cache_url_hash
                ON api_cache(url, response_hash)
                """
            )
    api_cache_initialized = True


def _serialize_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_json(serialized: str) -> str:
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_cached_response(url: str):
    conn = get_db_connection()
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT response_json, fetched_at
                FROM api_cache
                WHERE url = %s
                ORDER BY fetched_at DESC, id DESC
                LIMIT 1
                """,
                (url,)
            )
            row = cursor.fetchone()
    if not row:
        return None, None
    return row["response_json"], row["fetched_at"]


def cache_response(url: str, payload: dict) -> bool:
    serialized = _serialize_json(payload)
    response_hash = _hash_json(serialized)
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO api_cache (url, response_json, response_hash, fetched_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (url, response_hash) DO NOTHING
                """,
                (url, psycopg2.extras.Json(payload), response_hash)
            )
            return cursor.rowcount > 0


def is_cache_fresh(fetched_at, ttl_minutes: int) -> bool:
    if not fetched_at:
        return False
    cached_time = fetched_at
    if cached_time.tzinfo is None:
        cached_time = cached_time.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - cached_time < timedelta(minutes=ttl_minutes)


def get_http_session() -> requests.Session:
    global session
    if session is None:
        session = requests.Session()
        session.auth = HTTPBasicAuth(API_USERNAME, API_PASSWORD)
    return session


endpoints = {
    "seasonal_games": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/games.json",
    "daily_games": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/games.json",
    "current_season": "https://api.mysportsfeeds.com/v2.1/pull/nba/current_season.json",
    "latest_updates": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/latest_updates.json",
    "seasonal_venues": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/venues.json",
    "daily_team_gamelogs": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/date/{date}/team_gamelogs.json?team={team_abbr}",
    "seasonal_team_gamelogs": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/team_gamelogs.json?team={team_abbr}",
    "daily_player_gamelogs": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/date/{date}/player_gamelogs.json?team={team_abbr}",
    "seasonal_player_gamelogs": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/player_gamelogs.json?team={team_abbr}",
    "seasonal_team_stats": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/team_stats_totals.json",
    "seasonal_player_stats": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/player_stats_totals.json",
    "seasonal_standings": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/standings.json",
    "game_boxscore": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/games/{date}-{away_team_abbr}-{home_team_abbr}/boxscore.json",
    "game_lineups": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/games/{date}-{away_team_abbr}-{home_team_abbr}/lineup.json",
    "game_playbyplay": "https://api.mysportsfeeds.com/v2.1/pull/nba/{season}-regular/games/{date}-{away_team_abbr}-{home_team_abbr}/playbyplay.json",
    "players": "https://api.mysportsfeeds.com/v2.1/pull/nba/players.json",
    "injuries": "https://api.mysportsfeeds.com/v2.1/pull/nba/injuries.json"
}

NBA_TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
             "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
             "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]


# =============================================================================
# NEW: Continuous Monitoring and Feedback
# =============================================================================
def perform_residual_analysis(merged_features, predictions):
    """
    Perform residual analysis to identify systematic prediction errors.
    """
    if "actual_point_diff" not in merged_features.columns:
        print("No actual game outcomes available for residual analysis.")
        return None
    residuals = merged_features["actual_point_diff"] - predictions
    print("\n=== Residual Analysis ===")
    print("Mean Residual:", np.mean(residuals))
    print("Residual Std Dev:", np.std(residuals))
    if "home_games_last_7" in merged_features.columns:
        congestion_analysis = merged_features.groupby("home_games_last_7")["actual_point_diff"].mean()
        print("\nResiduals by Home Games in Last 7 Days:")
        print(congestion_analysis)
    return residuals

def handle_missing_and_outliers(df, columns, method='median', use_robust_scaling=False):
    df = df.copy()
    impute_vals = {}
    for col in columns:
        # Use median or mean (but median is more robust for skewed data)
        impute_val = df[col].median() if method == 'median' else df[col].mean()
        impute_vals[col] = impute_val
        df[col] = df[col].fillna(impute_val)
        # Clip values to within 1.5*IQR limits
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    if use_robust_scaling:
        # Apply a robust scaler to reduce the influence of outliers further.
        scaler = RobustScaler()
        df[columns] = scaler.fit_transform(df[columns])
        # Optionally use inverse scaling (if you want predictions in the original space)
        # df[columns] = scaler.inverse_transform(df[columns])
    return df

def perform_residual_analysis_main(merged_features, game_predictions):
    """
    Analyze residuals to check for systematic errors.
    """
    if "actual_point_diff" in merged_features.columns:
        residuals = merged_features["actual_point_diff"] - game_predictions
        print("\n=== Residual Analysis ===")
        print("Mean Residual:", np.mean(residuals))
        print("Residual Std Dev:", np.std(residuals))
        if "home_games_last_7" in merged_features.columns:
            congestion_resid = merged_features.groupby("home_games_last_7")["actual_point_diff"].mean()
            print("\nResiduals by Home Games in Last 7 Days:")
            print(congestion_resid)
    else:
        print("Actual game outcomes not available for residual analysis.")


def cap_extreme_predictions(predictions, residuals, threshold=3):
    """
    Caps predictions if they are more than 'threshold' standard deviations away
    from the mean of the residuals.
    """
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    lower_bound = mean_resid - threshold * std_resid
    upper_bound = mean_resid + threshold * std_resid
    predictions_capped = np.clip(predictions, lower_bound, upper_bound)
    print(f"Capping predictions to the range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    return predictions_capped


def schedule_retraining():
    """
    Placeholder function to trigger retraining.
    In production, use a scheduler (e.g., cron, APScheduler) to run this script periodically.
    """
    print("Retraining check complete. (Implement scheduling externally.)")


def backtest_betting_strategy(historical_df, model, game_features, betting_thresholds):
    """
    Backtest the betting strategy on historical data.
    """
    total_return = 0
    bets = 0
    for idx, row in historical_df.iterrows():
        X = pd.DataFrame([row[game_features].fillna(0)])
        pred_diff = model.predict(X)[0]
        win_prob = 1.0 / (1.0 + np.exp(-pred_diff / 10.0))
        sportsbook_odds = 2.0  # example odds
        kelly = compute_betting_edge(win_prob, sportsbook_odds)
        bet_result = sportsbook_odds - 1 if row.get("actual_point_diff", 0) > 0 else -1
        total_return += kelly * bet_result
        bets += 1
    print("\n=== Backtesting Betting Strategy ===")
    print(f"Total simulated return over {bets} bets: {total_return:.2f}")


# -------------------------------
# NEW: Data Quality and Feature Engineering Functions
# -------------------------------
def robust_merge(left, right, on, how='left'):
    merged = left.merge(right, on=on, how=how, indicator=True)
    merge_failures = merged[merged['_merge'] != 'both']
    if not merge_failures.empty:
        print("Warning: Merge issues detected", merge_failures.head())
    return merged.drop(columns=['_merge'])


def compute_off_efficiency(df):
    df = df.copy()
    df['possessions'] = df['FGA'] + 0.44 * df['FTA'] - df['ORB'] + df['TO']
    df['off_efficiency'] = (df['PTS'] / df['possessions']) * 100
    return df


def add_interaction_terms(df):
    df = df.copy()
    # Existing interaction between rest and travel.
    if 'days_rest' in df.columns and 'travel_distance' in df.columns:
        df['rest_travel_interaction'] = df['days_rest'] * df['travel_distance']
    if 'days_rest' in df.columns:
        df['days_rest_squared'] = df['days_rest'] ** 2
    # New: Interaction between offensive efficiency and team pace.
    if 'home_off_eff' in df.columns and 'home_team_pace' in df.columns:
        df['home_efficiency_pace'] = df['home_off_eff'] * df['home_team_pace']
    if 'away_off_eff' in df.columns and 'away_team_pace' in df.columns:
        df['away_efficiency_pace'] = df['away_off_eff'] * df['away_team_pace']
    return df

def add_temporal_features(df, metric, span=3, add_lag_features=True, n_lags=2):
    df = df.copy()
    if metric in df.columns and 'game_date' in df.columns:
        df = df.sort_values('game_date')
        # Create an exponential moving average feature.
        df[f'ema_{metric}'] = df[metric].ewm(span=span, adjust=False).mean()
        if add_lag_features:
            # Create lag features of the metric (e.g. previous game values)
            for lag in range(1, n_lags + 1):
                df[f'{metric}_lag{lag}'] = df[metric].shift(lag)
    return df

def compute_pace(team_stats_df, team_abbr):
    minutes = 240.0
    fg_att = get_team_stat(team_stats_df, team_abbr, "stats_fieldGoals.fgAttPerGame") or 0
    ft_att = get_team_stat(team_stats_df, team_abbr, "stats_freeThrows.ftAttPerGame") or 0
    tov = get_team_stat(team_stats_df, team_abbr, "stats_defense.tovPerGame") or 0
    possessions = fg_att + 0.44 * ft_att + tov
    pace = possessions * (100 / minutes)
    return pace


# =============================================================================
# NEW: Defensive Adjustment Functions (using historical team stats)
# =============================================================================
def get_defense_adjustment(team_stats_df, team_abbr):
    """
    Returns an adjustment factor based on the opponent's defensive rating.
    Uses the team's points allowed per game compared to the league average.
    A team with a lower points allowed than the league average (good defense)
    yields a factor less than 1.0, reducing offensive predictions.
    """
    team_rating = get_team_stat(team_stats_df, team_abbr, "stats_defense.ptsAgainstPerGame")
    if not team_rating or team_rating == 0:
        return 1.0
    try:
        league_avg = team_stats_df["stats_defense.ptsAgainstPerGame"].mean()
    except Exception:
        league_avg = 110  # fallback average value
    return team_rating / league_avg


def adjust_game_outcome_for_defense(predicted_score, opponent_team_abbr, team_stats_df):
    """
    Adjusts a predicted team score based on the opponent's defensive quality.
    """
    def_factor = get_defense_adjustment(team_stats_df, opponent_team_abbr)
    return predicted_score * def_factor


def adjust_player_prop_for_defense(predicted_value, opponent_team_abbr, player_position, team_stats_df):
    """
    Adjusts a predicted player prop based on the opponent’s defense.
    Applies a defense factor and an additional position-specific adjustment.
    """
    def_factor = get_defense_adjustment(team_stats_df, opponent_team_abbr)
    # Example position-specific multipliers (tune these based on historical data)
    position_adjustments = {
        "PG": 0.97,
        "SG": 0.97,
        "SF": 0.98,
        "PF": 0.98,
        "C": 0.95
    }
    pos_factor = position_adjustments.get(player_position.upper(), 1.0) if player_position else 1.0
    return predicted_value * def_factor * pos_factor


# =============================================================================
# NEW: MODEL SELECTION, STACKING, CALIBRATION & BETTING EDGE
# =============================================================================
def time_series_cv_evaluation(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, y_pred))
    return np.mean(mse_scores)


def train_stacking_ensemble(X, y):
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    import lightgbm as lgb
    from sklearn.linear_model import LinearRegression

    base_models = [
        ('xgb', XGBRegressor(objective="reg:pseudohubererror", random_state=42)),
        ('lgb', lgb.LGBMRegressor(random_state=42)),
        ('rf', RandomForestRegressor(random_state=42))
    ]
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
    cv_mse = time_series_cv_evaluation(stacking_model, X, y, n_splits=5)
    print("Stacking Ensemble CV MSE:", cv_mse)
    stacking_model.fit(X, y)
    return stacking_model


def calibrate_win_probabilities(raw_probs, actual_outcomes):
    from sklearn.isotonic import IsotonicRegression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(raw_probs, actual_outcomes)
    calibrated_probs = iso_reg.predict(raw_probs)
    return calibrated_probs, iso_reg


def compute_betting_edge(calibrated_prob, sportsbook_odds, fraction=1.0):
    edge = (calibrated_prob * (sportsbook_odds - 1)) - (1 - calibrated_prob)
    if edge <= 0:
        return 0.0
    kelly_fraction = edge / (sportsbook_odds - 1)
    return kelly_fraction * fraction


# -------------------------------
# NEW: Neural Network Integration
# -------------------------------
def train_nn_game_model(X, y, epochs=200, batch_size=64):
    """
    Train a simple feed-forward neural network (using TensorFlow/Keras) for predicting the game point differential.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)

    return model


def train_nn_player_model(X, y, epochs=200, batch_size=64):
    """
    Train a neural network for multi-output regression on player stats.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    input_dim = X.shape[1]
    output_dim = y.shape[1] if len(y.shape) > 1 else 1

    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop], verbose=1)
    return model, history


# -------------------------------
# NEW: AutoML Integration using AutoKeras
# -------------------------------
try:
    import autokeras as ak
except ImportError:
    print("AutoKeras is not installed. Please install with: pip install autokeras")
    ak = None


def train_autokeras_game_model(X, y, max_trials=3, epochs=50):
    """
    Train an AutoML model using AutoKeras StructuredDataRegressor.
    """
    if ak is None:
        raise ImportError("AutoKeras module is required for AutoML model training.")
    reg = ak.StructuredDataRegressor(max_trials=max_trials, overwrite=True)
    reg.fit(X, y, epochs=epochs)
    model = reg.export_model()
    return model, reg.history


# -------------------------------
# NEW: Reinforcement Learning (RL) for Dynamic Betting Adjustment
# -------------------------------
try:
    os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("Gymnasium is not installed. Please install with: pip install gymnasium")
    gym = None

try:
    from stable_baselines3 import DQN
except ImportError:
    print("Stable Baselines3 is not installed. Please install with: pip install stable-baselines3")
    DQN = None


class BettingEnv(gym.Env):
    """A simple Gym environment for simulating dynamic betting adjustments."""

    def __init__(self, predictions, actuals, odds, initial_balance=1000):
        super(BettingEnv, self).__init__()
        self.predictions = predictions
        self.actuals = actuals
        self.odds = odds
        self.current_step = 0
        self.balance = initial_balance

        # Define action space: discrete bet fraction [0, 0.25, 0.5, 0.75, 1.0]
        self.action_space = spaces.Discrete(5)
        # Observation: predicted win probability, odds, current balance
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

    def step(self, action):
        bet_fraction = action * 0.25
        pred = self.predictions[self.current_step]
        actual = self.actuals[self.current_step]
        odd = self.odds[self.current_step]
        # Simplified payout: if actual > 0, we win (odd - 1) times our bet, else lose our bet.
        result = (odd - 1) if actual > 0 else -1
        bet_amount = self.balance * bet_fraction
        reward = bet_amount * result
        self.balance += reward
        self.current_step += 1
        done = self.current_step >= len(self.predictions)
        obs = self._get_obs()
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        return self._get_obs()

    def _get_obs(self):
        if self.current_step < len(self.predictions):
            return np.array([self.predictions[self.current_step], self.odds[self.current_step], self.balance],
                            dtype=np.float32)
        else:
            return np.array([0, 0, self.balance], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}")


def train_rl_agent_on_betting(env, total_timesteps=10000):
    if DQN is None:
        raise ImportError("Stable Baselines3 DQN agent is required for RL training.")
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


# -------------------------------
# Ingestion, Feature Engineering, and Modeling Functions
# (The functions below remain unchanged from your original code.)
# -------------------------------
def extract_in_game_features(playbyplay_df, game_id):
    game_plays = playbyplay_df[playbyplay_df["game_id"] == game_id]
    if game_plays.empty:
        return None
    if "timeRemaining" not in game_plays.columns:
        logging.warning("timeRemaining column not found in play-by-play data for game_id %s", game_id)
        return None
    last_play = game_plays.sort_values("timeRemaining", ascending=True).iloc[-1]
    time_remaining_str = last_play.get("timeRemaining", "0:00")
    try:
        minutes, seconds = map(int, time_remaining_str.split(":"))
    except Exception:
        minutes, seconds = 0, 0
    seconds_remaining = minutes * 60 + seconds
    current_diff = last_play.get("scoreDifferential", 0)
    return {"current_score_diff": current_diff, "seconds_remaining": seconds_remaining}


def blend_predictions(pre_game_pred_diff, in_game_features, total_game_seconds=2880):
    seconds_remaining = in_game_features["seconds_remaining"]
    current_diff = in_game_features.get("scoreDifferential", 0)
    elapsed_fraction = 1 - (seconds_remaining / total_game_seconds)
    blended_diff = (1 - elapsed_fraction) * pre_game_pred_diff + elapsed_fraction * current_diff
    return blended_diff


def add_schedule_congestion(games_df, window_days=7):
    games_df = games_df.copy()
    games_df["local_date"] = pd.to_datetime(games_df["local_date"]).dt.date
    games_df["home_games_last_7"] = 0
    games_df["away_games_last_7"] = 0
    return games_df


def override_team_info(dedup_df, players_data):
    import pandas as pd
    players_info_df = pd.DataFrame(players_data.get("players", []))

    def get_current_team_abbr(p):
        if isinstance(p, dict):
            current_team = p.get("currentTeam")
            if isinstance(current_team, dict):
                return current_team.get("abbreviation", "").upper().strip()
        return ""

    players_info_df["current_team_abbr"] = players_info_df["player"].apply(get_current_team_abbr)
    players_info_df["player_id"] = players_info_df["player"].apply(
        lambda p: p.get("id") if isinstance(p, dict) else None
    )
    mapping = players_info_df.set_index("player_id")["current_team_abbr"].to_dict()
    print("Mapping sample (first 5):", dict(list(mapping.items())[:5]))
    dedup_df["player_id"] = dedup_df["player_id"].astype(int)
    dedup_df["team_abbr"] = dedup_df["player_id"].map(mapping)
    sample = dedup_df.loc[dedup_df["player_id"] == 9343, "team_abbr"].tolist()
    print("For player id 9343, team_abbr after override:", sample)
    return dedup_df


def deduplicate_by_average(df, group_key):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols and col != group_key]
    df_numeric = df.groupby(group_key)[numeric_cols].mean().reset_index()
    df_non_numeric = df.groupby(group_key)[non_numeric_cols].first().reset_index()
    deduped_df = pd.merge(df_non_numeric, df_numeric, on=group_key, how="left")
    return deduped_df


def deduplicate_player_stats(player_stats_df, players_data):
    import numpy as np
    import pandas as pd
    if "player_id" not in player_stats_df.columns:
        player_stats_df["player_id"] = player_stats_df["player"].apply(
            lambda p: p.get("id") if isinstance(p, dict) else p
        )

    def mode_agg(s):
        first_val = s.iloc[0]
        if isinstance(first_val, dict):
            return first_val
        mode = s.mode()
        return mode.iloc[0] if not mode.empty else first_val

    numeric_cols = player_stats_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "player_id"]
    non_numeric_cols = [col for col in player_stats_df.columns if col not in numeric_cols and col != "player_id"]
    df_numeric = player_stats_df.groupby("player_id")[numeric_cols].mean().reset_index()
    df_non_numeric = player_stats_df.groupby("player_id")[non_numeric_cols].agg(mode_agg).reset_index()
    dedup_df = pd.merge(df_non_numeric, df_numeric, on="player_id", how="left")
    players_info_df = pd.DataFrame(players_data.get("players", []))
    if not players_info_df.empty and "team" in players_info_df.columns:
        players_info_df["current_team_abbr"] = players_info_df["team"].apply(
            lambda t: t.get("abbreviation").upper().strip() if isinstance(t, dict) else str(t).upper().strip()
        )
        if "id" in players_info_df.columns:
            players_info_df = players_info_df[["id", "current_team_abbr"]].rename(columns={"id": "player_id"})
            dedup_df = pd.merge(dedup_df, players_info_df, on="player_id", how="left")
            dedup_df["team_abbr"] = dedup_df["current_team_abbr"].combine_first(dedup_df.get("team_abbr"))
    return dedup_df


async def async_fetch_api_data(url: str, refresh: bool = False) -> dict:
    return await asyncio.to_thread(fetch_api_data, url, refresh=refresh)



def fetch_games_data(api_url: str) -> dict:
    logger = logging.getLogger(__name__)
    data = fetch_api_data(api_url)
    if data is None:
        logger.error("No data returned for games endpoint %s", api_url)
        raise ValueError("No data returned for games endpoint")
    if "games" not in data:
        logger.error("The JSON data does not contain a 'games' key.")
        raise ValueError("Invalid JSON structure for games: missing 'games' key")
    return data


def flatten_games_data(api_url: str) -> pd.DataFrame:
    data = fetch_games_data(api_url)
    games = data["games"]
    print(f"Loaded {len(games)} games from API.")
    df_main = pd.json_normalize(games, sep="_")
    print("Basic games dataframe shape:", df_main.shape)
    quarters_data = []
    for game in games:
        game_id = game.get("schedule", {}).get("id")
        score = game.get("score", {})
        quarters = score.get("quarters", [])
        for q in quarters:
            q["schedule_id"] = game_id
            quarters_data.append(q)
    if quarters_data:
        df_quarters = pd.DataFrame(quarters_data)
        print("Quarters data shape:", df_quarters.shape)
        df_pivot = df_quarters.pivot(index="schedule_id", columns="quarterNumber", values=["awayScore", "homeScore"])
        df_pivot.columns = [f"{col[0]}_Q{col[1]}" for col in df_pivot.columns]
        df_pivot.reset_index(inplace=True)
        df_main = pd.merge(df_main, df_pivot, how="left", on="schedule_id")
    print("Final flattened games dataframe shape:", df_main.shape)
    return df_main


def fetch_venues_data(api_url: str) -> dict:
    logger = logging.getLogger(__name__)
    data = fetch_api_data(api_url)
    if data is None:
        logger.error("No data returned for venues endpoint %s", api_url)
        raise ValueError("No data returned for venues endpoint")
    if "venues" not in data:
        logger.error("The JSON data does not contain a 'venues' key.")
        raise ValueError("Invalid JSON structure for venues: missing 'venues' key")
    return data


def flatten_venues_data(api_url: str):
    data = fetch_venues_data(api_url)
    venues = data["venues"]
    print(f"Loaded {len(venues)} venues from API.")
    df_main = pd.json_normalize(venues, sep="_")
    capacities_data = []
    for item in venues:
        if not isinstance(item, dict):
            logging.warning("Skipping venue record because it is not a dictionary: %s", item)
            continue
        venue_info = item.get("venue", {})
        capacities = venue_info.get("capacitiesByEventType", [])
        for cap in capacities:
            cap["venue_id"] = venue_info.get("id")
            capacities_data.append(cap)
    if capacities_data:
        df_capacities = pd.DataFrame(capacities_data)
        print("Capacities data shape:", df_capacities.shape)
        df_cap_pivot = df_capacities.pivot(index="venue_id", columns="eventType", values="capacity")
        df_cap_pivot.columns = [f"capacity_{col}" for col in df_cap_pivot.columns]
        df_cap_pivot.reset_index(inplace=True)
        df_main = pd.merge(df_main, df_cap_pivot, how="left", on="venue_id")
    print("Final flattened venues dataframe shape:", df_main.shape)
    return df_main, capacities_data


def get_team_locations(venues_df):
    locations = {}
    print("Venues DataFrame columns:", venues_df.columns.tolist())
    for idx, row in venues_df.iterrows():
        home_team = row.get("homeTeam")
        if home_team is None or pd.isna(home_team):
            continue
        team_abbr = home_team.get("abbreviation")
        if not team_abbr:
            continue
        team_abbr = team_abbr.strip().upper()
        if team_abbr not in NBA_TEAMS:
            continue
        venue_info = row.get("venue")
        if venue_info is None or pd.isna(venue_info):
            continue
        geo = venue_info.get("geoCoordinates")
        if not geo:
            continue
        lat = geo.get("latitude")
        lon = geo.get("longitude")
        if lat is None or lon is None:
            continue
        try:
            locations[team_abbr] = (float(lat), float(lon))
        except Exception as e:
            print(f"Error converting coordinates for team {team_abbr}: {e}")
            continue
    print("Team locations loaded:", locations)
    return locations


def haversine_distance(coord1, coord2):
    R = 6371  # km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def add_travel_distance_feature(games_df, team_locations, historical_games):
    games_df = games_df.copy()
    games_df["home_travel_distance"] = 0.0
    games_df["away_travel_distance"] = 0.0
    return games_df


def enhance_feature_engineering(games_df, team_stats_df, boxscore_df=None, historical_games=None):
    df = games_df.copy()
    if "home_team_pts" in df.columns and "away_team_pts" in df.columns:
        df["total_team_pts"] = df["home_team_pts"] + df["away_team_pts"]
    else:
        print("WARNING: 'home_team_pts' or 'away_team_pts' missing!")

    def calc_eff(pts, fgPct):
        try:
            return pts / fgPct if fgPct and fgPct != 0 else 0
        except Exception:
            return 0

    if "home_team_pts" in df.columns and "home_team_fgPct" in df.columns:
        df["home_off_eff"] = df.apply(lambda row: calc_eff(row["home_team_pts"], row["home_team_fgPct"]), axis=1)
    if "away_team_pts" in df.columns and "away_team_fgPct" in df.columns:
        df["away_off_eff"] = df.apply(lambda row: calc_eff(row["away_team_pts"], row["away_team_fgPct"]), axis=1)
    df["off_eff_diff"] = df["home_off_eff"] - df["away_off_eff"]
    if "team_momentum_home" in df.columns and "team_momentum_away" in df.columns:
        df["momentum_diff"] = df["team_momentum_home"] - df["team_momentum_away"]
    else:
        df["momentum_diff"] = 0
    df["points_eff_ratio"] = df["home_team_pts"] / (df["away_team_pts_allowed"] + 1e-5)
    df["home_team_pace"] = df["home_team_abbr"].apply(lambda abbr: compute_pace(team_stats_df, abbr))
    df["away_team_pace"] = df["away_team_abbr"].apply(lambda abbr: compute_pace(team_stats_df, abbr))
    df["home_team_adj_pts"] = df["home_team_pts"] / df["home_team_pace"] * 100
    df["away_team_adj_pts"] = df["away_team_pts"] / df["away_team_pace"] * 100
    df["home_team_opp_adj_pts_allowed"] = df["away_team_pts_allowed"] / df["away_team_pace"] * 100
    df["away_team_opp_adj_pts_allowed"] = df["home_team_pts_allowed"] / df["home_team_pace"] * 100
    df["predicted_home_score_adj"] = (df["home_team_adj_pts"] + df["home_team_opp_adj_pts_allowed"]) / 2
    df["predicted_away_score_adj"] = (df["away_team_adj_pts"] + df["away_team_opp_adj_pts_allowed"]) / 2
    df["predicted_point_diff_adj"] = df["predicted_home_score_adj"] - df["predicted_away_score_adj"]
    if all(col in team_stats_df.columns for col in ['PTS', 'FGA', 'FTA', 'ORB', 'TO']):
        team_stats_df = compute_off_efficiency(team_stats_df)
    df = add_interaction_terms(df)
    if 'game_date' in df.columns:
        df = add_temporal_features(df, 'predicted_home_score', span=3)
    feature_cols = ["home_team_pts", "away_team_pts", "home_team_pts_allowed", "away_team_pts_allowed",
                    "def_diff", "home_off_eff", "away_off_eff", "off_eff_diff", "momentum_diff",
                    "points_eff_ratio", "home_team_pace", "away_team_pace",
                    "home_team_adj_pts", "away_team_adj_pts", "home_team_opp_adj_pts_allowed",
                    "away_team_opp_adj_pts_allowed", "predicted_point_diff_adj"]
    df = handle_missing_and_outliers(df, feature_cols, method='median')
    print("Enhanced features added. Here’s a sample:")
    print(df[["total_team_pts", "home_off_eff", "away_off_eff", "off_eff_diff", "momentum_diff",
              "points_eff_ratio", "home_team_pace", "away_team_pace",
              "home_team_adj_pts", "away_team_adj_pts",
              "predicted_point_diff_adj"]].head())
    df["def_diff"] = df["away_team_pts_allowed"] - df["home_team_pts_allowed"]
    df["predicted_point_diff"] = df["predicted_home_score"] - df["predicted_away_score"]
    if boxscore_df is not None and not boxscore_df.empty and "match_key" in boxscore_df.columns:
        merged_df = pd.merge(df, boxscore_df, on=["match_key", "season"], how="left", suffixes=("", "_box"))
    else:
        merged_df = df
    return merged_df


def compute_team_momentum(historical_games, team_abbr, n=3):
    def extract_abbr(team):
        if isinstance(team, dict):
            return team.get("abbreviation")
        return team

    team_games = historical_games[historical_games.apply(
        lambda row: (extract_abbr(row.get("homeTeam")) == team_abbr) or (
                    extract_abbr(row.get("awayTeam")) == team_abbr),
        axis=1)]
    if team_games.empty:
        return 0
    team_games = team_games.sort_values("local_date").tail(n)
    diffs = team_games.apply(lambda row: (row.get("homeTeamPts", 0) - row.get("awayTeamPts", 0))
    if extract_abbr(row.get("homeTeam")) == team_abbr
    else (row.get("awayTeamPts", 0) - row.get("homeTeamPts", 0)), axis=1)
    return diffs.mean()


def compute_usage_rate_for_players(df, team_stats_df):
    required = [
        "stats_fieldGoals.fgAttPerGame",
        "stats_freeThrows.ftAttPerGame",
        "stats_defense.tovPerGame",
        "stats_miscellaneous.minSecondsPerGame"
    ]
    for col in required:
        if col not in df.columns:
            print(f"Cannot compute usageRate because {col} is missing.")
            return df
    df["player_offensive_plays"] = (
                df["stats_fieldGoals.fgAttPerGame"] + 0.44 * df["stats_freeThrows.ftAttPerGame"] + df[
            "stats_defense.tovPerGame"])
    df["player_minutes"] = df["stats_miscellaneous.minSecondsPerGame"] / 60.0

    def get_team_value(abbr, stat):
        value = get_team_stat(team_stats_df, abbr, stat)
        if value is None:
            print(f"[DEBUG] Team stat for team '{abbr}' stat '{stat}' not found, defaulting to 0.")
            return 0
        return value

    df["team_fgAttPerGame"] = df["team_id"].apply(
        lambda team_id: get_team_value(team_id, "stats_fieldGoals.fgAttPerGame"))
    df["team_ftAttPerGame"] = df["team_id"].apply(
        lambda team_id: get_team_value(team_id, "stats_freeThrows.ftAttPerGame"))
    df["team_tovPerGame"] = df["team_id"].apply(lambda team_id: get_team_value(team_id, "stats_defense.tovPerGame"))
    df["team_offensive_plays"] = df["team_fgAttPerGame"] + 0.44 * df["team_ftAttPerGame"] + df["team_tovPerGame"]
    team_minutes = 240.0

    def safe_usage(row):
        if row["player_minutes"] == 0 or row["team_offensive_plays"] == 0:
            return 0.0
        return 100 * (row["player_offensive_plays"] / row["team_offensive_plays"]) * (
                    team_minutes / row["player_minutes"])

    df["stats_offense.usageRate"] = df.apply(safe_usage, axis=1)
    return df


def build_html_predictions(merged_features, player_stats_df, player_model, feat_cols, target_cols, odds_dict,
                           props_odds, thresholds, team_stats_df):
    market_to_pred_key = {
        "player_points": "stats_offense.ptsPerGame",
        "player_assists": "stats_offense.astPerGame",
        "player_rebounds": "stats_rebounds.rebPerGame",
        "player_threes": "stats_fieldGoals.fg3PtMadePerGame"
    }
    label_mapping = {
        "player_points": "Points",
        "player_assists": "Assists",
        "player_rebounds": "Rebounds",
        "player_threes": "3s"
    }
    games_output = []
    for _, game in merged_features.iterrows():
        home_abbr = game.get("home_team_abbr")
        away_abbr = game.get("away_team_abbr")
        predicted_home_score = game.get("predicted_home_score")
        predicted_away_score = game.get("predicted_away_score")
        game_dict = {
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "game_label": f"{away_abbr} vs {home_abbr}",
            "predicted_score": f"{home_abbr} {predicted_home_score:.1f} - {away_abbr} {predicted_away_score:.1f}",
            "total_score": predicted_home_score + predicted_away_score,
            "team_players": {}
        }
        for team_label, abbr in [("Home", home_abbr), ("Away", away_abbr)]:
            players = player_stats_df[player_stats_df["team_abbr"] == abbr]
            players_list = []
            for _, player in players.iterrows():
                player_info = player.get("player")
                if isinstance(player_info, dict):
                    name = f"{player_info.get('firstName', '')} {player_info.get('lastName', '')}".strip()
                    player_position = player_info.get("position")
                else:
                    name = str(player_info)
                    player_position = None
                name_upper = name.upper().strip()
                X_player = pd.DataFrame([player[feat_cols].fillna(0)])
                preds = player_model.predict(X_player)[0]
                pred_dict = dict(zip(target_cols, preds))
                predicted_points = pred_dict.get(market_to_pred_key["player_points"], 0)
                opponent_abbr = game.get("away_team_abbr") if team_label == "Home" else game.get("home_team_abbr")
                adjusted_points = adjust_player_prop_for_defense(predicted_points, opponent_abbr, player_position,
                                                                 team_stats_df)
                fanduel_points = props_odds.get(f"{name_upper}_player_points", "-")
                predicted_assists = pred_dict.get(market_to_pred_key["player_assists"], 0)
                fanduel_assists = props_odds.get(f"{name_upper}_player_assists", "-")
                predicted_rebounds = pred_dict.get(market_to_pred_key["player_rebounds"], 0)
                fanduel_rebounds = props_odds.get(f"{name_upper}_player_rebounds", "-")
                predicted_threes = pred_dict.get(market_to_pred_key["player_threes"], 0)
                fanduel_threes = props_odds.get(f"{name_upper}_player_threes", "-")
                value_bets = []
                for market, pred_key in market_to_pred_key.items():
                    pred_value = pred_dict.get(pred_key, 0)
                    if market == "player_points":
                        pred_value = adjusted_points
                    key = f"{name_upper}_{market}"
                    if key in props_odds:
                        prop_line = props_odds[key]
                        diff = pred_value - prop_line
                        if abs(diff) > thresholds.get(market, 0):
                            if diff > 0:
                                value_bets.append(
                                    f"{label_mapping.get(market)}: {pred_value:.2f} vs. {prop_line:.2f} - over")
                            else:
                                value_bets.append(
                                    f"{label_mapping.get(market)}: {pred_value:.2f} vs. {prop_line:.2f} - under")
                value_bet_str = "<br>".join(value_bets) if value_bets else "-"
                player_dict = {
                    "name": name,
                    "predicted_points": f"{adjusted_points:.2f}",
                    "fanduel_points": fanduel_points,
                    "predicted_assists": f"{predicted_assists:.2f}",
                    "fanduel_assists": fanduel_assists,
                    "predicted_rebounds": f"{predicted_rebounds:.2f}",
                    "fanduel_rebounds": fanduel_rebounds,
                    "predicted_threes": f"{predicted_threes:.2f}",
                    "fanduel_threes": fanduel_threes,
                    "value_bet": value_bet_str
                }
                players_list.append(player_dict)
            game_dict["team_players"][team_label] = players_list
        games_output.append(game_dict)
    template_str = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>NBA Predictions Output</title>
        <style>
          body { font-family: Arial, sans-serif; background-color: #f7f7f7; padding: 20px; }
          .container { max-width: 800px; margin: auto; background: #fff; padding: 20px; }
          table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
          th { background-color: #efefef; }
          h2, h3, h4 { text-align: center; }
          hr { border: none; border-top: 2px solid #ccc; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <h2>NBA Predictions Summary</h2>
          <table>
            <tr>
              <th>Game</th>
              <th>Predicted Score</th>
              <th>Total</th>
            </tr>
            {% for game in games %}
            <tr>
              <td>{{ game.game_label }}</td>
              <td>{{ game.predicted_score }}</td>
              <td>{{ game.total_score | round(1) }}</td>
            </tr>
            {% endfor %}
          </table>
          {% for game in games %}
            <hr/>
            <h3>Game: {{ game.game_label }}</h3>
            <p><strong>Predicted Score:</strong> {{ game.predicted_score }}</p>
            <p style="color: gray;">Odds not available for this matchup.</p>
            <h4>Home Team Players ({{ game.home_abbr }})</h4>
            {% if game.team_players.Home|length == 0 %}
              <p>No player predictions available for Home team.</p>
            {% else %}
              <table>
                <tr>
                  <th>Player</th>
                  <th>Predicted Points</th>
                  <th>Fanduel Points</th>
                  <th>Predicted Assists</th>
                  <th>Fanduel Assists</th>
                  <th>Predicted Rebounds</th>
                  <th>Fanduel Rebounds</th>
                  <th>Predicted 3s</th>
                  <th>Fanduel 3s</th>
                  <th>Value Bet</th>
                </tr>
                {% for player in game.team_players.Home %}
                <tr>
                  <td>{{ player.name }}</td>
                  <td>{{ player.predicted_points }}</td>
                  <td>{{ player.fanduel_points }}</td>
                  <td>{{ player.predicted_assists }}</td>
                  <td>{{ player.fanduel_assists }}</td>
                  <td>{{ player.predicted_rebounds }}</td>
                  <td>{{ player.fanduel_rebounds }}</td>
                  <td>{{ player.predicted_threes }}</td>
                  <td>{{ player.fanduel_threes }}</td>
                  <td>{{ player.value_bet }}</td>
                </tr>
                {% endfor %}
              </table>
            {% endif %}
            <h4>Away Team Players ({{ game.away_abbr }})</h4>
            {% if game.team_players.Away|length == 0 %}
              <p>No player predictions available for Away team.</p>
            {% else %}
              <table>
                <tr>
                  <th>Player</th>
                  <th>Predicted Points</th>
                  <th>Fanduel Points</th>
                  <th>Predicted Assists</th>
                  <th>Fanduel Assists</th>
                  <th>Predicted Rebounds</th>
                  <th>Fanduel Rebounds</th>
                  <th>Predicted 3s</th>
                  <th>Fanduel 3s</th>
                  <th>Value Bet</th>
                </tr>
                {% for player in game.team_players.Away %}
                <tr>
                  <td>{{ player.name }}</td>
                  <td>{{ player.predicted_points }}</td>
                  <td>{{ player.fanduel_points }}</td>
                  <td>{{ player.predicted_assists }}</td>
                  <td>{{ player.fanduel_assists }}</td>
                  <td>{{ player.predicted_rebounds }}</td>
                  <td>{{ player.fanduel_rebounds }}</td>
                  <td>{{ player.predicted_threes }}</td>
                  <td>{{ player.fanduel_threes }}</td>
                  <td>{{ player.value_bet }}</td>
                </tr>
                {% endfor %}
              </table>
            {% endif %}
          {% endfor %}
        </div>
      </body>
    </html>
    """
    template = Template(template_str)
    html_result = template.render(games=games_output)
    return html_result


def generate_html_email_body(plain_text):
    html_template = f"""<!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NBA Predictions Output</title>
        <style type="text/css">
          body {{
            margin: 0;
            padding: 0;
            background-color: #f7f7f7;
            font-family: Arial, sans-serif;
            font-size: 16px;
            color: #333333;
          }}
          .container {{
            width: 100%;
            max-width: 600px;
            margin: auto;
            background-color: #ffffff;
            padding: 20px;
          }}
          h2 {{
            text-align: center;
            color: #2a2a2a;
          }}
          pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
          }}
          @media only screen and (max-width: 600px) {{
            .container {{
              width: 100%;
              padding: 10px;
            }}
            h2 {{
              font-size: 20px;
            }}
            body {{
              font-size: 14px;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h2>NBA Predictions Output</h2>
          <pre>{plain_text}</pre>
        </div>
      </body>
    </html>
    """
    return html_template


def send_email(subject, plain_body, to_email, html_body=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = "jjohnson0636@gmail.com"
    msg['To'] = to_email
    msg.set_content(plain_body if plain_body else "Your email client does not support HTML.")
    if html_body:
        msg.add_alternative(html_body, subtype='html')
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login("jjohnson0636@gmail.com", "wodh yzme squo lmzc")
            smtp.send_message(msg)
        logging.info("Email sent successfully to %s", to_email)
    except smtplib.SMTPAuthenticationError as e:
        logging.error("SMTP Authentication Error: %s. Please use an application-specific password.", e)
    except Exception as e:
        logging.error("Failed to send email: %s", e)


def format_playbook_bet(player_name: str, market: str, line: float, direction: str) -> str:
    market_label = {
        "player_points": "points",
        "player_assists": "assists",
        "player_rebounds": "rebounds",
        "player_threes": "3s"
    }.get(market, market)
    return f"{player_name} {direction} {line} {market_label}"


def send_discord_value_bets(value_bets):
    if not DISCORD_WEBHOOK_URL:
        logging.info("DISCORD_WEBHOOK_URL not set; skipping Discord value bet post.")
        return
    if not value_bets:
        logging.info("No value bets to send to Discord.")
        return
    content = "**Value Bets**\n" + "\n".join(f"- {bet}" for bet in value_bets)
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=10)
        if response.status_code >= 400:
            logging.error("Discord webhook failed: %s %s", response.status_code, response.text)
    except Exception as exc:
        logging.error("Failed to send Discord webhook: %s", exc)


def get_teams_with_games(games_df, target_date):
    target_date = pd.to_datetime(target_date).date()
    scheduled_games = games_df[games_df["local_date"] == target_date]
    teams = set()
    for _, game in scheduled_games.iterrows():
        home = game.get("homeTeam")
        away = game.get("awayTeam")
        home_abbr = home.get("abbreviation") if isinstance(home, dict) else home
        away_abbr = away.get("abbreviation") if isinstance(away, dict) else away
        logging.info("Scheduled game on %s: %s vs %s, startTime: %s, playedStatus: %s", target_date, home_abbr,
                     away_abbr, game.get("startTime"), game.get("playedStatus", "N/A"))
        if home_abbr:
            teams.add(home_abbr)
        if away_abbr:
            teams.add(away_abbr)
    print("Teams with games on", target_date, ":", teams)
    return list(teams)


# Modify get_nba_odds to include real odds
def get_nba_odds():
    url = (
        f"{ODDS_BASE_URL}/{SPORT}/odds/?apiKey={ODDS_API_KEY}&regions={REGIONS}&markets=h2h&oddsFormat={ODDS_FORMAT}&dateFormat={DATE_FORMAT}")
    response = requests.get(url)
    if response.status_code != 200:
        logging.error("Failed to fetch game odds: %s", response.status_code)
        return {}
    data = response.json()
    odds_dict = {}
    for event in data:
        if not isinstance(event, dict):
            continue
        event_id = event.get("id")
        home_team_full = event.get("home_team", "")
        away_team_full = event.get("away_team", "")
        home_team = home_team_full.upper().strip()
        away_team = away_team_full.upper().strip()
        if not home_team or not away_team or not event_id:
            continue
        bookmakers = event.get("bookmakers", [])
        game_key = None
        for bookmaker in bookmakers:
            if bookmaker.get("key", "").lower() != "fanduel":
                continue
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    # Get the price for home and away
                    home_price = None
                    away_price = None
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "").upper().strip()
                        price = outcome.get("price")
                        if price is None:
                            continue
                        if name == home_team:
                            home_price = price
                        elif name == away_team:
                            away_price = price
                    if home_price is not None and away_price is not None:
                        # Convert the American odds to decimal odds for the home team
                        if home_price < 0:
                            home_odds = 1 + (100 / abs(home_price))
                            home_prob = -home_price / (-home_price + 100)
                        else:
                            home_odds = 1 + (home_price / 100)
                            home_prob = 100 / (home_price + 100)
                        game_key = f"{home_team} vs {away_team}"
                        odds_dict[game_key] = {"home_prob": home_prob, "home_odds": home_odds, "event_id": event_id}
                    break  # exit markets loop once h2h market is processed
            if game_key is not None:
                break  # exit bookmakers loop once odds for this event are found
    print("NBA Odds loaded for", len(odds_dict), "games.")
    return odds_dict


# New helper function to assign actual odds for each historical game.
def assign_game_odds(merged_features, nba_odds, default_odds=2.0):
    odds_array = []
    for idx, row in merged_features.iterrows():
        home_team = row.get("home_team_abbr")
        away_team = row.get("away_team_abbr")
        # Build the game key in the same way get_nba_odds did
        game_key = f"{home_team} vs {away_team}"
        if game_key in nba_odds and "home_odds" in nba_odds[game_key]:
            odds_array.append(nba_odds[game_key]["home_odds"])
        else:
            odds_array.append(default_odds)
    return np.array(odds_array)



def get_player_props_odds_for_event(event_id):
    url = (
        f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds?apiKey={ODDS_API_KEY}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}&dateFormat={DATE_FORMAT}")
    response = requests.get(url)
    if response.status_code == 404:
        logging.info("Player props odds not available for event %s (404)", event_id)
        return {}
    elif response.status_code != 200:
        logging.error("Failed to fetch player props odds for event %s: %s", event_id, response.status_code)
        return {}
    data = response.json()
    if not isinstance(data, dict):
        logging.error("Expected a dictionary for event odds, got %s instead", type(data))
        return {}
    props_dict = {}
    desired_markets = {s.strip() for s in MARKETS.split(',')}
    preferred_bookmakers = ["fanduel", "draftkings", "betonlineag", "bovada", "betmgm", "betrivers", "williamhill_us"]
    found = False
    for bookmaker in data.get("bookmakers", []):
        bk_key = bookmaker.get("key", "").lower()
        if bk_key not in preferred_bookmakers:
            continue
        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            if market_key in desired_markets:
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", "").upper().strip()
                    point_value = outcome.get("point")
                    if player_name and point_value is not None:
                        props_dict[f"{player_name}_{market_key}"] = point_value
                        found = True
        if found:
            break
    if not props_dict:
        print(f"No player props odds found for event {event_id}.")
    return props_dict


def get_all_player_props_odds(odds_dict):
    props_odds = {}
    for game_key, details in odds_dict.items():
        event_id = details.get("event_id")
        if event_id:
            event_props = get_player_props_odds_for_event(event_id)
            if not event_props:
                logging.info("No player props available for event %s", event_id)
            props_odds.update(event_props)
    print("Player props odds loaded for", len(props_odds), "player props.")
    return props_odds


def process_injuries(injuries_data):
    records = []
    for player in injuries_data.get("players", []):
        pid = player.get("id")
        injury = player.get("currentInjury")
        if injury:
            prob = injury.get("playingProbability", "").upper()
            if prob == "OUT":
                factor = 1.0
            elif prob == "QUESTIONABLE":
                factor = 0.5
            else:
                factor = 0.0
        else:
            factor = 0.0
        records.append({"player_id": pid, "injury_factor": factor})
    df_injuries = pd.DataFrame(records)
    print("Processed injuries data, shape:", df_injuries.shape)
    return df_injuries


def fetch_api_data(url, max_retries=3, refresh=False, cache_ttl_minutes=API_CACHE_TTL_MINUTES):
    init_api_cache()
    cached_data = None
    cached_at = None
    cached_data, cached_at = get_cached_response(url)
    if cached_data is not None and not refresh:
        if cache_ttl_minutes is None or is_cache_fresh(cached_at, cache_ttl_minutes):
            return cached_data
    session = get_http_session()
    for attempt in range(max_retries):
        try:
            response = session.get(url)
            if response.status_code == 200:
                payload = response.json()
                inserted = cache_response(url, payload)
                if inserted:
                    logging.info("Cached new API response for %s", url)
                return payload
            elif response.status_code == 204:
                logging.warning("Received 204 (No Content) from %s", url)
                payload = {}
                inserted = cache_response(url, payload)
                if inserted:
                    logging.info("Cached empty API response for %s", url)
                return cached_data or payload
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2))
                time.sleep(retry_after)
            else:
                logging.error("Failed to fetch data from %s. Status code: %s", url, response.status_code)
                break
        except Exception:
            logging.exception("Exception occurred while fetching data from %s", url)
            break
    return cached_data



def get_local_game_date(start_time, tz="US/Eastern"):
    dt_utc = parser.isoparse(start_time)
    local_tz = pytz.timezone(tz)
    dt_local = dt_utc.astimezone(local_tz)
    return dt_local.strftime("%Y%m%d")


def get_team_stat(ts_df, team_identifier, stat_col):
    team_identifier_str = str(team_identifier).upper().strip()
    filtered = ts_df[ts_df["team"].apply(lambda x: (
        x.get("abbreviation").upper().strip() if isinstance(x, dict) else str(
            x).upper().strip())) == team_identifier_str]
    if not filtered.empty:
        return filtered.iloc[0].get(stat_col)
    try:
        team_id = int(team_identifier)
        filtered = ts_df[ts_df["team"].apply(lambda x: x.get("id") if isinstance(x, dict) else None) == team_id]
        if not filtered.empty:
            return filtered.iloc[0].get(stat_col)
    except ValueError:
        pass
    print(f"[DEBUG] No team stat found for team '{team_identifier_str}' with stat '{stat_col}'.")
    return 0


# =============================================================================
# NEW: Game-Level Model Training with Optuna, Time-Series CV, and Stacking Ensemble
# =============================================================================
def train_game_prediction_model_with_optuna_cv(merged_features, n_trials=1000):
    # Keep the list of features as before.
    features = [
        "home_team_pts", "away_team_pts",
        "home_team_pts_allowed", "away_team_pts_allowed",
        "def_diff",
        "home_team_ast", "away_team_ast",
        "home_team_fgPct", "away_team_fgPct",
        "home_team_ftPct", "away_team_ftPct",
        "home_team_reb", "away_team_reb",
        "total_team_pts", "home_off_eff", "away_off_eff",
        "team_momentum_home", "team_momentum_away",
        "off_eff_diff", "momentum_diff", "points_eff_ratio",
        "home_team_pace", "away_team_pace",
        "home_team_adj_pts", "away_team_adj_pts",
        "home_team_opp_adj_pts_allowed", "away_team_opp_adj_pts_allowed",
        "predicted_point_diff_adj"
    ]
    features = [f for f in features if f in merged_features.columns]
    # Set up the target (use actual scores if available)
    if "scoring_homeScoreTotal" in merged_features.columns and "scoring_awayScoreTotal" in merged_features.columns:
        merged_features["actual_point_diff"] = merged_features["scoring_homeScoreTotal"] - merged_features[
            "scoring_awayScoreTotal"]
    else:
        merged_features["actual_point_diff"] = merged_features["predicted_point_diff"]
    historical = merged_features.dropna(subset=["actual_point_diff"] + features)
    print("Training data shape:", historical.shape)
    X = historical[features]
    y = historical["actual_point_diff"]
    print("Correlation of features with actual point diff:")
    corr_matrix = historical[features + ["actual_point_diff"]].corr()
    print(corr_matrix["actual_point_diff"].sort_values(ascending=False))

    # Objective function for Optuna
    def optuna_objective_cv(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }
        model = XGBRegressor(**params, objective="reg:pseudohubererror", random_state=42)
        mse = time_series_cv_evaluation(model, X, y, n_splits=5)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective_cv(trial), n_trials=n_trials)
    print("Best trial (CV):", study.best_trial.params)
    best_params = study.best_trial.params

    # Build an ensemble stacking pipeline with enhanced preprocessing.
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression
    # Base models remain similar.
    base_models = [
        ('xgb', XGBRegressor(**best_params, objective="reg:pseudohubererror", random_state=42)),
    ]
    # Use a pipeline that performs robust scaling and a PCA reduction before the regressor.
    stacking_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("pca", PCA(n_components=min(len(features), 10))),  # reduce to at most 10 components
        ("regressor", StackingRegressor(
            estimators=base_models,
            final_estimator=LinearRegression()))
    ])
    cv_mse = time_series_cv_evaluation(stacking_pipeline, X, y, n_splits=5)
    print("Stacking Ensemble CV MSE:", cv_mse)
    stacking_pipeline.fit(X, y)
    return stacking_pipeline, features


def filter_low_minutes_players(df, min_minutes=5):
    return df


def compute_weighted_player_rolling_averages(daily_logs_df, window=5, decay=0.8):
    if "player_id" not in daily_logs_df.columns:
        if "player" in daily_logs_df.columns:
            daily_logs_df["player_id"] = daily_logs_df["player"].apply(
                lambda x: x.get("id") if isinstance(x, dict) else None)
        else:
            logging.error("Neither 'player_id' nor 'player' column found in daily logs DataFrame.")
            return pd.DataFrame()
    if "stats" in daily_logs_df.columns and daily_logs_df["stats"].apply(lambda x: isinstance(x, dict)).all():
        stats_flat = pd.json_normalize(daily_logs_df["stats"])
        stats_flat.columns = [f"stats_{col}" for col in stats_flat.columns]
        daily_logs_df = pd.concat([daily_logs_df.drop(columns=["stats"]), stats_flat], axis=1)

    def select_col(*possible_keys):
        for key in possible_keys:
            if key in daily_logs_df.columns:
                return key
        return None

    pts_key = select_col("stats_offense.ptsPerGame", "stats_offense.pts", "pts")
    ast_key = select_col("stats_offense.astPerGame", "stats_offense.ast", "ast")
    reb_key = select_col("stats_rebounds.rebPerGame", "stats_rebounds.reb", "reb")
    fg3_key = select_col("stats_fieldGoals.fg3PtMadePerGame", "stats_fieldGoals.fg3PtMade", "fg3")
    stats_cols = {"pts": pts_key, "ast": ast_key, "reb": reb_key, "fg3": fg3_key}
    missing = [k for k, col in stats_cols.items() if col is None]
    if missing:
        logging.error("Missing expected stat columns for weighted averages: %s", missing)
        return pd.DataFrame()
    if "game.startTime" in daily_logs_df.columns:
        daily_logs_df["game_date"] = pd.to_datetime(daily_logs_df["game.startTime"])
    else:
        daily_logs_df["game_date"] = pd.Timestamp.today()
    for key, col in stats_cols.items():
        daily_logs_df[col] = pd.to_numeric(daily_logs_df[col], errors="coerce")
    group_key = "player_id"
    weighted_features = {}
    for key, col in stats_cols.items():
        def weighted_avg(group):
            group = group.sort_values("game_date")
            n = len(group)
            weights = np.array([decay ** (n - i - 1) for i in range(n)])
            weights /= weights.sum()
            return np.dot(group[col].fillna(0), weights)

        weighted_series = daily_logs_df.groupby(group_key).apply(weighted_avg)
        weighted_series.name = f"weighted_rolling_{key}"
        weighted_features[f"weighted_rolling_{key}"] = weighted_series
    weighted_df = pd.DataFrame(weighted_features).reset_index().rename(columns={group_key: "player.id"})
    print("Weighted rolling averages computed. Shape:", weighted_df.shape)
    logging.debug("Weighted rolling averages sample:\n%s", weighted_df.head())
    return weighted_df


def compute_team_momentum(historical_games, team_abbr, n=3):
    def extract_abbr(team):
        if isinstance(team, dict):
            return team.get("abbreviation")
        return team

    team_games = historical_games[historical_games.apply(
        lambda row: (extract_abbr(row.get("homeTeam")) == team_abbr) or (
                    extract_abbr(row.get("awayTeam")) == team_abbr), axis=1)]
    if team_games.empty:
        return 0
    team_games = team_games.sort_values("local_date").tail(n)
    diffs = team_games.apply(lambda row: (row.get("homeTeamPts", 0) - row.get("awayTeamPts", 0)) if extract_abbr(
        row.get("homeTeam")) == team_abbr else (row.get("awayTeamPts", 0) - row.get("homeTeamPts", 0)), axis=1)
    return diffs.mean()


def fetch_season_data(season):
    data = {}
    data["seasonal_games"] = fetch_api_data(endpoints["seasonal_games"].format(season=season))
    data["seasonal_venues"] = fetch_api_data(endpoints["seasonal_venues"].format(season=season))
    data["seasonal_team_stats"] = fetch_api_data(endpoints["seasonal_team_stats"].format(season=season))
    data["seasonal_player_stats"] = fetch_api_data(endpoints["seasonal_player_stats"].format(season=season))
    data["seasonal_standings"] = fetch_api_data(endpoints["seasonal_standings"].format(season=season))
    data["latest_updates"] = fetch_api_data(endpoints["latest_updates"].format(season=season))
    print(f"Fetched season data for {season}")
    return data


def fetch_game_details(season, game, target_date=TARGET_DATE):
    details = {}
    game_info = game.get("schedule", game)
    start_time = game_info.get("startTime")
    if not start_time:
        return details
    local_game_date = parser.isoparse(start_time).astimezone(pytz.timezone("US/Eastern")).date()
    if local_game_date != target_date:
        return details
    date_str = get_local_game_date(start_time, tz="US/Eastern")
    away_team = game_info.get("awayTeam", {}).get("abbreviation") or game_info.get("awayTeamAbbreviation")
    home_team = game_info.get("homeTeam", {}).get("abbreviation") or game_info.get("homeTeamAbbreviation")
    if not away_team or not home_team:
        return details
    url_boxscore = endpoints["game_boxscore"].format(season=season, date=date_str, away_team_abbr=away_team,
                                                     home_team_abbr=home_team)
    url_lineup = endpoints["game_lineups"].format(season=season, date=date_str, away_team_abbr=away_team,
                                                  home_team_abbr=home_team)
    url_playbyplay = endpoints["game_playbyplay"].format(season=season, date=date_str, away_team_abbr=away_team,
                                                         home_team_abbr=home_team)
    details["boxscore"] = fetch_api_data(url_boxscore)
    details["lineup"] = fetch_api_data(url_lineup)
    details["playbyplay"] = fetch_api_data(url_playbyplay)
    return details


def ingest_data():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [
        async_fetch_api_data(endpoints["current_season"]),
        async_fetch_api_data(endpoints["players"]),
        async_fetch_api_data(endpoints["injuries"])
    ]
    current_season_data, players_data, injuries_data = loop.run_until_complete(asyncio.gather(*tasks))
    print("Current season data keys:", current_season_data.keys() if current_season_data else "None")
    all_season_data = {}
    all_games_details = {}
    for season in seasons:
        season_data = fetch_season_data(season)
        all_season_data[season] = season_data
        game_details_for_season = {}
        seasonal_games = (season_data.get("seasonal_games") or {}).get("games", [])
        print(f"{season} has {len(seasonal_games)} games.")
        for game in seasonal_games:
            if not isinstance(game, dict):
                logging.warning("Skipping game record because it's not a dictionary: %s", game)
                continue
            game_id = game.get("id") or game.get("schedule", {}).get("id")
            if game_id:
                game_id = str(game_id)
                details = fetch_game_details(season, game, target_date=TARGET_DATE)
                if details:
                    game_details_for_season[game_id] = details
        all_games_details[season] = game_details_for_season
    print("Ingested all season data.")
    return current_season_data, players_data, injuries_data, all_season_data, all_games_details


def parse_seasonal_data(all_season_data):
    games_list = []
    venues_list = []
    team_stats_list = []
    player_stats_list = []
    standings_list = []
    updates_list = []
    for season, data in all_season_data.items():
        seasonal_games = (data.get("seasonal_games") or {}).get("games", [])
        print(f"Parsing {len(seasonal_games)} games for season {season}.")
        for game in seasonal_games:
            if not isinstance(game, dict):
                logging.warning("Skipping game record because it's not a dictionary: %s", game)
                continue
            game_info = game.get("schedule", game)
            game_info["season"] = season
            try:
                dt = parser.isoparse(game_info["startTime"])
                local_date = dt.astimezone(pytz.timezone("US/Eastern")).date()
            except Exception:
                local_date = None
            game_info["local_date"] = local_date
            home_abbr = game_info.get("homeTeam", {}).get("abbreviation")
            away_abbr = game_info.get("awayTeam", {}).get("abbreviation")
            if home_abbr and away_abbr and local_date:
                game_info["match_key"] = f"{season}_{local_date.strftime('%Y%m%d')}_{away_abbr}_{home_abbr}"
            else:
                game_info["match_key"] = None
            game_info["home_advantage"] = 1
            games_list.append(game_info)
        seasonal_venues = (data.get("seasonal_venues") or {}).get("venues", [])
        for venue in seasonal_venues:
            if not isinstance(venue, dict):
                logging.warning("Skipping venue record because it's not a dictionary: %s", venue)
                continue
            venue["season"] = season
            venues_list.append(venue)
        team_stats_data = data.get("seasonal_team_stats") or {}
        for team in team_stats_data.get("teamStatsTotals", []):
            if not isinstance(team, dict):
                logging.warning("Skipping team stats record because it's not a dictionary: %s", team)
                continue
            team["season"] = season
            team_stats_list.append(team)
        player_stats_data = data.get("seasonal_player_stats") or {}
        for player in player_stats_data.get("playerStatsTotals", []):
            if not isinstance(player, dict):
                logging.warning("Skipping player stats record because it's not a dictionary: %s", player)
                continue
            player["season"] = season
            player_stats_list.append(player)
        for standing in (data.get("seasonal_standings") or {}).get("standings", []):
            if not isinstance(standing, dict):
                logging.warning("Skipping standings record because it's not a dictionary: %s", standing)
                continue
            standing["season"] = season
            standings_list.append(standing)
        latest_updates_data = data.get("latest_updates") or {}
        for update in latest_updates_data.get("updates", []):
            if not isinstance(update, dict):
                logging.warning("Skipping update record because it's not a dictionary: %s", update)
                continue
            update["season"] = season
            updates_list.append(update)
    games_df = pd.DataFrame(games_list)
    venues_df = pd.DataFrame(venues_list)
    team_stats_df = pd.DataFrame(team_stats_list)
    player_stats_df = pd.DataFrame(player_stats_list)
    standings_df = pd.DataFrame(standings_list)
    updates_df = pd.DataFrame(updates_list)
    print("Parsed seasonal games shape:", games_df.shape)
    return games_df, venues_df, team_stats_df, player_stats_df, standings_df, updates_df


def parse_detailed_game_data(all_games_details):
    boxscore_list = []
    playbyplay_list = []
    for season, games in all_games_details.items():
        for game_id, details in games.items():
            box = details.get("boxscore")
            if box and isinstance(box, dict):
                box["season"] = season
                box["game_id"] = game_id
                game_obj = box.get("game", {})
                try:
                    dt = parser.isoparse(game_obj.get("startTime"))
                    local_date = dt.astimezone(pytz.timezone("US/Eastern")).strftime("%Y%m%d")
                except Exception:
                    local_date = None
                away_abbr = game_obj.get("awayTeam", {}).get("abbreviation")
                home_abbr = game_obj.get("homeTeam", {}).get("abbreviation")
                if away_abbr and home_abbr and local_date:
                    box["match_key"] = f"{season}_{local_date}_{away_abbr}_{home_abbr}"
                else:
                    box["match_key"] = None
                if "officials" in box:
                    if isinstance(box["officials"], list):
                        box["officials_names"] = [official.get("name") for official in box["officials"] if
                                                  isinstance(official, dict)]
                    else:
                        box["officials_names"] = []
                if "weather" in box and isinstance(box["weather"], dict):
                    box["temperature"] = box["weather"].get("temperature", {}).get("fahrenheit")
                boxscore_list.append(box)
            pbp = details.get("playbyplay")
            if pbp and isinstance(pbp, dict) and "plays" in pbp:
                for play in pbp["plays"]:
                    play_record = {"season": season, "game_id": game_id, **play}
                    playbyplay_list.append(play_record)
    boxscore_df = pd.DataFrame(boxscore_list)
    playbyplay_df = pd.DataFrame(playbyplay_list)
    print("Detailed game data: boxscore shape =", boxscore_df.shape, "playbyplay shape =", playbyplay_df.shape)
    return boxscore_df, playbyplay_df


def feature_engineering(games_df, boxscore_df, playbyplay_df, team_stats_df, historical_games=None):
    df = games_df.copy()
    df["home_team_abbr"] = df["homeTeam"].apply(
        lambda x: x.get("abbreviation").upper().strip() if isinstance(x, dict) else str(x).upper().strip())
    df["away_team_abbr"] = df["awayTeam"].apply(
        lambda x: x.get("abbreviation").upper().strip() if isinstance(x, dict) else str(x).upper().strip())
    df["home_team_pts"] = df["home_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_offense.ptsPerGame"))
    df["away_team_pts"] = df["away_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_offense.ptsPerGame"))
    df["home_team_pts_allowed"] = df["home_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_defense.ptsAgainstPerGame"))
    df["away_team_pts_allowed"] = df["away_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_defense.ptsAgainstPerGame"))
    df["home_team_ast"] = df["home_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_offense.astPerGame"))
    df["away_team_ast"] = df["away_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_offense.astPerGame"))
    df["home_team_fgPct"] = df["home_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_fieldGoals.fgPct"))
    df["away_team_fgPct"] = df["away_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_fieldGoals.fgPct"))
    df["home_team_ftPct"] = df["home_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_freeThrows.ftPct"))
    df["away_team_ftPct"] = df["away_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_freeThrows.ftPct"))
    df["home_team_reb"] = df["home_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_rebounds.rebPerGame"))
    df["away_team_reb"] = df["away_team_abbr"].apply(
        lambda abbr: get_team_stat(team_stats_df, abbr, "stats_rebounds.rebPerGame"))
    df["def_diff"] = df["away_team_pts_allowed"] - df["home_team_pts_allowed"]
    df["predicted_home_score"] = (df["home_team_pts"] + df["away_team_pts_allowed"]) / 2
    df["predicted_away_score"] = (df["away_team_pts"] + df["home_team_pts_allowed"]) / 2
    df["predicted_point_diff"] = df["predicted_home_score"] - df["predicted_away_score"]
    df = enhance_feature_engineering(df, team_stats_df, boxscore_df, historical_games)
    return df


def train_game_prediction_model_with_optuna(merged_features, n_trials=1000):
    features = [
        "home_team_pts", "away_team_pts",
        "home_team_pts_allowed", "away_team_pts_allowed",
        "def_diff",
        "home_team_ast", "away_team_ast",
        "home_team_fgPct", "away_team_fgPct",
        "home_team_ftPct", "away_team_ftPct",
        "home_team_reb", "away_team_reb",
        "total_team_pts", "home_off_eff", "away_off_eff",
        "team_momentum_home", "team_momentum_away",
        "off_eff_diff", "momentum_diff", "points_eff_ratio",
        "home_team_pace", "away_team_pace",
        "home_team_adj_pts", "away_team_adj_pts",
        "home_team_opp_adj_pts_allowed", "away_team_opp_adj_pts_allowed",
        "predicted_point_diff_adj"
    ]
    features = [f for f in features if f in merged_features.columns]
    if "scoring_homeScoreTotal" in merged_features.columns and "scoring_awayScoreTotal" in merged_features.columns:
        merged_features["actual_point_diff"] = merged_features["scoring_homeScoreTotal"] - merged_features[
            "scoring_awayScoreTotal"]
    else:
        merged_features["actual_point_diff"] = merged_features["predicted_point_diff"]
    historical = merged_features.dropna(subset=["actual_point_diff"] + features)
    print("Training data shape:", historical.shape)
    X = historical[features]
    y = historical["actual_point_diff"]
    print("Correlation of features with actual point diff:")
    corr_matrix = historical[features + ["actual_point_diff"]].corr()
    print(corr_matrix["actual_point_diff"].sort_values(ascending=False))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective(trial, X_train, X_val, y_train, y_val), n_trials=n_trials)
    print("Best trial:", study.best_trial.params)
    best_params = study.best_trial.params
    ensemble_models = []
    n_models = 10
    for seed in range(n_models):
        model = XGBRegressor(**best_params, objective="reg:pseudohubererror", random_state=42 + seed)
        pipeline = Pipeline([("scaler", StandardScaler()), ("xgb", model)])
        pipeline.fit(X_train, y_train)
        ensemble_models.append(pipeline)

    def ensemble_predict(X):
        preds = np.mean([model.predict(X) for model in ensemble_models], axis=0)
        return preds

    y_pred_ensemble = ensemble_predict(X_val)
    mse_ensemble = mean_squared_error(y_val, y_pred_ensemble)
    print("Ensemble Validation MSE:", mse_ensemble)
    return ensemble_predict, features


def merge_team_features(player_df, team_stats_df):
    player_df["team_id"] = player_df["team"].apply(lambda x: x.get("id") if isinstance(x, dict) else x)
    if "team_abbr" not in player_df.columns or player_df["team_abbr"].isna().all():
        player_df["team_abbr"] = player_df["team"].apply(
            lambda x: x.get("abbreviation").upper().strip() if isinstance(x, dict) else str(x).upper().strip())
    else:
        print("team_abbr already set—preserving override.")
    print("Sample team abbreviations from player_df:", player_df["team_abbr"].head())
    if "team_abbr" in team_stats_df.columns:
        team_stats_df["team_abbr"] = team_stats_df["team_abbr"].apply(lambda x: str(x).upper().strip())
        team_stats_df = team_stats_df.set_index("team_abbr")
        player_df = player_df.merge(team_stats_df, how="left", left_on="team_abbr", right_index=True,
                                    suffixes=("", "_team"))
    else:
        player_df["team_id"] = player_df["team_id"].astype(int)
        team_stats_df.index = team_stats_df.index.astype(int)
        player_df = player_df.merge(team_stats_df, how="left", left_on="team_id", right_index=True,
                                    suffixes=("", "_team"))
    print("Merged player stats shape:", player_df.shape)
    return player_df


def train_player_stats_model(player_stats_df, daily_player_gamelogs_df, injuries_df, team_stats_df):
    logging.debug("player_stats_df shape before processing: %s", player_stats_df.shape)
    logging.debug("daily_player_gamelogs_df shape: %s", daily_player_gamelogs_df.shape)
    df = player_stats_df.copy()
    if "stats" in df.columns:
        stats_flat = pd.json_normalize(df["stats"]).add_prefix("stats_")
        df = df.drop(columns=["stats"]).reset_index(drop=True)
        df = pd.concat([df, stats_flat], axis=1)
    if "player_id" not in df.columns:
        df["player_id"] = df["player"].apply(lambda x: x.get("id") if isinstance(x, dict) else x)
    print("Player stats dataframe shape after loading:", df.shape)
    df = merge_team_features(df, team_stats_df)
    if "stats_offense.usageRate" not in df.columns:
        df = compute_usage_rate_for_players(df, team_stats_df)
    if daily_player_gamelogs_df.empty:
        logging.warning("No daily player gamelog data available; skipping rolling averages.")
        rolling_df = pd.DataFrame()
    else:
        try:
            rolling_weighted = compute_weighted_player_rolling_averages(daily_player_gamelogs_df, window=5, decay=0.8)
            print("Rolling averages dataframe shape:", rolling_weighted.shape)
        except Exception:
            logging.exception("Error computing weighted rolling averages from daily gamelogs.")
            rolling_weighted = pd.DataFrame()
        rolling_df = rolling_weighted if not rolling_weighted.empty else pd.DataFrame()
    if not rolling_df.empty and "player_id" in rolling_df.columns:
        df = pd.merge(df, rolling_df, on="player_id", how="left")
        print("Player stats dataframe shape after merging rolling averages:", df.shape)
    else:
        logging.warning("Skipping merge of rolling averages due to missing data.")
    if injuries_df is not None and not injuries_df.empty:
        df["player_id"] = df["player_id"].astype(str)
        injuries_df["player_id"] = injuries_df["player_id"].astype(str)
        df = pd.merge(df, injuries_df, on="player_id", how="left")
        df["injury_factor"] = df["injury_factor"].fillna(0)
    else:
        df["injury_factor"] = 0
    print("Player stats dataframe shape after merging injuries:", df.shape)
    expected_features = [
        "stats_fieldGoals.fg2PtAttPerGame",
        "stats_fieldGoals.fg2PtMadePerGame",
        "stats_fieldGoals.fg3PtAttPerGame",
        "stats_fieldGoals.fgAttPerGame",
        "stats_fieldGoals.fgMadePerGame",
        "stats_fieldGoals.fgPct",
        "stats_freeThrows.ftAttPerGame",
        "stats_freeThrows.ftMadePerGame",
        "stats_freeThrows.ftPct",
        "stats_rebounds.rebPerGame",
        "stats_rebounds.offRebPerGame",
        "stats_rebounds.defRebPerGame",
        "weighted_rolling_pts",
        "weighted_rolling_ast",
        "weighted_rolling_reb",
        "weighted_rolling_fg3",
        "stats_miscellaneous.minSecondsPerGame",
        "injury_factor",
        "stats_defense.tovPerGame",
        "stats_defense.stlPerGame",
        "stats_defense.blkPerGame",
        "stats_offense.usageRate"
    ]
    target_columns = [
        "stats_offense.ptsPerGame",
        "stats_offense.astPerGame",
        "stats_rebounds.rebPerGame",
        "stats_fieldGoals.fg3PtMadePerGame"
    ]
    available_features = [col for col in expected_features if col in df.columns]
    available_targets = [col for col in target_columns if col in df.columns]
    print("Available features for player model:", available_features)
    print("Available targets for player model:", available_targets)
    print("Missing counts before dropping NA:")
    for col in available_features + available_targets:
        missing_count = df[col].isna().sum()
        print(f"  {col}: {missing_count} missing out of {len(df)} rows")
    df = df.dropna(subset=available_features + available_targets)
    print("Final training dataframe shape for player model:", df.shape)
    X = df[available_features]
    y = df[target_columns]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    from sklearn.multioutput import MultiOutputRegressor
    player_model = MultiOutputRegressor(
        XGBRegressor(objective="reg:pseudohubererror", n_estimators=500, learning_rate=0.01, random_state=42))
    player_model.fit(X_train, y_train)
    y_pred = player_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print("Player Stats Model - MSE:", mse)
    importances = np.mean([est.feature_importances_ for est in player_model.estimators_], axis=0)
    print("Player Stats Model Feature Importances:")
    for feat, imp in zip(available_features, importances):
        print(f"  {feat}: {imp:.4f}")
    return player_model, df, available_features, available_targets


def print_game_and_player_predictions(merged_features, player_stats_df, player_model, features, targets, odds_dict,
                                      props_odds, thresholds, team_stats_df):
    console = Console()
    value_bets_for_discord = []
    summary_table = Table(title="Game Predictions Summary", show_header=True, header_style="bold green")
    summary_table.add_column("Game", justify="center")
    summary_table.add_column("Predicted Score", justify="center")
    summary_table.add_column("Total", justify="center")
    for idx, game in merged_features.iterrows():
        home_abbr = game.get("home_team_abbr")
        away_abbr = game.get("away_team_abbr")
        predicted_home_score = game.get("predicted_home_score")
        predicted_away_score = game.get("predicted_away_score")
        total_score = predicted_home_score + predicted_away_score
        game_key = f"{away_abbr} vs {home_abbr}"
        summary_table.add_row(game_key,
                              f"{home_abbr} {predicted_home_score:.1f} - {away_abbr} {predicted_away_score:.1f}",
                              f"{total_score:.1f}")
    console.print(summary_table)
    console.print("\n")
    if "player_id" not in player_stats_df.columns:
        player_stats_df["player_id"] = player_stats_df["player"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else x)
    market_to_pred_key = {
        "player_points": "stats_offense.ptsPerGame",
        "player_assists": "stats_offense.astPerGame",
        "player_rebounds": "stats_rebounds.rebPerGame",
        "player_threes": "stats_fieldGoals.fg3PtMadePerGame"
    }
    label_mapping = {
        "player_points": "Points",
        "player_assists": "Assists",
        "player_rebounds": "Rebounds",
        "player_threes": "3s"
    }
    for idx, game in merged_features.iterrows():
        home_abbr = game.get("home_team_abbr")
        away_abbr = game.get("away_team_abbr")
        predicted_home_score = game.get("predicted_home_score")
        predicted_away_score = game.get("predicted_away_score")
        console.rule(f"[bold blue]Game: {away_abbr} vs {home_abbr}[/bold blue]")
        console.print(
            f"Predicted Score: [green]{home_abbr} {predicted_home_score:.1f}[/green] - [red]{away_abbr} {predicted_away_score:.1f}[/red]")
        console.print("[yellow]Odds not available for this matchup.[/yellow]")
        for team_label, abbr in [("Home", home_abbr), ("Away", away_abbr)]:
            players = player_stats_df[player_stats_df["team_abbr"] == abbr]
            console.print(f"\n[bold underline]{team_label} Team Players ({abbr})[/bold underline]")
            if players.empty:
                console.print(f"[yellow]No player predictions available for {team_label} team ({abbr}).[/yellow]")
                continue
            table = Table(title=f"{team_label} Team Players ({abbr})", show_header=True, header_style="bold cyan")
            table.add_column("Player", style="white", no_wrap=True)
            table.add_column("Predicted Points", justify="right")
            table.add_column("Fanduel Points", justify="right")
            table.add_column("Predicted Assists", justify="right")
            table.add_column("Fanduel Assists", justify="right")
            table.add_column("Predicted Rebounds", justify="right")
            table.add_column("Fanduel Rebounds", justify="right")
            table.add_column("Predicted 3s", justify="right")
            table.add_column("Fanduel 3s", justify="right")
            table.add_column("Value Bet", style="magenta")
            for _, player in players.iterrows():
                player_info = player.get("player")
                name = f"{player_info.get('firstName', '')} {player_info.get('lastName', '')}".strip() if isinstance(
                    player_info, dict) else str(player_info)
                name_upper = name.upper().strip()
                X_player = pd.DataFrame([player[features].fillna(0)])
                preds = player_model.predict(X_player)[0]
                pred_dict = dict(zip(targets, preds))
                predicted_points = pred_dict.get(market_to_pred_key["player_points"], 0)
                opponent_abbr = game.get("away_team_abbr") if team_label == "Home" else game.get("home_team_abbr")
                player_position = player_info.get("position") if isinstance(player_info, dict) else None
                adjusted_points = adjust_player_prop_for_defense(predicted_points, opponent_abbr, player_position,
                                                                 team_stats_df)
                fanduel_points = props_odds.get(f"{name_upper}_player_points", "-")
                predicted_assists = pred_dict.get(market_to_pred_key["player_assists"], 0)
                fanduel_assists = props_odds.get(f"{name_upper}_player_assists", "-")
                predicted_rebounds = pred_dict.get(market_to_pred_key["player_rebounds"], 0)
                fanduel_rebounds = props_odds.get(f"{name_upper}_player_rebounds", "-")
                predicted_threes = pred_dict.get(market_to_pred_key["player_threes"], 0)
                fanduel_threes = props_odds.get(f"{name_upper}_player_threes", "-")
                value_bets = []
                for market, pred_key in market_to_pred_key.items():
                    pred_value = pred_dict.get(pred_key, 0)
                    if market == "player_points":
                        pred_value = adjusted_points
                    key = f"{name_upper}_{market}"
                    if key in props_odds:
                        prop_line = props_odds[key]
                        diff = pred_value - prop_line
                        if abs(diff) > thresholds.get(market, 0):
                            if diff > 0:
                                value_bets.append(
                                    f"{label_mapping.get(market)}: {pred_value:.2f} vs. {prop_line:.2f} - over")
                                value_bets_for_discord.append(
                                    format_playbook_bet(name, market, prop_line, "over")
                                )
                            else:
                                value_bets.append(
                                    f"{label_mapping.get(market)}: {pred_value:.2f} vs. {prop_line:.2f} - under")
                                value_bets_for_discord.append(
                                    format_playbook_bet(name, market, prop_line, "under")
                                )
                value_bet_str = "\n".join(value_bets) if value_bets else "-"
                table.add_row(name, f"{adjusted_points:.2f}", f"{fanduel_points}", f"{predicted_assists:.2f}",
                              f"{fanduel_assists}", f"{predicted_rebounds:.2f}", f"{fanduel_rebounds}",
                              f"{predicted_threes:.2f}", f"{fanduel_threes}", value_bet_str)
            console.print(table)
        console.print("\n")
    if value_bets_for_discord:
        unique_bets = list(dict.fromkeys(value_bets_for_discord))
        send_discord_value_bets(unique_bets)


def add_rest_features(games_df, historical_games):
    def extract_abbr(team):
        if isinstance(team, dict):
            return team.get("abbreviation")
        return team

    games_df["home_team_abbr"] = games_df["homeTeam"].apply(lambda x: extract_abbr(x))
    games_df["away_team_abbr"] = games_df["awayTeam"].apply(lambda x: extract_abbr(x))

    def days_rest_for_team(team_abbr):
        team_games = historical_games[historical_games.apply(
            lambda row: (extract_abbr(row.get("homeTeam")) == team_abbr) or (
                        extract_abbr(row.get("awayTeam")) == team_abbr), axis=1)]
        if not team_games.empty:
            last_date = team_games["local_date"].max()
            return (TARGET_DATE - last_date).days
        return None

    games_df["home_team_days_rest"] = games_df["home_team_abbr"].apply(lambda abbr: days_rest_for_team(abbr))
    games_df["away_team_days_rest"] = games_df["away_team_abbr"].apply(lambda abbr: days_rest_for_team(abbr))
    games_df["home_team_back_to_back"] = games_df["home_team_days_rest"].apply(
        lambda x: 1 if x is not None and x <= 1 else 0)
    games_df["away_team_back_to_back"] = games_df["away_team_days_rest"].apply(
        lambda x: 1 if x is not None and x <= 1 else 0)
    return games_df


def compute_win_probability(predicted_point_diff, scale=10.0):
    return 1.0 / (1.0 + np.exp(-predicted_point_diff / scale))


def kelly_criterion(probability, odds, fraction=1.0):
    edge = (probability * (odds - 1)) - (1 - probability)
    if edge <= 0:
        return 0.0
    kelly_fraction = edge / (odds - 1)
    return kelly_fraction * fraction


def fetch_daily_player_gamelogs_for_teams(season, date_obj, teams):
    date_str = date_obj.strftime("%Y%m%d")
    print("Fetching daily player gamelogs for date:", date_str)
    all_gamelogs = []
    for team in teams:
        url = endpoints["daily_player_gamelogs"].format(season=season, date=date_str, team_abbr=team)
        print(f"Fetching URL for team {team}: {url}")
        data = fetch_api_data(url)
        if data and isinstance(data, dict) and "gamelogs" in data:
            if data["gamelogs"]:
                all_gamelogs.extend(data["gamelogs"])
            else:
                print(f"No gamelogs found for team {team} on date {date_str}")
        else:
            print(f"Invalid data format for team {team} on date {date_str}")
    df_logs = pd.DataFrame(all_gamelogs)
    print(f"Daily player gamelogs for {date_str} loaded. Shape: {df_logs.shape}")
    return df_logs


def optuna_objective(trial, X_train, X_val, y_train, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
    model = XGBRegressor(**params, objective="reg:pseudohubererror", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)


def train_individual_models(X, y, trial):
    models = {}
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    for target in y.columns:
        model = XGBRegressor(objective='reg:squarederror', **params)
        model.fit(X, y[target], verbose=False)
        models[target] = model
    return models


import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_player_stats_model_with_optuna(player_stats_df, daily_player_gamelogs_df, injuries_df, team_stats_df,
                                         games_df, n_trials=10000, n_jobs=4):
    df = player_stats_df.copy()
    if "stats" in df.columns:
        stats_flat = pd.json_normalize(df["stats"]).add_prefix("stats_")
        df = pd.concat([df.drop(columns=["stats"]), stats_flat], axis=1)
    if "player_id" not in df.columns:
        df["player_id"] = df["player"].apply(lambda x: x.get("id") if isinstance(x, dict) else x)
    print("Player stats dataframe shape after loading:", df.shape)
    df = merge_team_features(df, team_stats_df)
    if "stats_offense.usageRate" not in df.columns:
        df = compute_usage_rate_for_players(df, team_stats_df)
    if daily_player_gamelogs_df.empty:
        logging.warning("No daily player gamelog data available; skipping rolling averages.")
        rolling_df = pd.DataFrame()
    else:
        try:
            rolling_weighted = compute_weighted_player_rolling_averages(daily_player_gamelogs_df, window=5, decay=0.8)
            print("Rolling averages dataframe shape:", rolling_weighted.shape)
        except Exception:
            logging.exception("Error computing weighted rolling averages from daily gamelogs.")
            rolling_weighted = pd.DataFrame()
        rolling_df = rolling_weighted if not rolling_weighted.empty else pd.DataFrame()
    if not rolling_df.empty and "player_id" in rolling_df.columns:
        df = pd.merge(df, rolling_df, on="player_id", how="left")
        print("Player stats dataframe shape after merging rolling averages:", df.shape)
    else:
        logging.warning("Skipping merge of rolling averages due to missing data.")
    if injuries_df is not None and not injuries_df.empty:
        df["player_id"] = df["player_id"].astype(str)
        injuries_df["player_id"] = injuries_df["player_id"].astype(str)
        df = pd.merge(df, injuries_df, on="player_id", how="left")
        df["injury_factor"] = df["injury_factor"].fillna(0)
    else:
        df["injury_factor"] = 0
    print("Player stats dataframe shape after merging injuries:", df.shape)
    expected_features = [
        "stats_fieldGoals.fg2PtAttPerGame",
        "stats_fieldGoals.fg2PtMadePerGame",
        "stats_fieldGoals.fg3PtAttPerGame",
        "stats_fieldGoals.fgAttPerGame",
        "stats_fieldGoals.fgMadePerGame",
        "stats_fieldGoals.fgPct",
        "stats_freeThrows.ftAttPerGame",
        "stats_freeThrows.ftMadePerGame",
        "stats_freeThrows.ftPct",
        "stats_rebounds.rebPerGame",
        "stats_rebounds.offRebPerGame",
        "stats_rebounds.defRebPerGame",
        "weighted_rolling_pts",
        "weighted_rolling_ast",
        "weighted_rolling_reb",
        "weighted_rolling_fg3",
        "stats_miscellaneous.minSecondsPerGame",
        "injury_factor",
        "stats_defense.tovPerGame",
        "stats_defense.stlPerGame",
        "stats_defense.blkPerGame",
        "stats_offense.usageRate"
    ]
    target_columns = [
        "stats_offense.ptsPerGame",
        "stats_offense.astPerGame",
        "stats_rebounds.rebPerGame",
        "stats_fieldGoals.fg3PtMadePerGame"
    ]
    available_features = [col for col in expected_features if col in df.columns]
    available_targets = [col for col in target_columns if col in df.columns]
    print("Available features for player model:", available_features)
    print("Available targets for player model:", available_targets)
    print("Missing counts before dropping NA:")
    for col in available_features + available_targets:
        missing_count = df[col].isna().sum()
        print(f"  {col}: {missing_count} missing out of {len(df)} rows")
    df = df.dropna(subset=available_features + available_targets)
    print("Final training dataframe shape for player model:", df.shape)
    X = df[available_features]
    y = df[target_columns]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    from sklearn.multioutput import MultiOutputRegressor
    player_model = MultiOutputRegressor(
        XGBRegressor(objective="reg:pseudohubererror", n_estimators=500, learning_rate=0.01, random_state=42))
    player_model.fit(X_train, y_train)
    y_pred = player_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print("Player Stats Model - MSE:", mse)
    importances = np.mean([est.feature_importances_ for est in player_model.estimators_], axis=0)
    print("Player Stats Model Feature Importances:")
    for feat, imp in zip(available_features, importances):
        print(f"  {feat}: {imp:.4f}")
    return player_model, df, available_features, available_targets


def print_game_and_player_predictions(merged_features, player_stats_df, player_model, features, targets, odds_dict,
                                      props_odds, thresholds, team_stats_df):
    console = Console()
    value_bets_for_discord = []
    summary_table = Table(title="Game Predictions Summary", show_header=True, header_style="bold green")
    summary_table.add_column("Game", justify="center")
    summary_table.add_column("Predicted Score", justify="center")
    summary_table.add_column("Total", justify="center")
    for idx, game in merged_features.iterrows():
        home_abbr = game.get("home_team_abbr")
        away_abbr = game.get("away_team_abbr")
        predicted_home_score = game.get("predicted_home_score")
        predicted_away_score = game.get("predicted_away_score")
        total_score = predicted_home_score + predicted_away_score
        game_key = f"{away_abbr} vs {home_abbr}"
        summary_table.add_row(game_key,
                              f"{home_abbr} {predicted_home_score:.1f} - {away_abbr} {predicted_away_score:.1f}",
                              f"{total_score:.1f}")
    console.print(summary_table)
    console.print("\n")
    if "player_id" not in player_stats_df.columns:
        player_stats_df["player_id"] = player_stats_df["player"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else x)
    market_to_pred_key = {
        "player_points": "stats_offense.ptsPerGame",
        "player_assists": "stats_offense.astPerGame",
        "player_rebounds": "stats_rebounds.rebPerGame",
        "player_threes": "stats_fieldGoals.fg3PtMadePerGame"
    }
    label_mapping = {
        "player_points": "Points",
        "player_assists": "Assists",
        "player_rebounds": "Rebounds",
        "player_threes": "3s"
    }
    for idx, game in merged_features.iterrows():
        home_abbr = game.get("home_team_abbr")
        away_abbr = game.get("away_team_abbr")
        predicted_home_score = game.get("predicted_home_score")
        predicted_away_score = game.get("predicted_away_score")
        console.rule(f"[bold blue]Game: {away_abbr} vs {home_abbr}[/bold blue]")
        console.print(
            f"Predicted Score: [green]{home_abbr} {predicted_home_score:.1f}[/green] - [red]{away_abbr} {predicted_away_score:.1f}[/red]")
        console.print("[yellow]Odds not available for this matchup.[/yellow]")
        for team_label, abbr in [("Home", home_abbr), ("Away", away_abbr)]:
            players = player_stats_df[player_stats_df["team_abbr"] == abbr]
            console.print(f"\n[bold underline]{team_label} Team Players ({abbr})[/bold underline]")
            if players.empty:
                console.print(f"[yellow]No player predictions available for {team_label} team ({abbr}).[/yellow]")
                continue
            table = Table(title=f"{team_label} Team Players ({abbr})", show_header=True, header_style="bold cyan")
            table.add_column("Player", style="white", no_wrap=True)
            table.add_column("Predicted Points", justify="right")
            table.add_column("Fanduel Points", justify="right")
            table.add_column("Predicted Assists", justify="right")
            table.add_column("Fanduel Assists", justify="right")
            table.add_column("Predicted Rebounds", justify="right")
            table.add_column("Fanduel Rebounds", justify="right")
            table.add_column("Predicted 3s", justify="right")
            table.add_column("Fanduel 3s", justify="right")
            table.add_column("Value Bet", style="magenta")
            for _, player in players.iterrows():
                player_info = player.get("player")
                name = f"{player_info.get('firstName', '')} {player_info.get('lastName', '')}".strip() if isinstance(
                    player_info, dict) else str(player_info)
                name_upper = name.upper().strip()
                X_player = pd.DataFrame([player[features].fillna(0)])
                preds = player_model.predict(X_player)[0]
                pred_dict = dict(zip(targets, preds))
                predicted_points = pred_dict.get(market_to_pred_key["player_points"], 0)
                opponent_abbr = game.get("away_team_abbr") if team_label == "Home" else game.get("home_team_abbr")
                player_position = player_info.get("position") if isinstance(player_info, dict) else None
                adjusted_points = adjust_player_prop_for_defense(predicted_points, opponent_abbr, player_position,
                                                                 team_stats_df)
                fanduel_points = props_odds.get(f"{name_upper}_player_points", "-")
                predicted_assists = pred_dict.get(market_to_pred_key["player_assists"], 0)
                fanduel_assists = props_odds.get(f"{name_upper}_player_assists", "-")
                predicted_rebounds = pred_dict.get(market_to_pred_key["player_rebounds"], 0)
                fanduel_rebounds = props_odds.get(f"{name_upper}_player_rebounds", "-")
                predicted_threes = pred_dict.get(market_to_pred_key["player_threes"], 0)
                fanduel_threes = props_odds.get(f"{name_upper}_player_threes", "-")
                value_bets = []
                for market, pred_key in market_to_pred_key.items():
                    pred_value = pred_dict.get(pred_key, 0)
                    if market == "player_points":
                        pred_value = adjusted_points
                    key = f"{name_upper}_{market}"
                    if key in props_odds:
                        prop_line = props_odds[key]
                        diff = pred_value - prop_line
                        if abs(diff) > thresholds.get(market, 0):
                            if diff > 0:
                                value_bets.append(
                                    f"{label_mapping.get(market)}: {pred_value:.2f} vs. {prop_line:.2f} - over")
                                value_bets_for_discord.append(
                                    format_playbook_bet(name, market, prop_line, "over")
                                )
                            else:
                                value_bets.append(
                                    f"{label_mapping.get(market)}: {pred_value:.2f} vs. {prop_line:.2f} - under")
                                value_bets_for_discord.append(
                                    format_playbook_bet(name, market, prop_line, "under")
                                )
                value_bet_str = "\n".join(value_bets) if value_bets else "-"
                table.add_row(name, f"{adjusted_points:.2f}", f"{fanduel_points}", f"{predicted_assists:.2f}",
                              f"{fanduel_assists}", f"{predicted_rebounds:.2f}", f"{fanduel_rebounds}",
                              f"{predicted_threes:.2f}", f"{fanduel_threes}", value_bet_str)
            console.print(table)
        console.print("\n")
    if value_bets_for_discord:
        unique_bets = list(dict.fromkeys(value_bets_for_discord))
        send_discord_value_bets(unique_bets)


def add_rest_features(games_df, historical_games):
    def extract_abbr(team):
        if isinstance(team, dict):
            return team.get("abbreviation")
        return team

    games_df["home_team_abbr"] = games_df["homeTeam"].apply(lambda x: extract_abbr(x))
    games_df["away_team_abbr"] = games_df["awayTeam"].apply(lambda x: extract_abbr(x))

    def days_rest_for_team(team_abbr):
        team_games = historical_games[historical_games.apply(
            lambda row: (extract_abbr(row.get("homeTeam")) == team_abbr) or (
                        extract_abbr(row.get("awayTeam")) == team_abbr), axis=1)]
        if not team_games.empty:
            last_date = team_games["local_date"].max()
            return (TARGET_DATE - last_date).days
        return None

    games_df["home_team_days_rest"] = games_df["home_team_abbr"].apply(lambda abbr: days_rest_for_team(abbr))
    games_df["away_team_days_rest"] = games_df["away_team_abbr"].apply(lambda abbr: days_rest_for_team(abbr))
    games_df["home_team_back_to_back"] = games_df["home_team_days_rest"].apply(
        lambda x: 1 if x is not None and x <= 1 else 0)
    games_df["away_team_back_to_back"] = games_df["away_team_days_rest"].apply(
        lambda x: 1 if x is not None and x <= 1 else 0)
    return games_df


def compute_win_probability(predicted_point_diff, scale=10.0):
    return 1.0 / (1.0 + np.exp(-predicted_point_diff / scale))


def kelly_criterion(probability, odds, fraction=1.0):
    edge = (probability * (odds - 1)) - (1 - probability)
    if edge <= 0:
        return 0.0
    kelly_fraction = edge / (odds - 1)
    return kelly_fraction * fraction


def fetch_daily_player_gamelogs_for_teams(season, date_obj, teams):
    date_str = date_obj.strftime("%Y%m%d")
    print("Fetching daily player gamelogs for date:", date_str)
    all_gamelogs = []
    for team in teams:
        url = endpoints["daily_player_gamelogs"].format(season=season, date=date_str, team_abbr=team)
        print(f"Fetching URL for team {team}: {url}")
        data = fetch_api_data(url)
        if data and isinstance(data, dict) and "gamelogs" in data:
            if data["gamelogs"]:
                all_gamelogs.extend(data["gamelogs"])
            else:
                print(f"No gamelogs found for team {team} on date {date_str}")
        else:
            print(f"Invalid data format for team {team} on date {date_str}")
    df_logs = pd.DataFrame(all_gamelogs)
    print(f"Daily player gamelogs for {date_str} loaded. Shape: {df_logs.shape}")
    return df_logs


# -------------------------------
# NEW: AutoML & Reinforcement Learning Integration
# -------------------------------
def train_rl_agent_for_betting(predictions, actuals, odds, total_timesteps=10000):
    # Create the custom Gym environment for betting
    env = BettingEnv(predictions, actuals, odds, initial_balance=1000)
    rl_agent = train_rl_agent_on_betting(env, total_timesteps=total_timesteps)
    return rl_agent


# -------------------------------
# Main Routine
# -------------------------------
def main():
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    pd.set_option("display.max_columns", None)

    print("Fetching games data...")
    try:
        df_games = flatten_games_data(endpoints["seasonal_games"].format(season="2024-2025"))
        print("Games dataframe shape:", df_games.shape)
    except Exception as e:
        logging.error("Failed to process games data: %s", e)

    if "match_key" in df_games.columns:
        df_games = deduplicate_by_average(df_games, "match_key")
        print("Deduplicated games dataframe shape:", df_games.shape)

    print("Fetching venues data...")
    try:
        df_venues, _ = flatten_venues_data(endpoints["seasonal_venues"].format(season="2024-2025"))
        print("Venues dataframe shape:", df_venues.shape)
    except Exception as e:
        logging.error("Failed to process venues data: %s", e)

    current_season_data, players_data, injuries_data, all_season_data, all_games_details = ingest_data()

    print("Parsing seasonal data...")
    games_df, venues_df, team_stats_df, player_stats_df, standings_df, updates_df = parse_seasonal_data(all_season_data)
    if "match_key" in games_df.columns:
        games_df = deduplicate_by_average(games_df, "match_key")
        print("Deduplicated parsed games dataframe shape:", games_df.shape)

    print("Games parsed shape:", games_df.shape)
    games_df["local_date"] = pd.to_datetime(games_df["local_date"]).dt.date
    player_stats_df = deduplicate_player_stats(player_stats_df, players_data)
    print("Deduplicated player stats dataframe shape:", player_stats_df.shape)
    player_stats_df = override_team_info(player_stats_df, players_data)
    print("Player stats dataframe shape after team override:", player_stats_df.shape)

    boxscore_df, playbyplay_df = parse_detailed_game_data(all_games_details)
    TEAM_LOCATIONS = get_team_locations(venues_df)
    teams_with_games = get_teams_with_games(games_df, TARGET_DATE)
    print("Teams with games on", TARGET_DATE, ":", teams_with_games)
    daily_player_gamelogs = fetch_daily_player_gamelogs_for_teams(seasons[-1], TARGET_DATE, teams_with_games)
    if daily_player_gamelogs.empty:
        print("No daily player gamelog data available for TARGET_DATE.")

    # Separate historical games and today's games.
    historical_games = games_df[games_df["local_date"] < TARGET_DATE].copy()
    todays_games = games_df[games_df["local_date"] == TARGET_DATE].copy()
    print(f"Total games ingested: {games_df.shape[0]}")
    print(f"Historical games (for training): {historical_games.shape}")
    print(f"Target date games (for prediction): {todays_games.shape}")

    # ---
    # INSERTED IMPROVEMENTS: Enhanced feature engineering for historical games.
    print("Starting feature engineering for historical games...")
    # Here we use the historical_games parsed earlier
    historical_games = add_temporal_features(historical_games, metric="predicted_home_score", span=3,
                                             add_lag_features=True, n_lags=2)
    merged_features_hist = feature_engineering(historical_games, boxscore_df, playbyplay_df, team_stats_df,
                                               historical_games)
    merged_features_hist = add_interaction_terms(merged_features_hist)
    merged_features_hist = handle_missing_and_outliers(merged_features_hist,
                                                       columns=["home_team_pts", "home_team_fgPct", "away_team_fgPct"],
                                                       use_robust_scaling=True)

    if "stats" in team_stats_df.columns:
        team_stats_flat = pd.json_normalize(team_stats_df["stats"]).add_prefix("stats_")
        team_stats_df = pd.concat([team_stats_df.drop(columns=["stats"]), team_stats_flat], axis=1)
    historical_games = games_df[games_df["local_date"] < TARGET_DATE].copy()
    todays_games = games_df[games_df["local_date"] == TARGET_DATE].copy()
    print(f"Total games ingested: {games_df.shape[0]}")
    print(f"Historical games (for training): {historical_games.shape}")
    print(f"Target date games (for prediction): {todays_games.shape}")
    todays_games = add_rest_features(todays_games, historical_games)
    todays_games = add_schedule_congestion(todays_games, window_days=7)
    todays_games = add_travel_distance_feature(todays_games, TEAM_LOCATIONS, historical_games)
    print("Starting feature engineering for historical games...")
    merged_features_hist = feature_engineering(historical_games, boxscore_df, playbyplay_df, team_stats_df,
                                               historical_games)
    print("Historical features shape:", merged_features_hist.shape)
    print("Starting feature engineering for today's games...")
    merged_features_today = feature_engineering(todays_games, boxscore_df, playbyplay_df, team_stats_df,
                                                historical_games)
    merged_features_today = enhance_feature_engineering(merged_features_today, team_stats_df,
                                                        historical_games=historical_games)
    for idx, row in merged_features_today.iterrows():
        game_id = row.get("game_id")
        if not game_id:
            continue
        in_game_feats = extract_in_game_features(playbyplay_df, game_id)
        if in_game_feats:
            pre_game_diff = row.get("predicted_point_diff", 0)
            updated_diff = blend_predictions(pre_game_diff, in_game_feats)
            merged_features_today.at[idx, "predicted_point_diff"] = updated_diff
            base_avg = (row.get("predicted_home_score", 0) + row.get("predicted_away_score", 0)) / 2
            merged_features_today.at[idx, "predicted_home_score"] = base_avg + updated_diff / 2
            merged_features_today.at[idx, "predicted_away_score"] = base_avg - updated_diff / 2
    merged_features_today["predicted_home_score"] = merged_features_today.apply(
        lambda row: adjust_game_outcome_for_defense(row.get("predicted_home_score", 0), row.get("away_team_abbr"),
                                                    team_stats_df), axis=1)
    merged_features_today["predicted_away_score"] = merged_features_today.apply(
        lambda row: adjust_game_outcome_for_defense(row.get("predicted_away_score", 0), row.get("home_team_abbr"),
                                                    team_stats_df), axis=1)
    print("Final today's features shape:", merged_features_today.shape)
    print("Historical features shape:", merged_features_hist.shape)
    print("Today's features shape:", merged_features_today.shape)
    game_model, game_features = train_game_prediction_model_with_optuna_cv(merged_features_hist, n_trials=1000)
    if game_model is not None:
        X_today = merged_features_today[game_features].fillna(0)
        if X_today.shape[0] == 0:
            print("No games found for today's slate after feature prep; skipping game predictions.")
            return
        X_hist = merged_features_hist[game_features].fillna(0).values
        y_hist = merged_features_hist["actual_point_diff"].values
        nn_model = train_nn_game_model(X_hist, y_hist, epochs=200, batch_size=64)
        ensemble_preds = game_model.predict(X_today)
        nn_preds = nn_model.predict(X_today.values).flatten()
        combined_preds = (ensemble_preds + nn_preds) / 2.0
        historical_preds = game_model.predict(merged_features_hist[game_features].fillna(0))
        residuals = perform_residual_analysis(merged_features_hist, historical_preds)
        combined_preds_capped = cap_extreme_predictions(combined_preds, residuals, threshold=3)
        merged_features_today["predicted_point_diff"] = combined_preds_capped
        merged_features_today["predicted_home_score"] = merged_features_today.get("home_team_pts",
                                                                                  0) + combined_preds_capped / 2
        merged_features_today["predicted_away_score"] = merged_features_today.get("away_team_pts",
                                                                                  0) - combined_preds_capped / 2
        print(
            "Game prediction models (stacking ensemble and NN) trained. Predictions for today's games updated with combined model.")
        perform_residual_analysis(merged_features_hist, historical_preds)
        backtest_betting_strategy(merged_features_hist, game_model, game_features,
                                  {"player_points": 3.5, "player_assists": 2.5, "player_rebounds": 2.5,
                                   "player_threes": 2.5})
        schedule_retraining()
    else:
        print("Game-level model training was skipped; using pre-game predictions.")

    # --- AutoML Integration ---
    try:
        X_hist = merged_features_hist[game_features].fillna(0).values
        y_hist = merged_features_hist["actual_point_diff"].values
        auto_model, automl_history = train_autokeras_game_model(X_hist, y_hist, max_trials=3, epochs=30)
        auto_preds = auto_model.predict(merged_features_today[game_features].fillna(0).values).flatten()
        print("AutoML model predictions (first 5):", auto_preds[:5])
    except Exception as e:
        print("AutoML model training failed, using previous ensemble. Error:", e)

    nba_odds = get_nba_odds()  # fetch actual game odds

    # --- Reinforcement Learning Integration ---
    # For demonstration, we create dummy arrays (from historical predictions) as input to the RL environment.
    dummy_predictions = merged_features_hist["predicted_point_diff"].fillna(0).values
    dummy_actuals = merged_features_hist["actual_point_diff"].fillna(0).values
    # Use the new assign_game_odds to get an array of odds corresponding to each historical game.
    actual_odds = assign_game_odds(merged_features_hist, nba_odds, default_odds=2.0)

    try:
        rl_agent = train_rl_agent_for_betting(dummy_predictions, dummy_actuals, actual_odds, total_timesteps=1000)
        print("RL agent trained for betting adjustment using realistic odds.")
    except Exception as e:
        print("RL agent training failed. Error:", e)

    games_df["home_team_abbr"] = games_df["homeTeam"].apply(
        lambda x: x.get("abbreviation").upper().strip() if isinstance(x, dict) else str(x).upper().strip())
    games_df["away_team_abbr"] = games_df["awayTeam"].apply(
        lambda x: x.get("abbreviation").upper().strip() if isinstance(x, dict) else str(x).upper().strip())
    games_today_df = games_df[games_df['local_date'] == TARGET_DATE]
    print("games_today_df shape:", games_today_df.shape)
    nba_odds = get_nba_odds()
    player_model, player_stats_ready_df, feat_cols, target_cols = train_player_stats_model_with_optuna(
        player_stats_df,
        daily_player_gamelogs,
        process_injuries(injuries_data),
        team_stats_df,
        games_today_df,
        n_trials=1000
    )

    X = player_stats_ready_df[feat_cols].values
    y = player_stats_ready_df[target_cols].values
    nn_player_model, history = train_nn_player_model(X, y, epochs=200, batch_size=64)
    player_stats_ready_df = override_team_info(player_stats_ready_df, players_data)
    print("Player models trained successfully.")
    if player_model is not None:
        print("Player stats model training completed.")
    else:
        print("Player stats model training failed.")
    global props_odds
    props_odds = get_all_player_props_odds(nba_odds)
    thresholds = {"player_points": 3.5, "player_assists": 2.5, "player_rebounds": 2.5, "player_threes": 2.5}
    print_game_and_player_predictions(merged_features_today, player_stats_ready_df, player_model, feat_cols,
                                      target_cols, nba_odds, props_odds, thresholds, team_stats_df)
    predictions_html = build_html_predictions(merged_features_today, player_stats_ready_df, player_model, feat_cols,
                                              target_cols, nba_odds, props_odds, thresholds, team_stats_df)
    send_email(subject="NBA Predictions Output", plain_body="Here is the plain text fallback if HTML is not supported.",
               to_email="jjohnson0636@gmail.com", html_body=predictions_html)
    print("Main routine processing complete.")


if __name__ == '__main__':
    init_api_cache()
    get_http_session()
    main()
