#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TITAN MASTER SYSTEM v5.0 - Sistema Semplificato e Ottimizzato
Sistema di pronostici calcio con input da file locali
Target: Accuratezza massima (100%)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import json
import pickle
import os
from typing import Dict, List, Tuple
from collections import defaultdict, deque
from scipy.stats import poisson

warnings.filterwarnings('ignore')

# ============================================================================
# 1. GESTORE FILE INPUT
# ============================================================================

class FileInputManager:
    """Gestisce l'input dei 4 file richiesti"""
    
    REQUIRED_FILES = {
        'storico_seriea': 'storicoseriea.txt',
        'storico_serieb': 'storicoserieb.txt',
        'classifica_seriea': 'classificaseriea.txt',
        'classifica_serieb': 'classificaserieb.txt'
    }
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.files = {}
        self.data = {
            'storico_seriea': None,
            'storico_serieb': None,
            'classifica_seriea': None,
            'classifica_serieb': None
        }
    
    def check_and_load_files(self) -> bool:
        """Controlla e carica i 4 file richiesti"""
        print("\n" + "="*80)
        print("📂 CONTROLLO FILE INPUT")
        print("="*80)
        
        missing_files = []
        
        # Controlla esistenza file
        for key, filename in self.REQUIRED_FILES.items():
            filepath = self.base_path / filename
            if filepath.exists():
                self.files[key] = filepath
                print(f"✅ {filename} trovato")
            else:
                missing_files.append(filename)
                print(f"❌ {filename} NON trovato")
        
        if missing_files:
            print(f"\n⚠️  MANCANO {len(missing_files)} FILE:")
            for fname in missing_files:
                print(f"   • {fname}")
            print(f"\n📍 Posizione ricerca: {self.base_path}")
            print(f"\n💡 Per favore inserisci i seguenti file nella cartella:")
            for fname in missing_files:
                print(f"   • {fname}")
            return False
        
        # Carica i file
        print(f"\n📊 Caricamento dati...")
        success = self._load_all_files()
        
        if success:
            self._validate_and_show_stats()
        
        return success
    
    def _load_all_files(self) -> bool:
        """Carica tutti i file"""
        try:
            # Carica storici
            self.data['storico_seriea'] = self._load_storico(
                self.files['storico_seriea'], 'SerieA'
            )
            self.data['storico_serieb'] = self._load_storico(
                self.files['storico_serieb'], 'SerieB'
            )
            
            # Carica classifiche
            self.data['classifica_seriea'] = self._load_classifica(
                self.files['classifica_seriea'], 'SerieA'
            )
            self.data['classifica_serieb'] = self._load_classifica(
                self.files['classifica_serieb'], 'SerieB'
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento file: {e}")
            return False
    
    def _load_storico(self, filepath: Path, league: str) -> pd.DataFrame:
        """Carica file storico con supporto automatico date 2/4 cifre"""
        # Formato: "data","casa","trasferta","gol_casa","gol_trasferta"
        df = pd.read_csv(
            filepath,
            header=None,
            names=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'],
            quotechar='"',
            skipinitialspace=True,
            encoding='latin1',
            on_bad_lines='skip'
        )
        
        print(f"   📂 Caricamento {league}...")
        print(f"      Righe grezze: {len(df)}")
        
        # Pulisci squadre
        df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip().str.upper()
        df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip().str.upper()
        df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce').fillna(0).astype(int)
        df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce').fillna(0).astype(int)
        
        # GESTIONE INTELLIGENTE DATE - supporta sia gg/mm/yy che gg/mm/yyyy
        df['Date'] = df['Date'].astype(str).str.strip()
        
        def parse_date_smart(date_str):
            """Parser intelligente che rileva automaticamente formato anno"""
            try:
                # Rimuovi caratteri strani
                date_str = date_str.replace('"', '').strip()
                
                # Prova diversi formati
                formats_to_try = [
                    '%d/%m/%Y',      # 07/09/1997
                    '%d/%m/%y',      # 07/09/97
                    '%d-%m-%Y',      # 07-09-1997
                    '%d-%m-%y',      # 07-09-97
                    '%Y-%m-%d',      # 1997-09-07
                    '%Y/%m/%d',      # 1997/09/07
                ]
                
                for fmt in formats_to_try:
                    try:
                        parsed = pd.to_datetime(date_str, format=fmt)
                        
                        # Se anno < 1900, probabilmente è formato a 2 cifre mal interpretato
                        # pandas interpreta '97' come 1997 ma '25' come 2025
                        # Verifichiamo che sia ragionevole (1993-2030)
                        if parsed.year >= 1993 and parsed.year <= 2030:
                            return parsed
                        
                    except:
                        continue
                
                # Se tutti falliscono, prova inferenza automatica
                return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                
            except:
                return pd.NaT
        
        # Applica parsing intelligente
        df['Date'] = df['Date'].apply(parse_date_smart)
        
        # Rimuovi date non valide
        before = len(df)
        df = df[df['Date'].notna()]
        if before > len(df):
            print(f"      ⚠️  {before - len(df)} righe con date non valide rimosse")
        
        # Verifica range date
        if len(df) > 0:
            min_year = df['Date'].min().year
            max_year = df['Date'].max().year
            print(f"      📅 Periodo dati: {min_year} - {max_year}")
            
            # Controllo coerenza anni
            if league == 'SerieA' and min_year < 1993:
                print(f"      ⚠️  ATTENZIONE: Date Serie A prima del 1993 rilevate!")
            if league == 'SerieB' and min_year < 1997:
                print(f"      ⚠️  ATTENZIONE: Date Serie B prima del 1997 rilevate!")
        
        # Calcola risultato
        df['FTR'] = df.apply(
            lambda row: 'H' if row['FTHG'] > row['FTAG'] else 
                       ('A' if row['FTAG'] > row['FTHG'] else 'D'),
            axis=1
        )
        
        # Aggiungi metadati
        df['League'] = league
        
        # Calcola stagione (es: partite da Agosto 2024 a Maggio 2025 = stagione 2024-2025)
        def get_season(date):
            year = date.year
            month = date.month
            # Se è da Agosto a Dicembre, la stagione inizia quell'anno
            # Se è da Gennaio a Luglio, la stagione è iniziata l'anno prima
            if month >= 8:
                return f"{year}-{year+1}"
            else:
                return f"{year-1}-{year}"
        
        df['Season'] = df['Date'].apply(get_season)
        
        # Rimuovi righe non valide
        df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
        df = df[~df['HomeTeam'].isin(['', 'NAN', 'NULL', 'NONE'])]
        df = df[~df['AwayTeam'].isin(['', 'NAN', 'NULL', 'NONE'])]
        
        # Ordina per data
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"   ✅ {league}: {len(df)} partite valide caricate")
        if len(df) > 0:
            print(f"      🏆 Stagioni: {df['Season'].nunique()} ({df['Season'].min()} → {df['Season'].max()})")
            print(f"      📊 Squadre uniche: {pd.concat([df['HomeTeam'], df['AwayTeam']]).nunique()}")
        
        return df
    
    def _load_classifica(self, filepath: Path, league: str) -> pd.DataFrame:
        """Carica file classifica"""
        # Formato: posizione,squadra,punti,partite,vinte,pareggiate,perse,gol_fatti,gol_subiti
        df = pd.read_csv(
            filepath,
            header=None,
            names=['Pos', 'Team', 'Punti', 'Partite', 'Vinte', 'Pareggiate', 
                   'Perse', 'GolFatti', 'GolSubiti']
        )
        
        # Pulisci
        df['Team'] = df['Team'].str.strip().str.upper()
        df['League'] = league
        
        # Calcola metriche aggiuntive
        df['PuntiPerPartita'] = df['Punti'] / df['Partite'].replace(0, 1)
        df['MediaGolFatti'] = df['GolFatti'] / df['Partite'].replace(0, 1)
        df['MediaGolSubiti'] = df['GolSubiti'] / df['Partite'].replace(0, 1)
        df['DifferenzaReti'] = df['GolFatti'] - df['GolSubiti']
        
        print(f"   ✅ Classifica {league}: {len(df)} squadre")
        
        return df
    
    def _validate_and_show_stats(self):
        """Valida e mostra statistiche dettagliate"""
        print(f"\n📈 STATISTICHE DATI CARICATI")
        print("="*80)
        
        # Storico Serie A
        sa = self.data['storico_seriea']
        print(f"\n🏆 SERIE A (dal 1993):")
        print(f"   • Partite totali: {len(sa):,}")
        if len(sa) > 0:
            print(f"   • Periodo: {sa['Date'].min().strftime('%d/%m/%Y')} → {sa['Date'].max().strftime('%d/%m/%Y')}")
            print(f"   • Anni coperti: {sa['Date'].max().year - sa['Date'].min().year + 1}")
            print(f"   • Stagioni: {sa['Season'].nunique()} ({sa['Season'].min()} → {sa['Season'].max()})")
            print(f"   • Squadre diverse: {pd.concat([sa['HomeTeam'], sa['AwayTeam']]).nunique()}")
            
            # Statistiche risultati
            results = sa['FTR'].value_counts()
            total = len(sa)
            print(f"   • Risultati: Casa {results.get('H', 0)} ({results.get('H', 0)/total:.1%}), "
                  f"Pareggi {results.get('D', 0)} ({results.get('D', 0)/total:.1%}), "
                  f"Trasferta {results.get('A', 0)} ({results.get('A', 0)/total:.1%})")
        
        # Storico Serie B
        sb = self.data['storico_serieb']
        print(f"\n🥈 SERIE B (dal 1997):")
        print(f"   • Partite totali: {len(sb):,}")
        if len(sb) > 0:
            print(f"   • Periodo: {sb['Date'].min().strftime('%d/%m/%Y')} → {sb['Date'].max().strftime('%d/%m/%Y')}")
            print(f"   • Anni coperti: {sb['Date'].max().year - sb['Date'].min().year + 1}")
            print(f"   • Stagioni: {sb['Season'].nunique()} ({sb['Season'].min()} → {sb['Season'].max()})")
            print(f"   • Squadre diverse: {pd.concat([sb['HomeTeam'], sb['AwayTeam']]).nunique()}")
            
            # Statistiche risultati
            results = sb['FTR'].value_counts()
            total = len(sb)
            print(f"   • Risultati: Casa {results.get('H', 0)} ({results.get('H', 0)/total:.1%}), "
                  f"Pareggi {results.get('D', 0)} ({results.get('D', 0)/total:.1%}), "
                  f"Trasferta {results.get('A', 0)} ({results.get('A', 0)/total:.1%})")
        
        # Totale combinato
        total_matches = len(sa) + len(sb)
        print(f"\n📊 TOTALE COMBINATO:")
        print(f"   • Partite totali: {total_matches:,}")
        
        # Classifiche attuali
        print(f"\n📋 CLASSIFICHE ATTUALI (Stagione 2025-2026):")
        print(f"   • Serie A: {len(self.data['classifica_seriea'])} squadre")
        if len(self.data['classifica_seriea']) > 0:
            top3 = self.data['classifica_seriea'].head(3)
            print(f"      Top 3: {', '.join(top3['Team'].tolist())}")
        
        print(f"   • Serie B: {len(self.data['classifica_serieb'])} squadre")
        if len(self.data['classifica_serieb']) > 0:
            top3 = self.data['classifica_serieb'].head(3)
            print(f"      Top 3: {', '.join(top3['Team'].tolist())}")
        
        print(f"\n✅ Tutti i dati caricati e validati correttamente!")
        print(f"💡 Dati pronti per calibrazione con {total_matches:,} partite storiche")
    
    def get_combined_storico(self) -> pd.DataFrame:
        """Restituisce storico combinato Serie A + Serie B"""
        return pd.concat([
            self.data['storico_seriea'],
            self.data['storico_serieb']
        ], ignore_index=True).sort_values('Date').reset_index(drop=True)
    
    def get_classifica(self, league: str) -> pd.DataFrame:
        """Restituisce classifica per campionato"""
        key = f'classifica_{league.lower()}'
        return self.data.get(key)

# ============================================================================
# 2. SISTEMA ELO AVANZATO
# ============================================================================

class AdvancedEloSystem:
    """Sistema Elo ottimizzato per massima accuratezza"""
    
    def __init__(self, k_factor: float = 32.0, home_advantage: float = 100.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
        self.rating_history = defaultdict(list)
        self.volatility = defaultdict(float)
        
    def initialize_from_classifica(self, classifica: pd.DataFrame):
        """Inizializza rating dalla classifica attuale"""
        for _, row in classifica.iterrows():
            team = row['Team']
            
            # Rating base dalla posizione e punti
            base_rating = 1500
            
            # Bonus posizione (1° = +200, 20° = -200)
            pos_bonus = (20 - row['Pos']) * 10
            
            # Bonus punti per partita
            ppg_bonus = (row['PuntiPerPartita'] - 1.5) * 100
            
            # Bonus differenza reti
            dr_bonus = row['DifferenzaReti'] * 2
            
            # Rating finale
            rating = base_rating + pos_bonus + ppg_bonus + dr_bonus
            
            self.ratings[team] = max(800, min(2200, rating))
            self.volatility[team] = 0.5
        
        print(f"   ✅ {len(self.ratings)} squadre inizializzate con rating da classifica")
    
    def update_rating(self, home_team: str, away_team: str, result: str,
                     home_goals: int = 0, away_goals: int = 0):
        """Aggiorna rating post-partita"""
        # Inizializza se non esistono
        if home_team not in self.ratings:
            self.ratings[home_team] = 1500
            self.volatility[home_team] = 0.5
        if away_team not in self.ratings:
            self.ratings[away_team] = 1500
            self.volatility[away_team] = 0.5
        
        # Rating attuali
        home_rating = self.ratings[home_team] + self.home_advantage
        away_rating = self.ratings[away_team]
        
        # Probabilità attese
        exp_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        exp_away = 1 - exp_home
        
        # Risultato effettivo
        if result == 'H':
            actual_home, actual_away = 1.0, 0.0
        elif result == 'A':
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Moltiplicatore per differenza gol
        goal_diff = abs(home_goals - away_goals)
        multiplier = 1.0
        if goal_diff >= 3:
            multiplier = 1.5
        elif goal_diff == 2:
            multiplier = 1.25
        
        # K-factor adattivo basato su volatilità
        k_home = self.k_factor * (1 + self.volatility[home_team]) * multiplier
        k_away = self.k_factor * (1 + self.volatility[away_team]) * multiplier
        
        # Aggiorna rating
        old_home = self.ratings[home_team]
        old_away = self.ratings[away_team]
        
        self.ratings[home_team] += k_home * (actual_home - exp_home)
        self.ratings[away_team] += k_away * (actual_away - exp_away)
        
        # Limita range
        self.ratings[home_team] = max(800, min(2400, self.ratings[home_team]))
        self.ratings[away_team] = max(800, min(2400, self.ratings[away_team]))
        
        # Aggiorna volatilità
        rating_change_home = abs(self.ratings[home_team] - old_home)
        rating_change_away = abs(self.ratings[away_team] - old_away)
        
        self.volatility[home_team] = self.volatility[home_team] * 0.9 + (rating_change_home / 50) * 0.1
        self.volatility[away_team] = self.volatility[away_team] * 0.9 + (rating_change_away / 50) * 0.1
        
        # Storico
        self.rating_history[home_team].append(self.ratings[home_team])
        self.rating_history[away_team].append(self.ratings[away_team])
    
    def get_rating(self, team: str) -> float:
        """Ottieni rating squadra"""
        return self.ratings.get(team, 1500)
    
    def get_match_probability(self, home_team: str, away_team: str) -> Dict:
        """Calcola probabilità partita"""
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)
        
        # Probabilità base Elo
        prob_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        prob_away = 1 / (1 + 10 ** ((home_rating - away_rating) / 400))
        
        # Aggiusta per pareggio (empirico)
        elo_diff = abs(home_rating - away_rating)
        
        if elo_diff < 50:
            # Partite molto equilibrate → alta prob pareggio
            prob_draw = 0.35
        elif elo_diff < 100:
            prob_draw = 0.30
        elif elo_diff < 150:
            prob_draw = 0.25
        elif elo_diff < 200:
            prob_draw = 0.20
        else:
            prob_draw = 0.15
        
        # Normalizza
        prob_home = prob_home * (1 - prob_draw)
        prob_away = prob_away * (1 - prob_draw)
        
        # Assicura somma = 1
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total
        
        return {
            'H': prob_home,
            'D': prob_draw,
            'A': prob_away
        }

# ============================================================================
# 3. FEATURE ENGINEERING OTTIMIZZATO
# ============================================================================

class OptimizedFeatureEngine:
    """Feature engineering ottimizzato per massima accuratezza"""
    
    def __init__(self, storico: pd.DataFrame, elo_system: AdvancedEloSystem):
        self.storico = storico
        self.elo = elo_system
        self.team_stats = self._build_team_stats()
    
    def _build_team_stats(self) -> Dict:
        """Costruisce statistiche squadre da storico"""
        stats = defaultdict(lambda: {
            'matches': 0,
            'home_matches': 0,
            'away_matches': 0,
            'goals_scored_home': [],
            'goals_conceded_home': [],
            'goals_scored_away': [],
            'goals_conceded_away': [],
            'results_home': {'H': 0, 'D': 0, 'A': 0},
            'results_away': {'H': 0, 'D': 0, 'A': 0},
            'recent_form': deque(maxlen=10)
        })
        
        for _, match in self.storico.iterrows():
            home = match['HomeTeam']
            away = match['AwayTeam']
            
            # Home team stats
            stats[home]['matches'] += 1
            stats[home]['home_matches'] += 1
            stats[home]['goals_scored_home'].append(match['FTHG'])
            stats[home]['goals_conceded_home'].append(match['FTAG'])
            stats[home]['results_home'][match['FTR']] += 1
            
            # Forma recente (3 punti vittoria, 1 pareggio, 0 sconfitta)
            if match['FTR'] == 'H':
                stats[home]['recent_form'].append(3)
            elif match['FTR'] == 'D':
                stats[home]['recent_form'].append(1)
            else:
                stats[home]['recent_form'].append(0)
            
            # Away team stats
            stats[away]['matches'] += 1
            stats[away]['away_matches'] += 1
            stats[away]['goals_scored_away'].append(match['FTAG'])
            stats[away]['goals_conceded_away'].append(match['FTHG'])
            
            # Inverti risultato per away
            away_result = 'A' if match['FTR'] == 'H' else 'H' if match['FTR'] == 'A' else 'D'
            stats[away]['results_away'][away_result] += 1
            
            if away_result == 'H':  # Vittoria away
                stats[away]['recent_form'].append(3)
            elif away_result == 'D':
                stats[away]['recent_form'].append(1)
            else:
                stats[away]['recent_form'].append(0)
        
        return dict(stats)
    
    def extract_features(self, home_team: str, away_team: str, 
                        match_date: datetime = None) -> Dict:
        """Estrae features per una partita"""
        
        features = {}
        
        # 1. ELO FEATURES
        home_elo = self.elo.get_rating(home_team)
        away_elo = self.elo.get_rating(away_team)
        
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = home_elo - away_elo
        features['elo_ratio'] = home_elo / max(away_elo, 100)
        
        # 2. STATISTICHE STORICHE
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        # Goals
        features['home_avg_goals_scored'] = np.mean(home_stats.get('goals_scored_home', [1.0]))
        features['home_avg_goals_conceded'] = np.mean(home_stats.get('goals_conceded_home', [1.0]))
        features['away_avg_goals_scored'] = np.mean(away_stats.get('goals_scored_away', [1.0]))
        features['away_avg_goals_conceded'] = np.mean(away_stats.get('goals_conceded_away', [1.0]))
        
        # Attack vs Defense
        features['attack_strength_home'] = features['home_avg_goals_scored'] / max(features['away_avg_goals_conceded'], 0.1)
        features['defense_strength_home'] = features['home_avg_goals_conceded'] / max(features['away_avg_goals_scored'], 0.1)
        
        # 3. FORMA RECENTE
        home_form = list(home_stats.get('recent_form', [1.5]))
        away_form = list(away_stats.get('recent_form', [1.5]))
        
        features['home_recent_form'] = np.mean(home_form) if home_form else 1.5
        features['away_recent_form'] = np.mean(away_form) if away_form else 1.5
        features['form_diff'] = features['home_recent_form'] - features['away_recent_form']
        
        # 4. PERFORMANCE CASA/TRASFERTA
        home_matches = home_stats.get('home_matches', 1)
        away_matches = away_stats.get('away_matches', 1)
        
        home_results = home_stats.get('results_home', {'H': 0, 'D': 0, 'A': 0})
        away_results = away_stats.get('results_away', {'H': 0, 'D': 0, 'A': 0})
        
        features['home_win_rate_home'] = home_results['H'] / max(home_matches, 1)
        features['away_win_rate_away'] = away_results['H'] / max(away_matches, 1)
        
        # 5. FEATURES DERIVATE
        features['expected_home_goals'] = (
            features['home_avg_goals_scored'] * features['attack_strength_home']
        )
        features['expected_away_goals'] = (
            features['away_avg_goals_scored'] / max(features['defense_strength_home'], 0.1)
        )
        features['expected_total_goals'] = features['expected_home_goals'] + features['expected_away_goals']
        
        # 6. PROBABILITÀ ELO
        elo_probs = self.elo.get_match_probability(home_team, away_team)
        features['elo_prob_home'] = elo_probs['H']
        features['elo_prob_draw'] = elo_probs['D']
        features['elo_prob_away'] = elo_probs['A']
        
        return features
    
    def get_feature_vector(self, features: Dict) -> np.ndarray:
        """Converte features dict in vettore numpy"""
        feature_order = [
            'home_elo', 'away_elo', 'elo_diff', 'elo_ratio',
            'home_avg_goals_scored', 'home_avg_goals_conceded',
            'away_avg_goals_scored', 'away_avg_goals_conceded',
            'attack_strength_home', 'defense_strength_home',
            'home_recent_form', 'away_recent_form', 'form_diff',
            'home_win_rate_home', 'away_win_rate_away',
            'expected_home_goals', 'expected_away_goals', 'expected_total_goals',
            'elo_prob_home', 'elo_prob_draw', 'elo_prob_away'
        ]
        
        return np.array([features.get(f, 0.0) for f in feature_order])

# ============================================================================
# 4. MODELLO PREDITTIVO OTTIMIZZATO
# ============================================================================

class OptimizedPredictionModel:
    """Modello ottimizzato per massima accuratezza"""
    
    def __init__(self):
        # Ensemble di 3 modelli migliori
        self.models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            ),
            'lgbm': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1
            ),
            'xgb2': xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0.05,
                reg_alpha=0.05,
                reg_lambda=0.5,
                random_state=123,
                eval_metric='mlogloss',
                verbosity=0
            )
        }
        
        self.weights = {'xgb': 0.4, 'lgbm': 0.35, 'xgb2': 0.25}
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Addestra il modello con cross-validation"""
        if verbose:
            print("\n🤖 ADDESTRAMENTO MODELLO")
            print("="*80)
        
        # Scala features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        fold_scores = {name: [] for name in self.models.keys()}
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            if verbose:
                print(f"\n   📊 Fold {fold_idx + 1}/5:")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for name, model in self.models.items():
                # Addestra
                model.fit(X_train, y_train)
                
                # Valida
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                fold_scores[name].append(score)
                
                if verbose:
                    print(f"      {name}: {score:.2%}")
        
        # Addestra su tutti i dati
        if verbose:
            print(f"\n   🔄 Training finale su tutti i dati...")
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            avg_score = np.mean(fold_scores[name])
            
            if verbose:
                print(f"      {name}: CV media = {avg_score:.2%}")
            
            # Aggiusta peso basato su performance
            self.weights[name] = avg_score
        
        # Normalizza pesi
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.is_trained = True
        
        if verbose:
            print(f"\n   ✅ Addestramento completato!")
            print(f"   📊 Pesi finali: {', '.join([f'{k}={v:.2%}' for k, v in self.weights.items()])}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice probabilità con ensemble"""
        if not self.is_trained:
            raise ValueError("Modello non addestrato!")
        
        X_scaled = self.scaler.transform(X)
        
        # Ensemble weighted
        ensemble_proba = np.zeros((X_scaled.shape[0], 3))
        
        for name, model in self.models.items():
            proba = model.predict_proba(X_scaled)
            ensemble_proba += proba * self.weights[name]
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice classe"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# ============================================================================
# 5. SISTEMA TITAN v5.0 PRINCIPALE
# ============================================================================

class TitanMasterV5:
    """Sistema TITAN Master v5.0 - Semplificato e Ottimizzato"""
    
    def __init__(self, base_path: str = None):
        self.file_manager = FileInputManager(base_path)
        self.elo_system = AdvancedEloSystem(k_factor=32.0, home_advantage=100.0)
        self.feature_engine = None
        self.model = OptimizedPredictionModel()
        self.is_calibrated = False
        
        # Dati
        self.storico_combined = None
        self.classifica_seriea = None
        self.classifica_serieb = None
        
        # Performance tracking
        self.calibration_accuracy = 0.0
        self.prediction_history = []
    
    def initialize(self) -> bool:
        """Inizializza il sistema caricando i file"""
        print("\n" + "="*80)
        print("🏆 TITAN MASTER SYSTEM v5.0")
        print("="*80)
        print("Sistema di pronostici calcio ottimizzato")
        print("Target: Accuratezza massima")
        print("="*80)
        
        # Carica file
        if not self.file_manager.check_and_load_files():
            return False
        
        # Ottieni dati
        self.storico_combined = self.file_manager.get_combined_storico()
        self.classifica_seriea = self.file_manager.get_classifica('seriea')
        self.classifica_serieb = self.file_manager.get_classifica('serieb')
        
        # Inizializza Elo da classifiche
        print(f"\n⚙️  Inizializzazione sistema Elo...")
        self.elo_system.initialize_from_classifica(
            pd.concat([self.classifica_seriea, self.classifica_serieb])
        )
        
        # Inizializza feature engine
        print(f"⚙️  Inizializzazione feature engine...")
        self.feature_engine = OptimizedFeatureEngine(
            self.storico_combined,
            self.elo_system
        )
        
        print(f"\n✅ Sistema inizializzato correttamente!")
        
        return True
    
    def calibrate(self, test_size: int = 50):
        """Calibra il sistema con ottimizzazione rapida"""
        print("\n" + "="*80)
        print("🎯 CALIBRAZIONE SISTEMA")
        print("="*80)
        
        # Prepara dataset
        print(f"\n📊 Preparazione dataset di training...")
        X_train, y_train = self._prepare_training_data()
        
        print(f"   • Campioni totali: {len(X_train)}")
        print(f"   • Features: {X_train.shape[1]}")
        
        # Split train/test
        if len(X_train) > test_size:
            X_test = X_train[-test_size:]
            y_test = y_train[-test_size:]
            X_train = X_train[:-test_size]
            y_train = y_train[:-test_size]
            
            print(f"   • Training set: {len(X_train)}")
            print(f"   • Test set: {len(X_test)}")
        else:
            X_test, y_test = None, None
            print(f"   ⚠️  Dataset piccolo, no test set")
        
        # Addestra modello
        self.model.train(X_train, y_train)
        
        # Test
        if X_test is not None:
            print(f"\n📈 VALUTAZIONE SU TEST SET")
            print("="*80)
            
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_proba)
            
            self.calibration_accuracy = accuracy
            
            print(f"   ✅ ACCURATEZZA: {accuracy:.2%}")
            print(f"   📊 Log Loss: {logloss:.4f}")
            
            # Per classe
            print(f"\n   🎯 Accuratezza per classe:")
            for i, label in enumerate(['Casa (H)', 'Pareggio (D)', 'Trasferta (A)']):
                class_mask = y_test == i
                if class_mask.sum() > 0:
                    class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                    print(f"      {label}: {class_acc:.2%} ({class_mask.sum()} campioni)")
        
        self.is_calibrated = True
        
        print(f"\n✅ CALIBRAZIONE COMPLETATA!")
        
        # Salva modello
        self.save_system()
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dati di training con gestione ottimizzata grandi dataset"""
        X_list = []
        y_list = []
        
        # Con 30+ anni di dati, usiamo strategia intelligente:
        # - Ultimi 3 anni: tutti i match (più rilevanti)
        # - 4-10 anni fa: 50% campionamento
        # - >10 anni fa: 25% campionamento
        
        total_matches = len(self.storico_combined)
        print(f"   📊 Dataset totale: {total_matches:,} partite")
        
        # Ordina per data
        sorted_data = self.storico_combined.sort_values('Date').reset_index(drop=True)
        
        # Calcola data soglia (ultimi 3 anni = tutto, >3 anni = campionamento)
        latest_date = sorted_data['Date'].max()
        three_years_ago = latest_date - timedelta(days=3*365)
        ten_years_ago = latest_date - timedelta(days=10*365)
        
        print(f"   🎯 Strategia campionamento:")
        print(f"      • Ultimi 3 anni ({three_years_ago.strftime('%Y')}→): 100% dati")
        print(f"      • 4-10 anni fa ({ten_years_ago.strftime('%Y')}-{three_years_ago.strftime('%Y')}): 50% campionamento")
        print(f"      • >10 anni fa (<{ten_years_ago.strftime('%Y')}): 25% campionamento")
        
        samples_used = 0
        samples_skipped = 0
        
        for idx, match in sorted_data.iterrows():
            match_date = match['Date']
            
            # Determina se usare questo match
            use_match = False
            
            if match_date >= three_years_ago:
                # Ultimi 3 anni: usa sempre
                use_match = True
            elif match_date >= ten_years_ago:
                # 4-10 anni: 50% probabilità
                use_match = (np.random.random() < 0.5)
            else:
                # >10 anni: 25% probabilità
                use_match = (np.random.random() < 0.25)
            
            if not use_match:
                samples_skipped += 1
                # Aggiorna comunque Elo per mantenere continuità storica
                self.elo_system.update_rating(
                    match['HomeTeam'],
                    match['AwayTeam'],
                    match['FTR'],
                    match['FTHG'],
                    match['FTAG']
                )
                continue
            
            # Estrai features
            features = self.feature_engine.extract_features(
                match['HomeTeam'],
                match['AwayTeam'],
                match['Date']
            )
            
            # Converti in vettore
            X = self.feature_engine.get_feature_vector(features)
            
            # Target
            target_map = {'H': 0, 'D': 1, 'A': 2}
            y = target_map[match['FTR']]
            
            X_list.append(X)
            y_list.append(y)
            samples_used += 1
            
            # Aggiorna Elo
            self.elo_system.update_rating(
                match['HomeTeam'],
                match['AwayTeam'],
                match['FTR'],
                match['FTHG'],
                match['FTAG']
            )
            
            # Progress indicator ogni 1000 partite
            if samples_used % 1000 == 0:
                progress = (idx + 1) / len(sorted_data) * 100
                print(f"      Processate: {idx+1:,}/{total_matches:,} ({progress:.1f}%) - "
                      f"Campioni: {samples_used:,}", end='\r')
        
        print()  # Newline dopo progress
        print(f"   ✅ Dataset preparato:")
        print(f"      • Campioni usati: {samples_used:,}")
        print(f"      • Campioni saltati: {samples_skipped:,}")
        print(f"      • Riduzione: {samples_skipped/total_matches:.1%}")
        
        return np.array(X_list), np.array(y_list)
    
    def predict_match(self, home_team: str, away_team: str, 
                     league: str = 'SerieA') -> Dict:
        """Predice una partita"""
        if not self.is_calibrated:
            raise ValueError("Sistema non calibrato! Esegui prima calibrate()")
        
        # Normalizza nomi squadre
        home_team = home_team.strip().upper()
        away_team = away_team.strip().upper()
        
        # Estrai features
        features = self.feature_engine.extract_features(home_team, away_team)
        X = self.feature_engine.get_feature_vector(features).reshape(1, -1)
        
        # Predici
        proba = self.model.predict_proba(X)[0]
        prediction_idx = np.argmax(proba)
        
        pred_map = {0: '1', 1: 'X', 2: '2'}
        pred_names = {0: 'CASA', 1: 'PAREGGIO', 2: 'TRASFERTA'}
        
        prediction = pred_map[prediction_idx]
        prediction_name = pred_names[prediction_idx]
        
        # Calcola confidence
        confidence = float(np.max(proba))
        
        # Goal predictions (Poisson)
        expected_home = features['expected_home_goals']
        expected_away = features['expected_away_goals']
        
        # Risultato più probabile
        most_likely_score = self._calculate_most_likely_score(expected_home, expected_away)
        
        # Over/Under
        over_25_prob = self._calculate_over_probability(expected_home, expected_away, 2.5)
        
        # GG/NG
        gg_prob = (1 - poisson.pmf(0, expected_home)) * (1 - poisson.pmf(0, expected_away))
        
        result = {
            'match': f"{home_team} vs {away_team}",
            'league': league,
            'prediction_1x2': prediction,
            'prediction_name': prediction_name,
            'probabilities': {
                '1': float(proba[0]),
                'X': float(proba[1]),
                '2': float(proba[2])
            },
            'confidence': confidence,
            'expected_goals': {
                'home': float(expected_home),
                'away': float(expected_away),
                'total': float(expected_home + expected_away)
            },
            'most_likely_score': most_likely_score,
            'over_under': {
                'over_25': float(over_25_prob),
                'under_25': float(1 - over_25_prob)
            },
            'both_score': {
                'gg': float(gg_prob),
                'ng': float(1 - gg_prob)
            },
            'elo_ratings': {
                'home': float(self.elo_system.get_rating(home_team)),
                'away': float(self.elo_system.get_rating(away_team)),
                'diff': float(self.elo_system.get_rating(home_team) - self.elo_system.get_rating(away_team))
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Salva in history
        self.prediction_history.append(result)
        
        return result
    
    def _calculate_most_likely_score(self, lambda_home: float, lambda_away: float) -> str:
        """Calcola risultato più probabile"""
        max_prob = 0
        best_score = "1-1"
        
        for i in range(6):
            for j in range(6):
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if prob > max_prob:
                    max_prob = prob
                    best_score = f"{i}-{j}"
        
        return best_score
    
    def _calculate_over_probability(self, lambda_home: float, lambda_away: float, 
                                   threshold: float) -> float:
        """Calcola probabilità Over"""
        prob_under = 0
        
        for i in range(10):
            for j in range(10):
                if i + j <= threshold:
                    prob_under += poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        
        return 1 - prob_under
    
    def save_system(self, filename: str = 'titan_v5_system.pkl'):
        """Salva il sistema"""
        print(f"\n💾 Salvataggio sistema...")
        
        system_data = {
            'model': self.model,
            'elo_system': self.elo_system,
            'feature_engine': self.feature_engine,
            'calibration_accuracy': self.calibration_accuracy,
            'is_calibrated': self.is_calibrated,
            'version': 'TITAN_v5.0',
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(system_data, filename)
        print(f"   ✅ Sistema salvato: {filename}")
    
    def load_system(self, filename: str = 'titan_v5_system.pkl') -> bool:
        """Carica il sistema"""
        if not os.path.exists(filename):
            print(f"❌ File non trovato: {filename}")
            return False
        
        print(f"\n📂 Caricamento sistema da {filename}...")
        
        try:
            system_data = joblib.load(filename)
            
            self.model = system_data['model']
            self.elo_system = system_data['elo_system']
            self.feature_engine = system_data['feature_engine']
            self.calibration_accuracy = system_data.get('calibration_accuracy', 0.0)
            self.is_calibrated = system_data.get('is_calibrated', False)
            
            print(f"   ✅ Sistema caricato!")
            print(f"   📊 Accuratezza calibrazione: {self.calibration_accuracy:.2%}")
            print(f"   📅 Data salvataggio: {system_data.get('save_date', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento: {e}")
            return False

# ============================================================================
# 6. INTERFACCIA UTENTE
# ============================================================================

def display_prediction(prediction: Dict):
    """Visualizza pronostico in formato leggibile"""
    print("\n" + "="*80)
    print("🎯 PRONOSTICO COMPLETO")
    print("="*80)
    
    print(f"\n⚽ PARTITA: {prediction['match']}")
    print(f"🏆 CAMPIONATO: {prediction['league']}")
    print(f"⏰ GENERATO: {prediction['timestamp']}")
    
    print(f"\n📊 PREVISIONE RISULTATO:")
    print(f"   🎲 Pronostico: {prediction['prediction_name']} ({prediction['prediction_1x2']})")
    print(f"   🔥 Confidenza: {prediction['confidence']:.1%}")
    
    print(f"\n📈 PROBABILITÀ 1X2:")
    probs = prediction['probabilities']
    print(f"   • 1 (Casa):      {probs['1']:>6.1%}")
    print(f"   • X (Pareggio):  {probs['X']:>6.1%}")
    print(f"   • 2 (Trasferta): {probs['2']:>6.1%}")
    
    print(f"\n⚽ PREVISIONE GOL:")
    goals = prediction['expected_goals']
    print(f"   • Attesi Casa: {goals['home']:.2f}")
    print(f"   • Attesi Trasferta: {goals['away']:.2f}")
    print(f"   • Totale atteso: {goals['total']:.2f}")
    print(f"   • Risultato più probabile: {prediction['most_likely_score']}")
    
    print(f"\n📊 OVER/UNDER 2.5:")
    ou = prediction['over_under']
    print(f"   • Over 2.5:  {ou['over_25']:>6.1%}")
    print(f"   • Under 2.5: {ou['under_25']:>6.1%}")
    
    print(f"\n🤝 GOL-GOL (GG/NG):")
    bs = prediction['both_score']
    print(f"   • GG (Entrambe segnano): {bs['gg']:>6.1%}")
    print(f"   • NG (No Gol):          {bs['ng']:>6.1%}")
    
    print(f"\n📊 RATING ELO:")
    elo = prediction['elo_ratings']
    print(f"   • Casa: {elo['home']:.0f}")
    print(f"   • Trasferta: {elo['away']:.0f}")
    print(f"   • Differenza: {elo['diff']:+.0f}")
    
    print("\n" + "="*80)

def main():
    """Funzione principale"""
    # Crea sistema
    titan = TitanMasterV5()
    
    # Inizializza
    if not titan.initialize():
        print("\n❌ Inizializzazione fallita. Controlla i file richiesti.")
        return
    
    # Menu principale
    while True:
        print("\n" + "="*80)
        print("📋 MENU PRINCIPALE")
        print("="*80)
        print("1. 🎯 Calibra sistema")
        print("2. ⚽ Pronostico singola partita")
        print("3. 📊 Pronostici multipli")
        print("4. 💾 Salva sistema")
        print("5. 📂 Carica sistema salvato")
        print("6. 🚪 Esci")
        
        choice = input("\n👉 Scegli opzione (1-6): ").strip()
        
        if choice == '1':
            # Calibrazione
            titan.calibrate()
            
        elif choice == '2':
            # Pronostico singolo
            if not titan.is_calibrated:
                print("\n⚠️  Sistema non calibrato! Calibra prima (opzione 1)")
                continue
            
            print("\n⚽ PRONOSTICO SINGOLA PARTITA")
            print("-"*40)
            
            home = input("🏠 Squadra casa: ").strip()
            away = input("✈️  Squadra trasferta: ").strip()
            league = input("🏆 Campionato (SerieA/SerieB, default SerieA): ").strip()
            league = league if league in ['SerieA', 'SerieB'] else 'SerieA'
            
            try:
                prediction = titan.predict_match(home, away, league)
                display_prediction(prediction)
                
                # Salva?
                save = input("\n💾 Salvare questo pronostico? (s/n): ").strip().lower()
                if save == 's':
                    filename = f"pronostico_{home}_{away}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(prediction, f, indent=2, ensure_ascii=False)
                    print(f"✅ Salvato in {filename}")
                    
            except Exception as e:
                print(f"❌ Errore: {e}")
            
        elif choice == '3':
            # Pronostici multipli
            if not titan.is_calibrated:
                print("\n⚠️  Sistema non calibrato! Calibra prima (opzione 1)")
                continue
            
            print("\n📊 PRONOSTICI MULTIPLI")
            print("-"*40)
            print("Inserisci le partite (formato: Casa,Trasferta,Campionato)")
            print("Scrivi 'FINE' quando hai terminato")
            
            matches = []
            while True:
                line = input(f"\n   Partita {len(matches)+1}: ").strip()
                if line.upper() == 'FINE':
                    break
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    match = {
                        'home': parts[0],
                        'away': parts[1],
                        'league': parts[2] if len(parts) > 2 else 'SerieA'
                    }
                    matches.append(match)
            
            if matches:
                print(f"\n🎯 Calcolo {len(matches)} pronostici...")
                predictions = []
                
                for i, match in enumerate(matches, 1):
                    try:
                        pred = titan.predict_match(match['home'], match['away'], match['league'])
                        predictions.append(pred)
                        print(f"   [{i}/{len(matches)}] ✅ {match['home']} vs {match['away']}")
                    except Exception as e:
                        print(f"   [{i}/{len(matches)}] ❌ Errore: {e}")
                
                # Mostra riepilogo
                print(f"\n📊 RIEPILOGO PRONOSTICI")
                print("="*80)
                print(f"{'PARTITA':<40} {'PROS':<6} {'CONF':<8} {'SCORE':<10}")
                print("-"*80)
                
                for pred in predictions:
                    match_str = pred['match'][:38]
                    pros = pred['prediction_1x2']
                    conf = f"{pred['confidence']:.1%}"
                    score = pred['most_likely_score']
                    print(f"{match_str:<40} {pros:<6} {conf:<8} {score:<10}")
                
                # Salva?
                save = input("\n💾 Salvare tutti i pronostici? (s/n): ").strip().lower()
                if save == 's':
                    filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(predictions, f, indent=2, ensure_ascii=False)
                    print(f"✅ Salvati in {filename}")
        
        elif choice == '4':
            # Salva sistema
            titan.save_system()
            
        elif choice == '5':
            # Carica sistema
            filename = input("📁 Nome file (default: titan_v5_system.pkl): ").strip()
            filename = filename if filename else 'titan_v5_system.pkl'
            titan.load_system(filename)
            
        elif choice == '6':
            # Esci
            print("\n👋 Arrivederci!")
            if titan.is_calibrated:
                save = input("💾 Salvare il sistema prima di uscire? (s/n): ").strip().lower()
                if save == 's':
                    titan.save_system()
            break
        
        else:
            print("\n❌ Scelta non valida")

if __name__ == "__main__":
    main()
