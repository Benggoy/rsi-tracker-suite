#!/usr/bin/env python3
"""
Professional RSI Stock Tracker with Charts and Black-Scholes Options Pricing
A comprehensive desktop trading application with technical analysis and options pricing.

Author: Benggoy
Repository: https://github.com/Benggoy/trading-analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta, date
import json
import os
import webbrowser
import csv
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Chart and visualization imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# Scientific computing
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import math

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

class BlackScholesCalculator:
    """Black-Scholes options pricing model with Greeks."""
    
    @staticmethod
    def black_scholes(S, K, T, r, sigma, option_type='call'):
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Option price
        """
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma):
        """
        Calculate all Greeks.
        
        Returns:
            Dictionary with Delta, Gamma, Theta, Vega, Rho for calls and puts
        """
        if T <= 0:
            return {
                'call_delta': 1.0 if S > K else 0.0,
                'put_delta': -1.0 if S < K else 0.0,
                'gamma': 0.0, 'theta_call': 0.0, 'theta_put': 0.0,
                'vega': 0.0, 'rho_call': 0.0, 'rho_put': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                      - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        theta_put = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'theta_call': theta_call,
            'theta_put': theta_put,
            'vega': vega,
            'rho_call': rho_call,
            'rho_put': rho_put
        }
    
    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using bisection method."""
        if T <= 0:
            return 0.0
        
        def objective(sigma):
            return abs(BlackScholesCalculator.black_scholes(S, K, T, r, sigma, option_type) - market_price)
        
        try:
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            return result.x
        except:
            return 0.30  # Default 30% volatility

class RSICalculator:
    """Calculate RSI (Relative Strength Index) for stock data."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI for given price series.
        
        Args:
            prices: Series of closing prices
            period: RSI calculation period (default 14)
            
        Returns:
            RSI series
        """
        if len(prices) < period + 1:
            return pd.Series([50.0] * len(prices), index=prices.index)
        
        # Calculate price changes
        deltas = prices.diff()
        
        # Separate gains and losses
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0)

class StockData:
    """Handle stock data fetching and processing."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # Cache data for 60 seconds
    
    def get_stock_data(self, symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', etc.)
            
        Returns:
            DataFrame with stock data or None if error
        """
        cache_key = f"{symbol}_{period}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return data
        
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            # Cache the data
            self.cache[cache_key] = (data, current_time)
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_comprehensive_stock_info(self, symbol: str) -> Dict:
        """Get comprehensive stock information including options data."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist_data = self.get_stock_data(symbol, "1mo")
            
            # Current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price and hist_data is not None and not hist_data.empty:
                current_price = float(hist_data['Close'].iloc[-1])
            
            # Volatility calculation
            volatility = self.calculate_volatility(hist_data) if hist_data is not None else 0.30
            
            # Market cap formatting
            market_cap = info.get('marketCap', 0)
            market_cap_formatted = self.format_market_cap(market_cap)
            
            # Volume data
            daily_volume = info.get('volume', 0) or info.get('regularMarketVolume', 0)
            avg_volume = info.get('averageVolume', 0) or info.get('averageVolume10days', 0)
            
            # Bid/Ask data
            bid_price = info.get('bid', 0)
            ask_price = info.get('ask', 0)
            
            # Options-specific data
            dividend_yield = info.get('dividendYield', 0) or 0
            beta = info.get('beta', 1.0) or 1.0
            
            # Calculate average volume from historical data if not available
            if avg_volume == 0 and hist_data is not None and not hist_data.empty:
                avg_volume = int(hist_data['Volume'].mean())
            
            return {
                'current_price': current_price,
                'market_cap': market_cap,
                'market_cap_formatted': market_cap_formatted,
                'daily_volume': daily_volume,
                'avg_volume': avg_volume,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'company_name': info.get('longName', symbol),
                'volatility': volatility,
                'dividend_yield': dividend_yield,
                'beta': beta
            }
            
        except Exception as e:
            print(f"Error fetching comprehensive info for {symbol}: {e}")
            return {
                'current_price': None,
                'market_cap': 0,
                'market_cap_formatted': 'N/A',
                'daily_volume': 0,
                'avg_volume': 0,
                'bid_price': 0,
                'ask_price': 0,
                'company_name': symbol,
                'volatility': 0.30,
                'dividend_yield': 0,
                'beta': 1.0
            }
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 30) -> float:
        """Calculate historical volatility."""
        if data is None or len(data) < 2:
            return 0.30
        
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        return float(volatility) if not np.isnan(volatility) else 0.30
    
    def format_market_cap(self, market_cap: int) -> str:
        """Format market cap in readable format (B, M, K)."""
        if market_cap == 0:
            return "N/A"
        elif market_cap >= 1_000_000_000_000:  # Trillions
            return f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:  # Billions
            return f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:  # Millions
            return f"${market_cap / 1_000_000:.1f}M"
        elif market_cap >= 1_000:  # Thousands
            return f"${market_cap / 1_000:.1f}K"
        else:
            return f"${market_cap:,.0f}"
    
    def format_volume(self, volume: int) -> str:
        """Format volume in readable format."""
        if volume == 0:
            return "N/A"
        elif volume >= 1_000_000_000:  # Billions
            return f"{volume / 1_000_000_000:.2f}B"
        elif volume >= 1_000_000:  # Millions
            return f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:  # Thousands
            return f"{volume / 1_000:.1f}K"
        else:
            return f"{volume:,}"

class ChartManager:
    """Handle chart creation and management."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.figures = {}
        self.canvases = {}
        
    def create_price_rsi_chart(self, symbol: str, data: pd.DataFrame) -> None:
        """Create price and RSI chart."""
        if symbol in self.figures:
            plt.close(self.figures[symbol])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      facecolor='#1e1e1e')
        
        # Price chart
        ax1.plot(data.index, data['Close'], color='#00ff88', linewidth=2, label='Price')
        ax1.fill_between(data.index, data['Low'], data['High'], alpha=0.3, color='#444444')
        ax1.set_title(f'{symbol} - Price & RSI Analysis', color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', color='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # RSI chart
        rsi_calc = RSICalculator()
        rsi = rsi_calc.calculate_rsi(data['Close'])
        
        ax2.plot(data.index, rsi, color='#ff8800', linewidth=2, label='RSI')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax2.axhline(y=50, color='white', linestyle='-', alpha=0.3)
        ax2.fill_between(data.index, 70, 100, alpha=0.2, color='red')
        ax2.fill_between(data.index, 0, 30, alpha=0.2, color='green')
        ax2.set_ylabel('RSI', color='white')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
        
        # Set colors
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        plt.tight_layout()
        self.figures[f'{symbol}_price_rsi'] = fig
        
        # Create canvas
        if hasattr(self, 'chart_frame'):
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvases[f'{symbol}_price_rsi'] = canvas
    
    def create_volume_chart(self, symbol: str, data: pd.DataFrame) -> None:
        """Create volume analysis chart."""
        if f'{symbol}_volume' in self.figures:
            plt.close(self.figures[f'{symbol}_volume'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                      gridspec_kw={'height_ratios': [2, 1]},
                                      facecolor='#1e1e1e')
        
        # Price with volume colors
        colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' 
                 for i in range(len(data))]
        ax1.plot(data.index, data['Close'], color='#00ff88', linewidth=2)
        ax1.set_title(f'{symbol} - Price & Volume Analysis', color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', color='white')
        ax1.grid(True, alpha=0.3)
        
        # Volume bars
        ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Volume', color='white')
        ax2.grid(True, alpha=0.3)
        
        # Average volume line
        avg_volume = data['Volume'].mean()
        ax2.axhline(y=avg_volume, color='yellow', linestyle='--', alpha=0.8, label=f'Avg Volume: {avg_volume:,.0f}')
        ax2.legend()
        
        # Format dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
        
        # Set colors
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        plt.tight_layout()
        self.figures[f'{symbol}_volume'] = fig
        
        # Create and display canvas
        if hasattr(self, 'chart_frame'):
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvases[f'{symbol}_volume'] = canvas
    
    def create_candlestick_chart(self, symbol: str, data: pd.DataFrame) -> None:
        """Create candlestick chart."""
        if f'{symbol}_candlestick' in self.figures:
            plt.close(self.figures[f'{symbol}_candlestick'])
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1e1e1e')
        
        # Create candlesticks
        for i, (idx, row) in enumerate(data.iterrows()):
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            
            # Body
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Open'], row['Close'])
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, alpha=0.8)
            ax.add_patch(rect)
            
            # Wicks
            ax.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1)
        
        ax.set_title(f'{symbol} - Candlestick Chart', color='white', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)', color='white')
        ax.set_xlim(-0.5, len(data) - 0.5)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        step = max(1, len(data) // 10)
        ax.set_xticks(range(0, len(data), step))
        ax.set_xticklabels([data.index[i].strftime('%m/%d') for i in range(0, len(data), step)])
        
        ax.set_facecolor('#2d2d2d')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        plt.tight_layout()
        self.figures[f'{symbol}_candlestick'] = fig
        
        # Create and display canvas
        if hasattr(self, 'chart_frame'):
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvases[f'{symbol}_candlestick'] = canvas

class OptionsCalculator:
    """Options pricing and Greeks calculator interface."""
    
    def __init__(self, parent):
        self.parent = parent
        self.bs_calc = BlackScholesCalculator()
        
    def create_options_interface(self, notebook):
        """Create options calculation interface."""
        options_frame = ttk.Frame(notebook)
        notebook.add(options_frame, text="📊 Options Pricing")
        
        # Left panel - Inputs
        input_frame = tk.Frame(options_frame, bg='#1e1e1e')
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Title
        tk.Label(input_frame, text="Black-Scholes Options Calculator", 
                font=('Arial', 14, 'bold'), bg='#1e1e1e', fg='#00ff88').pack(pady=10)
        
        # Input fields
        self.create_input_fields(input_frame)
        
        # Calculate button
        tk.Button(input_frame, text="🔢 Calculate Options", 
                 command=self.calculate_options,
                 bg='#00ff88', fg='black', font=('Arial', 12, 'bold')).pack(pady=20)
        
        # Results panel
        results_frame = tk.Frame(options_frame, bg='#1e1e1e')
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results display
        self.create_results_display(results_frame)
        
    def create_input_fields(self, parent):
        """Create input fields for options parameters."""
        self.inputs = {}
        
        fields = [
            ("Stock Price ($)", "stock_price", "100.00"),
            ("Strike Price ($)", "strike_price", "100.00"),
            ("Days to Expiration", "days_to_exp", "30"),
            ("Risk-free Rate (%)", "risk_free_rate", "5.0"),
            ("Volatility (%)", "volatility", "30.0"),
            ("Dividend Yield (%)", "dividend_yield", "0.0")
        ]
        
        for label, key, default in fields:
            frame = tk.Frame(parent, bg='#1e1e1e')
            frame.pack(fill=tk.X, pady=5)
            
            tk.Label(frame, text=label, bg='#1e1e1e', fg='white', 
                    font=('Arial', 10)).pack(anchor=tk.W)
            
            entry = tk.Entry(frame, font=('Arial', 11), width=15)
            entry.insert(0, default)
            entry.pack(anchor=tk.W, pady=2)
            
            self.inputs[key] = entry
        
    def create_results_display(self, parent):
        """Create results display area."""
        # Results text area
        results_label = tk.Label(parent, text="Options Pricing Results", 
                               font=('Arial', 14, 'bold'), bg='#1e1e1e', fg='#00ff88')
        results_label.pack(pady=10)
        
        self.results_text = tk.Text(parent, height=20, width=60, 
                                  bg='#2d2d2d', fg='white', 
                                  font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(parent, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def calculate_options(self):
        """Calculate and display options prices and Greeks."""
        try:
            # Get input values
            S = float(self.inputs['stock_price'].get())
            K = float(self.inputs['strike_price'].get())
            days = int(self.inputs['days_to_exp'].get())
            r = float(self.inputs['risk_free_rate'].get()) / 100
            sigma = float(self.inputs['volatility'].get()) / 100
            div_yield = float(self.inputs['dividend_yield'].get()) / 100
            
            T = days / 365.0  # Convert days to years
            
            # Calculate option prices
            call_price = self.bs_calc.black_scholes(S, K, T, r, sigma, 'call')
            put_price = self.bs_calc.black_scholes(S, K, T, r, sigma, 'put')
            
            # Calculate Greeks
            greeks = self.bs_calc.calculate_greeks(S, K, T, r, sigma)
            
            # Format results
            results = f"""
{'='*60}
BLACK-SCHOLES OPTIONS PRICING RESULTS
{'='*60}

INPUT PARAMETERS:
  Stock Price:         ${S:,.2f}
  Strike Price:        ${K:,.2f}
  Days to Expiration:  {days} days
  Time to Expiration:  {T:.4f} years
  Risk-free Rate:      {r*100:.2f}%
  Volatility:          {sigma*100:.2f}%
  Dividend Yield:      {div_yield*100:.2f}%

THEORETICAL OPTION PRICES:
  Call Option:         ${call_price:.2f}
  Put Option:          ${put_price:.2f}
  
CALL/PUT PARITY CHECK:
  Call - Put:          ${call_price - put_price:.2f}
  S - K*e^(-rT):       ${S - K*np.exp(-r*T):.2f}
  Difference:          ${abs((call_price - put_price) - (S - K*np.exp(-r*T))):.4f}

THE GREEKS:
  
CALL OPTION GREEKS:
  Delta:               {greeks['call_delta']:.4f}
  Gamma:               {greeks['gamma']:.4f}
  Theta:               ${greeks['theta_call']:.2f} per day
  Vega:                ${greeks['vega']:.2f} per 1% vol change
  Rho:                 ${greeks['rho_call']:.2f} per 1% rate change

PUT OPTION GREEKS:
  Delta:               {greeks['put_delta']:.4f}
  Gamma:               {greeks['gamma']:.4f}
  Theta:               ${greeks['theta_put']:.2f} per day
  Vega:                ${greeks['vega']:.2f} per 1% vol change
  Rho:                 ${greeks['rho_put']:.2f} per 1% rate change

RISK ANALYSIS:
  Intrinsic Value (Call): ${max(S - K, 0):.2f}
  Time Value (Call):      ${call_price - max(S - K, 0):.2f}
  Intrinsic Value (Put):  ${max(K - S, 0):.2f}
  Time Value (Put):       ${put_price - max(K - S, 0):.2f}
  
  Moneyness:           {'ITM' if S > K else 'OTM' if S < K else 'ATM'}
  Probability ITM (Call): {norm.cdf((np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T)))*100:.1f}%
  Probability ITM (Put):  {(1-norm.cdf((np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))))*100:.1f}%

TRADING RECOMMENDATIONS:
"""
            
            # Add trading recommendations
            if greeks['call_delta'] > 0.7:
                results += "  • High Delta Call - Consider selling covered calls\n"
            elif greeks['call_delta'] < 0.3:
                results += "  • Low Delta Call - Cheaper speculation, higher risk\n"
            
            if abs(greeks['theta_call']) > 0.05:
                results += f"  • High Time Decay - ${abs(greeks['theta_call']):.2f} per day\n"
            
            if greeks['gamma'] > 0.02:
                results += "  • High Gamma - Delta will change rapidly\n"
            
            results += f"\n{'='*60}"
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check your input values: {e}")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Error in calculation: {e}")

class ProfessionalRSITracker:
    """Main application class for Professional RSI Stock Tracker."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Professional RSI Stock Tracker - Charts & Options")
        self.root.geometry("1600x900")  # Larger window for charts
        self.root.configure(bg='#1e1e1e')
        
        # Data handlers
        self.stock_data = StockData()
        self.rsi_calculator = RSICalculator()
        self.bs_calculator = BlackScholesCalculator()
        
        # Watchlist
        self.watchlist = []
        self.watchlist_file = "watchlist.json"
        self.load_watchlist()
        
        # Update control
        self.update_interval = 30  # seconds
        self.is_updating = False
        self.update_thread = None
        
        # Current selected stock
        self.current_symbol = None
        
        # Setup UI
        self.setup_ui()
        self.setup_styles()
        
        # Start updates
        self.start_updates()
        
        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Setup custom styles for the application."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors for dark theme
        style.configure('Treeview', 
                       background='#2d2d2d', 
                       foreground='white',
                       fieldbackground='#2d2d2d',
                       rowheight=25)
        style.configure('Treeview.Heading',
                       background='#404040',
                       foreground='white',
                       font=('Arial', 9, 'bold'))
    
    def setup_ui(self):
        """Setup the user interface with tabs."""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_main_tab()
        self.create_charts_tab()
        self.create_options_tab()
        
    def create_main_tab(self):
        """Create main stock tracking tab."""
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="📈 Stock Tracker")
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="📈 Professional RSI Stock Tracker", 
                              font=('Arial', 16, 'bold'),
                              bg='#1e1e1e', fg='#00ff88')
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(main_frame, bg='#1e1e1e')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add stock section
        add_frame = tk.Frame(control_frame, bg='#1e1e1e')
        add_frame.pack(side=tk.LEFT)
        
        tk.Label(add_frame, text="Add Stock:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.symbol_entry = tk.Entry(add_frame, width=10, font=('Arial', 10))
        self.symbol_entry.pack(side=tk.LEFT, padx=(5, 5))
        self.symbol_entry.bind('<Return>', self.add_stock_event)
        
        tk.Button(add_frame, text="Add", command=self.add_stock,
                 bg='#00ff88', fg='black', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#1e1e1e')
        button_frame.pack(side=tk.RIGHT)
        
        tk.Button(button_frame, text="📊 View Charts", command=self.view_charts,
                 bg='#8844ff', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Remove Selected", command=self.remove_stock,
                 bg='#ff4444', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Seeking Alpha", command=self.open_seeking_alpha,
                 bg='#ff8800', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Refresh Now", command=self.manual_refresh,
                 bg='#4488ff', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = tk.Frame(main_frame, bg='#1e1e1e')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Ready", 
                                    bg='#1e1e1e', fg='#cccccc', font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT)
        
        self.last_update_label = tk.Label(status_frame, text="", 
                                         bg='#1e1e1e', fg='#888888', font=('Arial', 9))
        self.last_update_label.pack(side=tk.RIGHT)
        
        # Stock data table
        self.setup_table(main_frame)
    
    def create_charts_tab(self):
        """Create charts analysis tab."""
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="📊 Charts")
        
        # Chart controls
        control_frame = tk.Frame(charts_frame, bg='#1e1e1e')
        control_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(control_frame, text="Select Stock for Analysis:", 
                bg='#1e1e1e', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        
        self.chart_symbol_var = tk.StringVar()
        chart_combo = ttk.Combobox(control_frame, textvariable=self.chart_symbol_var, 
                                  values=self.watchlist, font=('Arial', 11))
        chart_combo.pack(side=tk.LEFT, padx=10)
        
        # Chart buttons
        tk.Button(control_frame, text="📈 Price & RSI", command=self.show_price_rsi_chart,
                 bg='#00ff88', fg='black', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="📊 Volume", command=self.show_volume_chart,
                 bg='#4488ff', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="🕯️ Candlestick", command=self.show_candlestick_chart,
                 bg='#ff8800', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Chart display area
        self.chart_frame = tk.Frame(charts_frame, bg='#1e1e1e')
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize chart manager
        self.chart_manager = ChartManager(self.chart_frame)
        self.chart_manager.chart_frame = self.chart_frame
    
    def create_options_tab(self):
        """Create options pricing tab."""
        self.options_calc = OptionsCalculator(self.root)
        self.options_calc.create_options_interface(self.notebook)
    
    def setup_table(self, parent):
        """Setup the stock data table."""
        # Table frame
        table_frame = tk.Frame(parent, bg='#1e1e1e')
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for stock data
        columns = ('Symbol', 'Price', 'Change', 'Change%', 'RSI', 'Status', 
                  'MarketCap', 'Volume', 'Volatility', 'Updated')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        headings = {
            'Symbol': ('Stock', 70),
            'Price': ('Price ($)', 80),
            'Change': ('Change ($)', 80), 
            'Change%': ('Change %', 80),
            'RSI': ('RSI', 60),
            'Status': ('RSI Status', 100),
            'MarketCap': ('Market Cap', 100),
            'Volume': ('Volume', 80),
            'Volatility': ('Volatility', 80),
            'Updated': ('Updated', 100)
        }
        
        for col, (heading, width) in headings.items():
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor=tk.CENTER)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack table and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.on_stock_select)
        self.tree.bind('<Double-1>', self.on_double_click)
    
    def on_stock_select(self, event):
        """Handle stock selection."""
        selected = self.tree.selection()
        if selected:
            self.current_symbol = selected[0]
            # Update chart combo box
            if hasattr(self, 'chart_symbol_var'):
                self.chart_symbol_var.set(self.current_symbol)
    
    def view_charts(self):
        """Switch to charts tab with selected stock."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock to view charts")
            return
        
        self.current_symbol = selected[0]
        self.chart_symbol_var.set(self.current_symbol)
        self.notebook.select(1)  # Switch to charts tab
        
        # Auto-load price & RSI chart
        self.show_price_rsi_chart()
    
    def show_price_rsi_chart(self):
        """Show price and RSI chart for selected stock."""
        symbol = self.chart_symbol_var.get()
        if not symbol:
            messagebox.showwarning("No Stock Selected", "Please select a stock symbol")
            return
        
        data = self.stock_data.get_stock_data(symbol, "3mo")
        if data is None or data.empty:
            messagebox.showerror("Data Error", f"Could not fetch data for {symbol}")
            return
        
        # Clear previous charts
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        self.chart_manager.create_price_rsi_chart(symbol, data)
    
    def show_volume_chart(self):
        """Show volume chart for selected stock."""
        symbol = self.chart_symbol_var.get()
        if not symbol:
            messagebox.showwarning("No Stock Selected", "Please select a stock symbol")
            return
        
        data = self.stock_data.get_stock_data(symbol, "3mo")
        if data is None or data.empty:
            messagebox.showerror("Data Error", f"Could not fetch data for {symbol}")
            return
        
        # Clear previous charts
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        self.chart_manager.create_volume_chart(symbol, data)
    
    def show_candlestick_chart(self):
        """Show candlestick chart for selected stock."""
        symbol = self.chart_symbol_var.get()
        if not symbol:
            messagebox.showwarning("No Stock Selected", "Please select a stock symbol")
            return
        
        data = self.stock_data.get_stock_data(symbol, "1mo")
        if data is None or data.empty:
            messagebox.showerror("Data Error", f"Could not fetch data for {symbol}")
            return
        
        # Clear previous charts
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        self.chart_manager.create_candlestick_chart(symbol, data)
    
    def add_stock_event(self, event):
        """Handle Enter key press in symbol entry."""
        self.add_stock()
    
    def add_stock(self):
        """Add a stock to the watchlist."""
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            return
        
        if symbol in self.watchlist:
            messagebox.showwarning("Duplicate", f"{symbol} is already in your watchlist!")
            return
        
        # Validate symbol
        self.status_label.config(text=f"Validating {symbol}...")
        self.root.update()
        
        data = self.stock_data.get_stock_data(symbol, "5d")
        if data is None or data.empty:
            messagebox.showerror("Invalid Symbol", f"Could not find data for {symbol}")
            self.status_label.config(text="Ready")
            return
        
        # Add to watchlist
        self.watchlist.append(symbol)
        self.symbol_entry.delete(0, tk.END)
        self.save_watchlist()
        
        # Update chart combo
        if hasattr(self, 'chart_symbol_var'):
            chart_combo = None
            for child in self.notebook.nametowidget(self.notebook.tabs()[1]).winfo_children():
                if isinstance(child, tk.Frame):
                    for widget in child.winfo_children():
                        if isinstance(widget, ttk.Combobox):
                            chart_combo = widget
                            break
            if chart_combo:
                chart_combo['values'] = self.watchlist
        
        # Add to table
        loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "")
        self.tree.insert('', tk.END, iid=symbol, values=loading_values)
        
        # Update display
        threading.Thread(target=self.update_stock_data, args=(symbol,), daemon=True).start()
        self.status_label.config(text=f"Added {symbol} to watchlist")
    
    def remove_stock(self):
        """Remove selected stock from watchlist."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock to remove")
            return
        
        symbol = selected[0]
        if messagebox.askyesno("Confirm Remove", f"Remove {symbol} from watchlist?"):
            self.watchlist.remove(symbol)
            self.tree.delete(symbol)
            self.save_watchlist()
            
            # Update chart combo
            if hasattr(self, 'chart_symbol_var'):
                chart_combo = None
                for child in self.notebook.nametowidget(self.notebook.tabs()[1]).winfo_children():
                    if isinstance(child, tk.Frame):
                        for widget in child.winfo_children():
                            if isinstance(widget, ttk.Combobox):
                                chart_combo = widget
                                break
                if chart_combo:
                    chart_combo['values'] = self.watchlist
            
            self.status_label.config(text=f"Removed {symbol} from watchlist")
    
    def open_seeking_alpha(self):
        """Open Seeking Alpha for selected stock."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock to view on Seeking Alpha")
            return
        
        symbol = selected[0]
        url = f"https://seekingalpha.com/symbol/{symbol}"
        try:
            webbrowser.open(url)
            self.status_label.config(text=f"Opened Seeking Alpha for {symbol}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Seeking Alpha: {e}")
    
    def on_double_click(self, event):
        """Handle double-click on table row."""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            url = f"https://seekingalpha.com/symbol/{item}"
            try:
                webbrowser.open(url)
            except:
                pass
    
    def manual_refresh(self):
        """Manually refresh all stock data."""
        if not self.watchlist:
            messagebox.showinfo("Empty Watchlist", "Add some stocks to your watchlist first!")
            return
        
        self.status_label.config(text="Refreshing all stocks...")
        threading.Thread(target=self.refresh_all_stocks, daemon=True).start()
    
    def refresh_all_stocks(self):
        """Refresh data for all stocks in watchlist."""
        for symbol in self.watchlist:
            self.update_stock_data(symbol)
            time.sleep(0.5)  # Rate limiting
        
        self.root.after(0, lambda: self.status_label.config(text="All stocks updated"))
    
    def update_stock_data(self, symbol: str):
        """Update comprehensive data for a specific stock."""
        hist_data = self.stock_data.get_stock_data(symbol, "1mo")
        stock_info = self.stock_data.get_comprehensive_stock_info(symbol)
        
        if hist_data is None or hist_data.empty:
            error_values = (symbol, "Error", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error")
            self.root.after(0, lambda: self.update_table_row(symbol, *error_values))
            return
        
        try:
            # Calculate values
            current_price = stock_info['current_price']
            if current_price is None:
                current_price = float(hist_data['Close'].iloc[-1])
            
            previous_price = float(hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else current_price
            price_change = current_price - previous_price
            percent_change = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            # Calculate RSI
            rsi_series = self.rsi_calculator.calculate_rsi(hist_data['Close'])
            rsi = float(rsi_series.iloc[-1])
            
            # Determine RSI status
            if rsi > 70:
                rsi_status = "🔴 Overbought"
            elif rsi < 30:
                rsi_status = "🟢 Oversold"
            else:
                rsi_status = "🟡 Neutral"
            
            # Format values
            price_str = f"${current_price:.2f}"
            change_str = f"${price_change:+.2f}"
            percent_str = f"{percent_change:+.2f}%"
            rsi_str = f"{rsi:.1f}"
            market_cap_str = stock_info['market_cap_formatted']
            volume_str = self.stock_data.format_volume(stock_info['daily_volume'])
            volatility_str = f"{stock_info['volatility']*100:.1f}%"
            updated_str = datetime.now().strftime("%H:%M:%S")
            
            # Update table
            values = (symbol, price_str, change_str, percent_str, rsi_str, rsi_status,
                     market_cap_str, volume_str, volatility_str, updated_str)
            
            self.root.after(0, lambda: self.update_table_row(symbol, *values))
            
        except Exception as e:
            print(f"Error updating {symbol}: {e}")
            error_values = (symbol, "Error", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error")
            self.root.after(0, lambda: self.update_table_row(symbol, *error_values))
    
    def update_table_row(self, symbol: str, *values):
        """Update a row in the stock table."""
        try:
            self.tree.item(symbol, values=values)
        except tk.TclError:
            pass
    
    def start_updates(self):
        """Start automatic updates."""
        if not self.is_updating:
            self.is_updating = True
            self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
            self.update_thread.start()
    
    def update_loop(self):
        """Main update loop."""
        while self.is_updating:
            if self.watchlist:
                for symbol in self.watchlist.copy():
                    if not self.is_updating:
                        break
                    self.update_stock_data(symbol)
                    time.sleep(2)
                
                current_time = datetime.now().strftime("%H:%M:%S")
                self.root.after(0, lambda: self.last_update_label.config(text=f"Last update: {current_time}"))
            
            time.sleep(self.update_interval)
    
    def load_watchlist(self):
        """Load watchlist from file."""
        try:
            if os.path.exists(self.watchlist_file):
                with open(self.watchlist_file, 'r') as f:
                    self.watchlist = json.load(f)
        except Exception as e:
            print(f"Error loading watchlist: {e}")
            self.watchlist = []
    
    def save_watchlist(self):
        """Save watchlist to file."""
        try:
            with open(self.watchlist_file, 'w') as f:
                json.dump(self.watchlist, f)
        except Exception as e:
            print(f"Error saving watchlist: {e}")
    
    def populate_initial_data(self):
        """Populate table with saved watchlist."""
        for symbol in self.watchlist:
            loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "")
            self.tree.insert('', tk.END, iid=symbol, values=loading_values)
    
    def on_closing(self):
        """Handle application closing."""
        self.is_updating = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        
        # Close all matplotlib figures
        plt.close('all')
        
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.populate_initial_data()
        
        if not self.watchlist:
            self.status_label.config(text="Add stock symbols to start tracking and analysis")
        
        self.root.mainloop()

def main():
    """Main function."""
    try:
        app = ProfessionalRSITracker()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()
