#!/usr/bin/env python3
"""
Enhanced Real-Time RSI Stock Tracker with Charts and CSV Import
A desktop application for tracking RSI and comprehensive financial metrics with advanced charting.

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
from datetime import datetime, timedelta
import json
import os
import webbrowser
import csv
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for better looking charts
plt.style.use('dark_background')
sns.set_palette("husl")

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
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0)

class StockData:
    """Handle stock data fetching and processing."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # Cache data for 60 seconds
    
    def get_stock_data(self, symbol: str, period: str = "5d", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', etc.)
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', etc.)
            
        Returns:
            DataFrame with stock data or None if error
        """
        cache_key = f"{symbol}_{period}_{interval}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return data
        
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Cache the data
            self.cache[cache_key] = (data, current_time)
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_comprehensive_stock_info(self, symbol: str) -> Dict:
        """Get comprehensive stock information including market cap, volume, bid/ask."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist_data = self.get_stock_data(symbol, "1mo")
            
            # Current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price and hist_data is not None and not hist_data.empty:
                current_price = float(hist_data['Close'].iloc[-1])
            
            # Market cap formatting
            market_cap = info.get('marketCap', 0)
            market_cap_formatted = self.format_market_cap(market_cap)
            
            # Volume data
            daily_volume = info.get('volume', 0) or info.get('regularMarketVolume', 0)
            avg_volume = info.get('averageVolume', 0) or info.get('averageVolume10days', 0)
            
            # Bid/Ask data
            bid_price = info.get('bid', 0)
            ask_price = info.get('ask', 0)
            
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
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A')
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
                'sector': 'N/A',
                'industry': 'N/A'
            }
    
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

class ChartWindow:
    """Advanced charting window with multiple chart types."""
    
    def __init__(self, parent, stock_data, rsi_calculator):
        self.parent = parent
        self.stock_data = stock_data
        self.rsi_calculator = rsi_calculator
        self.window = None
        
    def show_charts(self, symbol: str):
        """Show comprehensive charts for a stock symbol."""
        if self.window:
            self.window.destroy()
        
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"📈 Advanced Charts - {symbol}")
        self.window.geometry("1200x800")
        self.window.configure(bg='#1e1e1e')
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get stock data for different periods
        data_3mo = self.stock_data.get_stock_data(symbol, "3mo", "1d")
        data_1mo = self.stock_data.get_stock_data(symbol, "1mo", "1d")
        
        if data_3mo is None or data_3mo.empty:
            messagebox.showerror("Error", f"Could not fetch chart data for {symbol}")
            return
        
        # Create different chart tabs
        self.create_candlestick_tab(notebook, symbol, data_1mo)
        self.create_rsi_tab(notebook, symbol, data_3mo)
        self.create_volume_tab(notebook, symbol, data_3mo)
        self.create_comparison_tab(notebook, symbol, data_3mo)
        
    def create_candlestick_tab(self, notebook, symbol, data):
        """Create candlestick chart tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🕯️ Candlestick")
        
        fig = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        ax = fig.add_subplot(111, facecolor='#2d2d2d')
        
        if data is not None and not data.empty:
            # Create candlestick chart
            dates = data.index
            opens = data['Open']
            highs = data['High']
            lows = data['Low']
            closes = data['Close']
            
            # Plot candlesticks
            for i, date in enumerate(dates):
                color = '#00ff88' if closes.iloc[i] >= opens.iloc[i] else '#ff4444'
                
                # Candlestick body
                body_height = abs(closes.iloc[i] - opens.iloc[i])
                body_bottom = min(opens.iloc[i], closes.iloc[i])
                
                ax.add_patch(Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                                     facecolor=color, edgecolor='white', linewidth=0.5))
                
                # Wicks
                ax.plot([i, i], [lows.iloc[i], highs.iloc[i]], color='white', linewidth=1)
            
            # Formatting
            ax.set_xlim(-0.5, len(dates)-0.5)
            ax.set_xticks(range(0, len(dates), max(1, len(dates)//10)))
            ax.set_xticklabels([dates[i].strftime('%m/%d') for i in range(0, len(dates), max(1, len(dates)//10))], 
                              rotation=45, color='white')
            ax.set_ylabel('Price ($)', color='white')
            ax.set_title(f'{symbol} - Candlestick Chart', color='white', fontsize=16, pad=20)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            
            # Add moving averages
            if len(closes) >= 20:
                ma20 = closes.rolling(20).mean()
                ax.plot(range(len(ma20)), ma20, color='#ffaa00', linewidth=2, label='MA20', alpha=0.8)
            
            if len(closes) >= 50:
                ma50 = closes.rolling(50).mean()
                ax.plot(range(len(ma50)), ma50, color='#00aaff', linewidth=2, label='MA50', alpha=0.8)
            
            ax.legend(loc='upper left')
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_rsi_tab(self, notebook, symbol, data):
        """Create RSI chart tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📊 RSI Analysis")
        
        fig = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        
        # Price subplot
        ax1 = fig.add_subplot(211, facecolor='#2d2d2d')
        # RSI subplot
        ax2 = fig.add_subplot(212, facecolor='#2d2d2d')
        
        if data is not None and not data.empty:
            dates = data.index
            prices = data['Close']
            
            # Calculate RSI
            rsi_values = self.rsi_calculator.calculate_rsi(prices)
            
            # Price chart
            ax1.plot(dates, prices, color='#00ff88', linewidth=2, label=f'{symbol} Price')
            ax1.set_ylabel('Price ($)', color='white')
            ax1.set_title(f'{symbol} - Price and RSI Analysis', color='white', fontsize=16, pad=20)
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # RSI chart
            ax2.plot(dates, rsi_values, color='#ffaa00', linewidth=2, label='RSI (14)')
            
            # RSI levels
            ax2.axhline(y=70, color='#ff4444', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.axhline(y=50, color='white', linestyle='-', alpha=0.3, label='Neutral (50)')
            
            # Fill overbought/oversold areas
            ax2.fill_between(dates, 70, 100, alpha=0.2, color='red')
            ax2.fill_between(dates, 0, 30, alpha=0.2, color='green')
            
            ax2.set_ylabel('RSI', color='white')
            ax2.set_xlabel('Date', color='white')
            ax2.set_ylim(0, 100)
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, color='white')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_volume_tab(self, notebook, symbol, data):
        """Create volume analysis tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📊 Volume Analysis")
        
        fig = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        
        # Price subplot
        ax1 = fig.add_subplot(211, facecolor='#2d2d2d')
        # Volume subplot
        ax2 = fig.add_subplot(212, facecolor='#2d2d2d')
        
        if data is not None and not data.empty:
            dates = data.index
            prices = data['Close']
            volumes = data['Volume']
            
            # Price chart
            ax1.plot(dates, prices, color='#00ff88', linewidth=2, label=f'{symbol} Price')
            ax1.set_ylabel('Price ($)', color='white')
            ax1.set_title(f'{symbol} - Price and Volume Analysis', color='white', fontsize=16, pad=20)
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Volume bars
            colors = ['#00ff88' if prices.iloc[i] >= prices.iloc[i-1] if i > 0 else True else '#ff4444' 
                     for i in range(len(prices))]
            ax2.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
            
            # Average volume line
            avg_volume = volumes.mean()
            ax2.axhline(y=avg_volume, color='#ffaa00', linestyle='--', alpha=0.8, 
                       label=f'Avg Volume ({self.stock_data.format_volume(int(avg_volume))})')
            
            ax2.set_ylabel('Volume', color='white')
            ax2.set_xlabel('Date', color='white')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            # Format y-axis for volume
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.stock_data.format_volume(int(x))))
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, color='white')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_comparison_tab(self, notebook, symbol, data):
        """Create technical indicators comparison tab."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📈 Technical Analysis")
        
        fig = Figure(figsize=(12, 10), facecolor='#1e1e1e')
        
        # Create 3 subplots
        ax1 = fig.add_subplot(311, facecolor='#2d2d2d')  # Price with MA
        ax2 = fig.add_subplot(312, facecolor='#2d2d2d')  # RSI
        ax3 = fig.add_subplot(313, facecolor='#2d2d2d')  # MACD (simplified)
        
        if data is not None and not data.empty:
            dates = data.index
            prices = data['Close']
            
            # Calculate indicators
            rsi_values = self.rsi_calculator.calculate_rsi(prices)
            ma20 = prices.rolling(20).mean()
            ma50 = prices.rolling(50).mean()
            
            # Price with Moving Averages
            ax1.plot(dates, prices, color='#00ff88', linewidth=2, label=f'{symbol}')
            if len(prices) >= 20:
                ax1.plot(dates, ma20, color='#ffaa00', linewidth=1.5, label='MA20', alpha=0.8)
            if len(prices) >= 50:
                ax1.plot(dates, ma50, color='#00aaff', linewidth=1.5, label='MA50', alpha=0.8)
            
            ax1.set_ylabel('Price ($)', color='white')
            ax1.set_title(f'{symbol} - Technical Analysis Dashboard', color='white', fontsize=16, pad=20)
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # RSI
            ax2.plot(dates, rsi_values, color='#ffaa00', linewidth=2, label='RSI (14)')
            ax2.axhline(y=70, color='#ff4444', linestyle='--', alpha=0.7)
            ax2.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.7)
            ax2.fill_between(dates, 70, 100, alpha=0.15, color='red')
            ax2.fill_between(dates, 0, 30, alpha=0.15, color='green')
            ax2.set_ylabel('RSI', color='white')
            ax2.set_ylim(0, 100)
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            # Price Momentum (Rate of Change)
            if len(prices) >= 10:
                roc = ((prices - prices.shift(10)) / prices.shift(10)) * 100
                ax3.plot(dates, roc, color='#ff88aa', linewidth=2, label='10-Day ROC (%)')
                ax3.axhline(y=0, color='white', linestyle='-', alpha=0.5)
                ax3.set_ylabel('ROC (%)', color='white')
                ax3.set_xlabel('Date', color='white')
                ax3.tick_params(colors='white')
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper right')
            
            # Format x-axis for all subplots
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, color='white')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

class CSVImporter:
    """Handle CSV file import functionality."""
    
    def __init__(self, parent):
        self.parent = parent
    
    def import_stocks_from_csv(self) -> List[str]:
        """Import stock symbols from CSV file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select CSV file with stock symbols",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                parent=self.parent
            )
            
            if not file_path:
                return []
            
            # Try to read the CSV file
            df = pd.read_csv(file_path)
            
            # Show preview and column selection dialog
            symbols = self.show_csv_preview(df)
            return symbols
            
        except Exception as e:
            messagebox.showerror("CSV Import Error", f"Error reading CSV file: {e}")
            return []
    
    def show_csv_preview(self, df: pd.DataFrame) -> List[str]:
        """Show CSV preview and let user select column with symbols."""
        preview_window = tk.Toplevel(self.parent)
        preview_window.title("📊 CSV Import - Select Symbol Column")
        preview_window.geometry("800x600")
        preview_window.configure(bg='#1e1e1e')
        
        # Title
        title_label = tk.Label(preview_window, 
                              text="📊 CSV Import - Select Stock Symbol Column", 
                              font=('Arial', 14, 'bold'),
                              bg='#1e1e1e', fg='#00ff88')
        title_label.pack(pady=10)
        
        # Column selection frame
        selection_frame = tk.Frame(preview_window, bg='#1e1e1e')
        selection_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(selection_frame, text="Select column containing stock symbols:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        
        column_var = tk.StringVar()
        column_combo = ttk.Combobox(selection_frame, textvariable=column_var, 
                                   values=list(df.columns), state="readonly", width=20)
        column_combo.pack(side=tk.LEFT, padx=10)
        
        # Set default selection
        if 'symbol' in [col.lower() for col in df.columns]:
            default_col = [col for col in df.columns if col.lower() == 'symbol'][0]
            column_combo.set(default_col)
        elif 'ticker' in [col.lower() for col in df.columns]:
            default_col = [col for col in df.columns if col.lower() == 'ticker'][0]
            column_combo.set(default_col)
        elif len(df.columns) > 0:
            column_combo.set(df.columns[0])
        
        # Preview frame
        preview_frame = tk.Frame(preview_window, bg='#1e1e1e')
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create treeview for preview
        preview_tree = ttk.Treeview(preview_frame, columns=list(df.columns), show='headings', height=15)
        
        # Configure columns
        for col in df.columns:
            preview_tree.heading(col, text=col)
            preview_tree.column(col, width=100, anchor=tk.CENTER)
        
        # Add data (first 20 rows for preview)
        for i, row in df.head(20).iterrows():
            preview_tree.insert('', tk.END, values=list(row))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=preview_tree.yview)
        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=preview_tree.xview)
        preview_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack preview
        preview_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # Info label
        info_label = tk.Label(preview_window, 
                             text=f"Preview showing first 20 rows of {len(df)} total rows",
                             bg='#1e1e1e', fg='#888888', font=('Arial', 9))
        info_label.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(preview_window, bg='#1e1e1e')
        button_frame.pack(pady=20)
        
        selected_symbols = []
        
        def import_symbols():
            nonlocal selected_symbols
            selected_column = column_var.get()
            if not selected_column:
                messagebox.showwarning("No Column Selected", "Please select a column containing stock symbols")
                return
            
            try:
                # Extract symbols from selected column
                symbols = df[selected_column].dropna().astype(str).str.upper().str.strip()
                symbols = symbols[symbols != ''].unique().tolist()
                
                # Filter out invalid symbols (basic validation)
                valid_symbols = []
                for symbol in symbols:
                    if symbol.isalpha() and len(symbol) <= 5:  # Basic symbol validation
                        valid_symbols.append(symbol)
                
                selected_symbols = valid_symbols
                messagebox.showinfo("Import Successful", 
                                   f"Found {len(selected_symbols)} valid stock symbols")
                preview_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Import Error", f"Error processing symbols: {e}")
        
        def cancel_import():
            preview_window.destroy()
        
        tk.Button(button_frame, text="Import Symbols", command=import_symbols,
                 bg='#00ff88', fg='black', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Cancel", command=cancel_import,
                 bg='#666666', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        
        # Wait for window to close
        preview_window.wait_window()
        return selected_symbols
    
    def export_watchlist_to_csv(self, watchlist: List[str], stock_data_dict: Dict):
        """Export current watchlist with data to CSV."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Watchlist to CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                parent=self.parent
            )
            
            if not file_path:
                return
            
            # Prepare data for export
            export_data = []
            for symbol in watchlist:
                if symbol in stock_data_dict:
                    data = stock_data_dict[symbol]
                    export_data.append({
                        'Symbol': symbol,
                        'Company': data.get('company_name', symbol),
                        'Price': data.get('current_price', 'N/A'),
                        'Market_Cap': data.get('market_cap_formatted', 'N/A'),
                        'Sector': data.get('sector', 'N/A'),
                        'Industry': data.get('industry', 'N/A'),
                        'Daily_Volume': data.get('daily_volume', 'N/A'),
                        'Export_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    export_data.append({
                        'Symbol': symbol,
                        'Company': symbol,
                        'Price': 'N/A',
                        'Market_Cap': 'N/A',
                        'Sector': 'N/A',
                        'Industry': 'N/A',
                        'Daily_Volume': 'N/A',
                        'Export_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # Write to CSV
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Successful", 
                               f"Watchlist exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting watchlist: {e}")

class RSIStockTracker:
    """Main application class for Enhanced RSI Stock Tracker with Charts and CSV Import."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced RSI Stock Tracker - Real-Time Analysis with Charts & CSV Import")
        self.root.geometry("1600x800")  # Wider window for more features
        self.root.configure(bg='#1e1e1e')  # Dark theme
        
        # Data handlers
        self.stock_data = StockData()
        self.rsi_calculator = RSICalculator()
        self.chart_window = ChartWindow(self.root, self.stock_data, self.rsi_calculator)
        self.csv_importer = CSVImporter(self.root)
        
        # Watchlist and data storage
        self.watchlist = []
        self.stock_data_dict = {}  # Store comprehensive data for each stock
        self.watchlist_file = "watchlist.json"
        self.load_watchlist()
        
        # Update control
        self.update_interval = 30  # seconds
        self.is_updating = False
        self.update_thread = None
        
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
        """Setup the enhanced user interface."""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="📈 Enhanced RSI Stock Tracker - Charts & CSV Import", 
                              font=('Arial', 16, 'bold'),
                              bg='#1e1e1e', fg='#00ff88')
        title_label.pack(pady=(0, 10))
        
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
        
        # CSV Import section
        csv_frame = tk.Frame(control_frame, bg='#1e1e1e')
        csv_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        tk.Button(csv_frame, text="📊 Import CSV", command=self.import_csv,
                 bg='#4488ff', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(csv_frame, text="💾 Export CSV", command=self.export_csv,
                 bg='#ff8800', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#1e1e1e')
        button_frame.pack(side=tk.RIGHT)
        
        tk.Button(button_frame, text="📈 Show Chart", command=self.show_chart,
                 bg='#aa44ff', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
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
        
        # Enhanced legend
        self.setup_legend(main_frame)
    
    def setup_table(self, parent):
        """Setup the enhanced stock data table."""
        # Table frame with horizontal scrollbar
        table_frame = tk.Frame(parent, bg='#1e1e1e')
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for stock data with enhanced columns
        columns = ('Symbol', 'Company', 'Price', 'Change', 'Change%', 'RSI', 'Status', 
                  'MarketCap', 'Sector', 'DailyVol', 'AvgVol', 'Bid', 'Ask', 'Updated')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Define headings with optimized widths
        headings = {
            'Symbol': ('Stock', 70),
            'Company': ('Company', 120),
            'Price': ('Price ($)', 80),
            'Change': ('Change ($)', 80), 
            'Change%': ('Change %', 80),
            'RSI': ('RSI', 60),
            'Status': ('RSI Status', 100),
            'MarketCap': ('Market Cap', 90),
            'Sector': ('Sector', 100),
            'DailyVol': ('Daily Vol', 80),
            'AvgVol': ('Avg Vol', 80),
            'Bid': ('Bid ($)', 70),
            'Ask': ('Ask ($)', 70),
            'Updated': ('Updated', 80)
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
        
        # Bind events
        self.tree.bind('<Double-1>', self.on_double_click)
        self.tree.bind('<Button-3>', self.show_context_menu)  # Right-click menu
    
    def setup_legend(self, parent):
        """Setup enhanced legend with new features."""
        legend_frame = tk.Frame(parent, bg='#1e1e1e')
        legend_frame.pack(pady=10)
        
        tk.Label(legend_frame, text="RSI Legend & Features:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold')).pack()
        
        legend_text = ("🔴 Overbought (>70)  🟡 Neutral (30-70)  🟢 Oversold (<30)  |  "
                      "Double-click: Seeking Alpha  |  Right-click: Context Menu  |  📈 Charts Available")
        tk.Label(legend_frame, text=legend_text,
                bg='#1e1e1e', fg='#cccccc', font=('Arial', 9)).pack()
    
    def show_context_menu(self, event):
        """Show right-click context menu."""
        # Select the item that was right-clicked
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            
            # Create context menu
            context_menu = tk.Menu(self.root, tearoff=0, bg='#2d2d2d', fg='white')
            context_menu.add_command(label="📈 Show Charts", command=self.show_chart)
            context_menu.add_command(label="🔗 Open Seeking Alpha", command=self.open_seeking_alpha)
            context_menu.add_separator()
            context_menu.add_command(label="❌ Remove Stock", command=self.remove_stock)
            
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()
    
    def show_chart(self):
        """Show advanced charts for selected stock."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock to show charts")
            return
        
        symbol = selected[0]
        self.status_label.config(text=f"Loading charts for {symbol}...")
        
        # Show charts in separate thread to avoid blocking UI
        threading.Thread(target=self.chart_window.show_charts, args=(symbol,), daemon=True).start()
        
        self.status_label.config(text=f"Opened chart window for {symbol}")
    
    def import_csv(self):
        """Import stocks from CSV file."""
        self.status_label.config(text="Opening CSV import...")
        
        # Import symbols from CSV
        symbols = self.csv_importer.import_stocks_from_csv()
        
        if symbols:
            added_count = 0
            duplicate_count = 0
            
            for symbol in symbols:
                if symbol not in self.watchlist:
                    self.watchlist.append(symbol)
                    
                    # Add to table with loading state
                    loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "", "", "", "")
                    self.tree.insert('', tk.END, iid=symbol, values=loading_values)
                    added_count += 1
                else:
                    duplicate_count += 1
            
            if added_count > 0:
                self.save_watchlist()
                # Update display for new stocks
                threading.Thread(target=self.refresh_imported_stocks, args=(symbols,), daemon=True).start()
                
                message = f"Added {added_count} new stocks"
                if duplicate_count > 0:
                    message += f" ({duplicate_count} duplicates skipped)"
                
                self.status_label.config(text=message)
                messagebox.showinfo("CSV Import Complete", message)
            else:
                self.status_label.config(text="No new stocks imported")
        else:
            self.status_label.config(text="CSV import cancelled")
    
    def export_csv(self):
        """Export current watchlist to CSV."""
        if not self.watchlist:
            messagebox.showwarning("Empty Watchlist", "No stocks to export")
            return
        
        self.status_label.config(text="Exporting watchlist to CSV...")
        self.csv_importer.export_watchlist_to_csv(self.watchlist, self.stock_data_dict)
        self.status_label.config(text="Watchlist exported successfully")
    
    def refresh_imported_stocks(self, symbols):
        """Refresh data for newly imported stocks."""
        for symbol in symbols:
            if symbol in self.watchlist:  # Only update if still in watchlist
                self.update_stock_data(symbol)
                time.sleep(0.5)  # Avoid rate limiting
    
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
        
        # Validate symbol by trying to fetch data
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
        
        # Add to table with loading state
        loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "", "", "", "")
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
            if symbol in self.stock_data_dict:
                del self.stock_data_dict[symbol]
            self.tree.delete(symbol)
            self.save_watchlist()
            self.status_label.config(text=f"Removed {symbol} from watchlist")
    
    def open_seeking_alpha(self):
        """Open Seeking Alpha for selected stock."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock to view on Seeking Alpha")
            return
        
        symbol = selected[0]
        self.open_seeking_alpha_url(symbol)
    
    def on_double_click(self, event):
        """Handle double-click on table row."""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            self.open_seeking_alpha_url(item)
    
    def open_seeking_alpha_url(self, symbol: str):
        """Open Seeking Alpha URL for given symbol."""
        url = f"https://seekingalpha.com/symbol/{symbol}"
        try:
            webbrowser.open(url)
            self.status_label.config(text=f"Opened Seeking Alpha for {symbol}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Seeking Alpha: {e}")
    
    def manual_refresh(self):
        """Manually refresh all stock data."""
        if not self.watchlist:
            messagebox.showinfo("Empty Watchlist", "Add some stocks to your watchlist first!")
            return
        
        self.status_label.config(text="Refreshing all stocks...")
        threading.Thread(target=self.refresh_all_stocks, daemon=True).start()
    
    def refresh_all_stocks(self):
        """Refresh data for all stocks in watchlist."""
        for i, symbol in enumerate(self.watchlist):
            self.root.after(0, lambda s=symbol: self.status_label.config(text=f"Updating {s}..."))
            self.update_stock_data(symbol)
            time.sleep(0.5)  # Avoid rate limiting
        
        self.root.after(0, lambda: self.status_label.config(text="All stocks updated"))
    
    def update_stock_data(self, symbol: str):
        """Update comprehensive data for a specific stock."""
        # Get historical data for RSI calculation
        hist_data = self.stock_data.get_stock_data(symbol, "1mo")
        
        # Get comprehensive stock info
        stock_info = self.stock_data.get_comprehensive_stock_info(symbol)
        
        if hist_data is None or hist_data.empty:
            error_values = (symbol, "Error", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error")
            self.root.after(0, lambda: self.update_table_row(symbol, *error_values))
            return
        
        try:
            # Calculate price changes
            current_price = stock_info['current_price']
            if current_price is None:
                current_price = float(hist_data['Close'].iloc[-1])
            
            previous_price = float(hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else current_price
            price_change = current_price - previous_price
            percent_change = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            # Calculate RSI
            rsi_series = self.rsi_calculator.calculate_rsi(hist_data['Close'])
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
            
            # Determine RSI status
            if rsi > 70:
                rsi_status = "🔴 Overbought"
            elif rsi < 30:
                rsi_status = "🟢 Oversold"
            else:
                rsi_status = "🟡 Neutral"
            
            # Format all values
            company_name = stock_info['company_name'][:15] + "..." if len(stock_info['company_name']) > 15 else stock_info['company_name']
            price_str = f"${current_price:.2f}"
            change_str = f"${price_change:+.2f}"
            percent_str = f"{percent_change:+.2f}%"
            rsi_str = f"{rsi:.1f}"
            market_cap_str = stock_info['market_cap_formatted']
            sector_str = stock_info['sector'][:10] + "..." if len(stock_info['sector']) > 10 else stock_info['sector']
            daily_vol_str = self.stock_data.format_volume(stock_info['daily_volume'])
            avg_vol_str = self.stock_data.format_volume(stock_info['avg_volume'])
            bid_str = f"${stock_info['bid_price']:.2f}" if stock_info['bid_price'] > 0 else "N/A"
            ask_str = f"${stock_info['ask_price']:.2f}" if stock_info['ask_price'] > 0 else "N/A"
            updated_str = datetime.now().strftime("%H:%M:%S")
            
            # Store comprehensive data
            self.stock_data_dict[symbol] = stock_info
            
            # Update table in main thread
            values = (symbol, company_name, price_str, change_str, percent_str, rsi_str, rsi_status,
                     market_cap_str, sector_str, daily_vol_str, avg_vol_str, bid_str, ask_str, updated_str)
            
            self.root.after(0, lambda: self.update_table_row(symbol, *values))
            
        except Exception as e:
            print(f"Error updating {symbol}: {e}")
            error_values = (symbol, "Error", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error")
            self.root.after(0, lambda: self.update_table_row(symbol, *error_values))
    
    def update_table_row(self, symbol: str, *values):
        """Update a row in the stock table."""
        try:
            self.tree.item(symbol, values=values)
        except tk.TclError:
            pass  # Item might have been deleted
    
    def start_updates(self):
        """Start automatic updates."""
        if not self.is_updating:
            self.is_updating = True
            self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
            self.update_thread.start()
    
    def update_loop(self):
        """Main update loop running in background."""
        while self.is_updating:
            if self.watchlist:
                self.root.after(0, lambda: self.status_label.config(text="Updating stocks..."))
                
                for symbol in self.watchlist.copy():  # Use copy to avoid modification during iteration
                    if not self.is_updating:
                        break
                    self.update_stock_data(symbol)
                    time.sleep(2)  # Increased delay for comprehensive data fetching
                
                current_time = datetime.now().strftime("%H:%M:%S")
                self.root.after(0, lambda: self.last_update_label.config(text=f"Last update: {current_time}"))
                self.root.after(0, lambda: self.status_label.config(text="Ready"))
            
            # Wait before next update cycle
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
            loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "", "", "", "")
            self.tree.insert('', tk.END, iid=symbol, values=loading_values)
    
    def on_closing(self):
        """Handle application closing."""
        self.is_updating = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        # Populate with saved stocks
        self.populate_initial_data()
        
        # Show instructions if empty
        if not self.watchlist:
            self.status_label.config(text="Add stock symbols or import CSV to start tracking")
        
        # Start the GUI
        self.root.mainloop()

def main():
    """Main function to run the Enhanced RSI Stock Tracker with Charts and CSV Import."""
    try:
        app = RSIStockTracker()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()
