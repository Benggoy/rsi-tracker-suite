#!/usr/bin/env python3
"""
Enhanced RSI Stock Tracker with Working Charts & Portfolio Import
Fixed version with robust chart functionality and comprehensive error handling.

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

# Chart imports with error handling
CHARTS_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Force TkAgg backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    plt.style.use('dark_background')
    CHARTS_AVAILABLE = True
    print("✅ Charts available - matplotlib loaded successfully")
except ImportError as e:
    print(f"⚠️ Charts not available - matplotlib import failed: {e}")
    CHARTS_AVAILABLE = False

class RSICalculator:
    """Calculate RSI and technical indicators."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for given price series."""
        if len(prices) < period + 1:
            return pd.Series([50.0] * len(prices), index=prices.index)
        
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0)
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series, periods: List[int] = [20, 50]) -> Dict[str, pd.Series]:
        """Calculate moving averages."""
        mas = {}
        for period in periods:
            if len(prices) >= period:
                mas[f'MA{period}'] = prices.rolling(window=period).mean()
        return mas

class StockData:
    """Handle stock data fetching and processing."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60
    
    def get_stock_data(self, symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance."""
        cache_key = f"{symbol}_{period}"
        current_time = time.time()
        
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return data
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            self.cache[cache_key] = (data, current_time)
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_comprehensive_stock_info(self, symbol: str) -> Dict:
        """Get comprehensive stock information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist_data = self.get_stock_data(symbol, "1mo")
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price and hist_data is not None and not hist_data.empty:
                current_price = float(hist_data['Close'].iloc[-1])
            
            market_cap = info.get('marketCap', 0)
            market_cap_formatted = self.format_market_cap(market_cap)
            
            daily_volume = info.get('volume', 0) or info.get('regularMarketVolume', 0)
            avg_volume = info.get('averageVolume', 0) or info.get('averageVolume10days', 0)
            
            bid_price = info.get('bid', 0)
            ask_price = info.get('ask', 0)
            
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
                'sector': info.get('sector', 'N/A')
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
                'sector': 'N/A'
            }
    
    def format_market_cap(self, market_cap: int) -> str:
        """Format market cap in readable format."""
        if market_cap == 0:
            return "N/A"
        elif market_cap >= 1_000_000_000_000:
            return f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            return f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            return f"${market_cap / 1_000_000:.1f}M"
        elif market_cap >= 1_000:
            return f"${market_cap / 1_000:.1f}K"
        else:
            return f"${market_cap:,.0f}"
    
    def format_volume(self, volume: int) -> str:
        """Format volume in readable format."""
        if volume == 0:
            return "N/A"
        elif volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.1f}K"
        else:
            return f"{volume:,}"

class SimpleChartManager:
    """Simple, robust chart manager."""
    
    def __init__(self, rsi_calculator):
        self.rsi_calculator = rsi_calculator
        self.current_canvas = None
    
    def create_chart(self, parent_frame, symbol: str, data: pd.DataFrame, period: str = "1mo"):
        """Create a simple but effective chart."""
        if not CHARTS_AVAILABLE:
            self.show_no_charts_message(parent_frame, symbol)
            return None
        
        # Clear previous chart
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        try:
            # Create figure
            fig = Figure(figsize=(12, 8), facecolor='#1e1e1e', tight_layout=True)
            
            # Price chart (top)
            ax1 = fig.add_subplot(211, facecolor='#2d2d2d')
            
            # Plot price
            ax1.plot(data.index, data['Close'], color='#00ff88', linewidth=2, label='Price')
            
            # Add moving averages if enough data
            ma_data = self.rsi_calculator.calculate_moving_averages(data['Close'], [20, 50])
            if 'MA20' in ma_data and len(ma_data['MA20'].dropna()) > 0:
                ax1.plot(data.index, ma_data['MA20'], color='#ffaa00', linewidth=1, alpha=0.8, label='MA20')
            if 'MA50' in ma_data and len(ma_data['MA50'].dropna()) > 0:
                ax1.plot(data.index, ma_data['MA50'], color='#ff4444', linewidth=1, alpha=0.8, label='MA50')
            
            ax1.set_title(f'{symbol} - Price Chart ({period})', color='white', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', color='white')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='white')
            
            # RSI chart (bottom)
            ax2 = fig.add_subplot(212, facecolor='#2d2d2d')
            rsi_data = self.rsi_calculator.calculate_rsi(data['Close'])
            
            # Plot RSI
            ax2.plot(data.index, rsi_data, color='#4488ff', linewidth=2, label='RSI')
            
            # Add RSI levels
            ax2.axhline(y=70, color='#ff4444', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(y=30, color='#44ff44', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.axhline(y=50, color='#888888', linestyle='-', alpha=0.5, label='Neutral (50)')
            
            ax2.set_title('RSI (14-period)', color='white', fontsize=12)
            ax2.set_ylabel('RSI', color='white')
            ax2.set_xlabel('Date', color='white')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors='white')
            
            # Format dates
            if len(data) > 20:
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                fig.autofmt_xdate()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, parent_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.current_canvas = canvas
            return canvas
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            self.show_chart_error(parent_frame, symbol, str(e))
            return None
    
    def show_no_charts_message(self, parent_frame, symbol):
        """Show message when charts are not available."""
        for widget in parent_frame.winfo_children():
            widget.destroy()
            
        message_frame = tk.Frame(parent_frame, bg='#1e1e1e')
        message_frame.pack(expand=True, fill=tk.BOTH)
        
        tk.Label(message_frame, 
                text="📊 Charts Not Available", 
                bg='#1e1e1e', fg='#ffaa00', 
                font=('Arial', 16, 'bold')).pack(pady=20)
        
        tk.Label(message_frame, 
                text=f"Chart functionality is not available for {symbol}.\n\n"
                     "To enable charts, install matplotlib:\n"
                     "pip3 install matplotlib\n\n"
                     "Then rebuild your app.",
                bg='#1e1e1e', fg='#cccccc', 
                font=('Arial', 12)).pack(pady=10)
        
        tk.Button(message_frame, 
                 text="📈 View on Yahoo Finance", 
                 command=lambda: webbrowser.open(f"https://finance.yahoo.com/quote/{symbol}"),
                 bg='#4488ff', fg='white', 
                 font=('Arial', 11, 'bold')).pack(pady=20)
    
    def show_chart_error(self, parent_frame, symbol, error_msg):
        """Show error message when chart creation fails."""
        for widget in parent_frame.winfo_children():
            widget.destroy()
            
        error_frame = tk.Frame(parent_frame, bg='#1e1e1e')
        error_frame.pack(expand=True, fill=tk.BOTH)
        
        tk.Label(error_frame, 
                text="⚠️ Chart Error", 
                bg='#1e1e1e', fg='#ff4444', 
                font=('Arial', 16, 'bold')).pack(pady=20)
        
        tk.Label(error_frame, 
                text=f"Could not create chart for {symbol}.\n\n"
                     f"Error: {error_msg}\n\n"
                     "Try a different timeframe or check your internet connection.",
                bg='#1e1e1e', fg='#cccccc', 
                font=('Arial', 12)).pack(pady=10)
        
        tk.Button(error_frame, 
                 text="📈 View on Yahoo Finance", 
                 command=lambda: webbrowser.open(f"https://finance.yahoo.com/quote/{symbol}"),
                 bg='#4488ff', fg='white', 
                 font=('Arial', 11, 'bold')).pack(pady=20)

class PortfolioImporter:
    """Handle portfolio import from various sources."""
    
    @staticmethod
    def import_from_csv(file_path: str) -> List[str]:
        """Import symbols from CSV file."""
        symbols = []
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                sample = csvfile.read(1024)
                csvfile.seek(0)
                delimiter = ',' if ',' in sample else '\t' if '\t' in sample else ';'
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                symbol_columns = ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker', 'TICKER', 
                                'stock', 'Stock', 'STOCK', 'code', 'Code', 'CODE']
                
                symbol_col = None
                for col in reader.fieldnames:
                    if col in symbol_columns:
                        symbol_col = col
                        break
                
                if not symbol_col:
                    symbol_col = reader.fieldnames[0] if reader.fieldnames else None
                
                if symbol_col:
                    for row in reader:
                        symbol = row.get(symbol_col, '').strip().upper()
                        if symbol and len(symbol) <= 5:
                            symbols.append(symbol)
                            
        except Exception as e:
            print(f"Error importing CSV: {e}")
            
        return list(set(symbols))
    
    @staticmethod
    def import_from_text(file_path: str) -> List[str]:
        """Import symbols from text file."""
        symbols = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    symbol = line.strip().upper()
                    if symbol and len(symbol) <= 5:
                        symbols.append(symbol)
        except Exception as e:
            print(f"Error importing text file: {e}")
            
        return list(set(symbols))

class EnhancedRSITracker:
    """Enhanced RSI Stock Tracker with Working Charts."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced RSI Stock Tracker - Charts & Portfolio Import")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1e1e1e')
        
        # Data handlers
        self.stock_data = StockData()
        self.rsi_calculator = RSICalculator()
        self.portfolio_importer = PortfolioImporter()
        self.chart_manager = SimpleChartManager(self.rsi_calculator)
        
        # Current selected stock
        self.selected_symbol = None
        
        # Watchlist
        self.watchlist = []
        self.watchlist_file = "watchlist.json"
        self.load_watchlist()
        
        # Update control
        self.update_interval = 30
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
        """Setup custom styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Treeview', 
                       background='#2d2d2d', 
                       foreground='white',
                       fieldbackground='#2d2d2d',
                       rowheight=25)
        style.configure('Treeview.Heading',
                       background='#404040',
                       foreground='white',
                       font=('Arial', 9, 'bold'))
        
        style.configure('TNotebook', background='#1e1e1e')
        style.configure('TNotebook.Tab', background='#404040', foreground='white', padding=[20, 8])
        style.map('TNotebook.Tab', background=[('selected', '#00ff88')], foreground=[('selected', 'black')])
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        chart_status = "✅ Charts Enabled" if CHARTS_AVAILABLE else "⚠️ Charts Disabled"
        title_label = tk.Label(main_frame, 
                              text=f"📈 Enhanced RSI Stock Tracker - {chart_status}", 
                              font=('Arial', 18, 'bold'),
                              bg='#1e1e1e', fg='#00ff88')
        title_label.pack(pady=(0, 10))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.setup_tracker_tab()
        self.setup_charts_tab()
        
        # Bind tab change
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def setup_tracker_tab(self):
        """Setup the main tracker tab."""
        tracker_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(tracker_frame, text="📊 Stock Tracker")
        
        # Control frame
        control_frame = tk.Frame(tracker_frame, bg='#1e1e1e')
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
        
        # Import section
        import_frame = tk.Frame(control_frame, bg='#1e1e1e')
        import_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        tk.Button(import_frame, text="📁 Import Portfolio", command=self.import_portfolio,
                 bg='#8844ff', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        tk.Button(import_frame, text="📋 Bulk Add", command=self.bulk_add_stocks,
                 bg='#44ff88', fg='black', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(5, 0))
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#1e1e1e')
        button_frame.pack(side=tk.RIGHT)
        
        chart_button_text = "📈 View Chart" if CHARTS_AVAILABLE else "📈 Charts (Disabled)"
        chart_button_color = "#ff8844" if CHARTS_AVAILABLE else "#666666"
        tk.Button(button_frame, text=chart_button_text, command=self.view_chart,
                 bg=chart_button_color, fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Remove Selected", command=self.remove_stock,
                 bg='#ff4444', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Seeking Alpha", command=self.open_seeking_alpha,
                 bg='#ffaa00', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Refresh Now", command=self.manual_refresh,
                 bg='#4488ff', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = tk.Frame(tracker_frame, bg='#1e1e1e')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Ready", 
                                    bg='#1e1e1e', fg='#cccccc', font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT)
        
        self.last_update_label = tk.Label(status_frame, text="", 
                                         bg='#1e1e1e', fg='#888888', font=('Arial', 9))
        self.last_update_label.pack(side=tk.RIGHT)
        
        # Stock data table
        self.setup_table(tracker_frame)
        
        # Legend
        self.setup_legend(tracker_frame)
    
    def setup_charts_tab(self):
        """Setup the charts tab."""
        charts_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        chart_tab_text = "📈 Charts" if CHARTS_AVAILABLE else "📈 Charts (Disabled)"
        self.notebook.add(charts_frame, text=chart_tab_text)
        
        # Chart controls
        chart_control_frame = tk.Frame(charts_frame, bg='#1e1e1e')
        chart_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(chart_control_frame, text="Stock Symbol:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.chart_symbol_var = tk.StringVar()
        self.chart_symbol_combo = ttk.Combobox(chart_control_frame, textvariable=self.chart_symbol_var, 
                                              width=8, font=('Arial', 10))
        self.chart_symbol_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        tk.Label(chart_control_frame, text="Period:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.chart_period_var = tk.StringVar(value="3mo")
        period_combo = ttk.Combobox(chart_control_frame, textvariable=self.chart_period_var, 
                                   values=["5d", "1mo", "3mo", "6mo", "1y"], 
                                   width=8, font=('Arial', 10))
        period_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        load_button_text = "📊 Load Chart" if CHARTS_AVAILABLE else "Charts Disabled"
        load_button_color = "#00ff88" if CHARTS_AVAILABLE else "#666666"
        tk.Button(chart_control_frame, text=load_button_text, command=self.load_chart,
                 bg=load_button_color, fg='black' if CHARTS_AVAILABLE else 'white', 
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Chart display area
        self.chart_frame = tk.Frame(charts_frame, bg='#1e1e1e')
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        if CHARTS_AVAILABLE:
            initial_text = "Select a stock symbol and click 'Load Chart' to view price and RSI analysis"
        else:
            initial_text = "Charts are not available. Install matplotlib and rebuild the app to enable charts."
        
        self.chart_message = tk.Label(self.chart_frame, 
                                     text=initial_text,
                                     bg='#1e1e1e', fg='#cccccc', font=('Arial', 12))
        self.chart_message.pack(expand=True)
    
    def setup_table(self, parent):
        """Setup the stock data table."""
        table_frame = tk.Frame(parent, bg='#1e1e1e')
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Symbol', 'Price', 'Change', 'Change%', 'RSI', 'Status', 
                  'MarketCap', 'DailyVol', 'AvgVol', 'Bid', 'Ask', 'Updated')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        headings = {
            'Symbol': ('Stock', 70),
            'Price': ('Price ($)', 80),
            'Change': ('Change ($)', 80), 
            'Change%': ('Change %', 80),
            'RSI': ('RSI', 60),
            'Status': ('RSI Status', 100),
            'MarketCap': ('Market Cap', 90),
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
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        self.tree.bind('<<TreeviewSelect>>', self.on_stock_select)
        self.tree.bind('<Double-1>', self.on_double_click)
    
    def setup_legend(self, parent):
        """Setup RSI interpretation legend."""
        legend_frame = tk.Frame(parent, bg='#1e1e1e')
        legend_frame.pack(pady=10)
        
        tk.Label(legend_frame, text="RSI Legend:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold')).pack()
        
        chart_status = "Select + View Chart for Analysis" if CHARTS_AVAILABLE else "Charts Disabled (install matplotlib)"
        legend_text = f"🔴 Overbought (>70)  🟡 Neutral (30-70)  🟢 Oversold (<30)  | Double-click: Seeking Alpha | {chart_status}"
        tk.Label(legend_frame, text=legend_text,
                bg='#1e1e1e', fg='#cccccc', font=('Arial', 9)).pack()
    
    def on_stock_select(self, event):
        """Handle stock selection."""
        selected = self.tree.selection()
        if selected:
            self.selected_symbol = selected[0]
            self.chart_symbol_var.set(self.selected_symbol)
            self.chart_symbol_combo['values'] = self.watchlist
    
    def on_tab_changed(self, event):
        """Handle tab change."""
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        
        if "Charts" in tab_text:
            self.chart_symbol_combo['values'] = self.watchlist
            if self.selected_symbol:
                self.chart_symbol_var.set(self.selected_symbol)
    
    def view_chart(self):
        """Switch to charts tab and load chart."""
        if not CHARTS_AVAILABLE:
            messagebox.showwarning("Charts Not Available", 
                                 "Charts are disabled. Install matplotlib to enable charts:\n\n"
                                 "pip3 install matplotlib\n\n"
                                 "Then rebuild your app.")
            return
            
        if not self.selected_symbol:
            messagebox.showwarning("No Selection", "Please select a stock to view its chart")
            return
        
        self.notebook.select(1)  # Select charts tab
        self.chart_symbol_var.set(self.selected_symbol)
        self.load_chart()
    
    def load_chart(self):
        """Load chart for selected symbol."""
        if not CHARTS_AVAILABLE:
            messagebox.showwarning("Charts Not Available", 
                                 "Install matplotlib to enable charts: pip3 install matplotlib")
            return
            
        symbol = self.chart_symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showwarning("No Symbol", "Please enter a stock symbol")
            return
        
        period = self.chart_period_var.get()
        
        self.chart_message.config(text=f"Loading chart for {symbol}...")
        self.root.update()
        
        # Fetch data
        data = self.stock_data.get_stock_data(symbol, period)
        if data is None or data.empty:
            messagebox.showerror("Data Error", f"Could not fetch data for {symbol}")
            self.chart_message.config(text="Select a stock symbol and click 'Load Chart' to view analysis")
            return
        
        # Hide message and create chart
        self.chart_message.pack_forget()
        
        try:
            self.chart_manager.create_chart(self.chart_frame, symbol, data, period)
            self.status_label.config(text=f"Chart loaded for {symbol}")
        except Exception as e:
            print(f"Chart error: {e}")
            self.chart_message.pack(expand=True)
            self.chart_message.config(text="Error loading chart. Please try again.")
    
    # Import and other methods (keeping them simple and working)
    def import_portfolio(self):
        """Import portfolio from file."""
        file_path = filedialog.askopenfilename(
            title="Import Portfolio",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_label.config(text="Importing portfolio...")
        self.root.update()
        
        symbols = []
        if file_path.lower().endswith('.csv'):
            symbols = self.portfolio_importer.import_from_csv(file_path)
        elif file_path.lower().endswith('.txt'):
            symbols = self.portfolio_importer.import_from_text(file_path)
        else:
            symbols = self.portfolio_importer.import_from_csv(file_path)
            if not symbols:
                symbols = self.portfolio_importer.import_from_text(file_path)
        
        if not symbols:
            messagebox.showerror("Import Error", "No valid symbols found in the file.")
            self.status_label.config(text="Ready")
            return
        
        added_count = 0
        for symbol in symbols:
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)
                loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "", "")
                self.tree.insert('', tk.END, iid=symbol, values=loading_values)
                added_count += 1
        
        if added_count > 0:
            self.save_watchlist()
            threading.Thread(target=self.refresh_new_stocks, args=(symbols,), daemon=True).start()
            
        messagebox.showinfo("Import Complete", 
                          f"Successfully imported {added_count} new stocks.\n"
                          f"Skipped {len(symbols) - added_count} duplicates.")
        
        self.status_label.config(text=f"Imported {added_count} stocks from portfolio")
    
    def bulk_add_stocks(self):
        """Bulk add stocks dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Bulk Add Stocks")
        dialog.geometry("400x300")
        dialog.configure(bg='#1e1e1e')
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Enter stock symbols (one per line):", 
                bg='#1e1e1e', fg='white', font=('Arial', 12, 'bold')).pack(pady=10)
        
        tk.Label(dialog, text="Example:\nAAPL\nGOOGL\nTSLA\nMSFT", 
                bg='#1e1e1e', fg='#cccccc', font=('Arial', 10)).pack()
        
        text_frame = tk.Frame(dialog, bg='#1e1e1e')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        text_area = tk.Text(text_frame, height=8, width=40, font=('Arial', 11))
        scrollbar = tk.Scrollbar(text_frame, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)
        
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        button_frame = tk.Frame(dialog, bg='#1e1e1e')
        button_frame.pack(pady=10)
        
        def add_symbols():
            content = text_area.get("1.0", tk.END).strip()
            if not content:
                messagebox.showwarning("Empty Input", "Please enter some stock symbols.")
                return
            
            symbols = [line.strip().upper() for line in content.split('\n') 
                      if line.strip() and len(line.strip()) <= 5]
            
            if not symbols:
                messagebox.showwarning("Invalid Input", "No valid symbols found.")
                return
            
            added_count = 0
            for symbol in symbols:
                if symbol not in self.watchlist:
                    self.watchlist.append(symbol)
                    loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "", "")
                    self.tree.insert('', tk.END, iid=symbol, values=loading_values)
                    added_count += 1
            
            if added_count > 0:
                self.save_watchlist()
                threading.Thread(target=self.refresh_new_stocks, args=(symbols,), daemon=True).start()
            
            dialog.destroy()
            messagebox.showinfo("Bulk Add Complete", 
                              f"Successfully added {added_count} new stocks.\n"
                              f"Skipped {len(symbols) - added_count} duplicates.")
        
        tk.Button(button_frame, text="Add Stocks", command=add_symbols,
                 bg='#00ff88', fg='black', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                 bg='#ff4444', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
    
    def refresh_new_stocks(self, symbols: List[str]):
        """Refresh data for newly added stocks."""
        for symbol in symbols:
            if symbol in self.watchlist:
                self.update_stock_data(symbol)
                time.sleep(0.5)
    
    def add_stock_event(self, event):
        """Handle Enter key."""
        self.add_stock()
    
    def add_stock(self):
        """Add stock to watchlist."""
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            return
        
        if symbol in self.watchlist:
            messagebox.showwarning("Duplicate", f"{symbol} is already in your watchlist!")
            return
        
        self.status_label.config(text=f"Validating {symbol}...")
        self.root.update()
        
        data = self.stock_data.get_stock_data(symbol, "5d")
        if data is None or data.empty:
            messagebox.showerror("Invalid Symbol", f"Could not find data for {symbol}")
            self.status_label.config(text="Ready")
            return
        
        self.watchlist.append(symbol)
        self.symbol_entry.delete(0, tk.END)
        self.save_watchlist()
        
        loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "", "")
        self.tree.insert('', tk.END, iid=symbol, values=loading_values)
        
        threading.Thread(target=self.update_stock_data, args=(symbol,), daemon=True).start()
        self.status_label.config(text=f"Added {symbol} to watchlist")
    
    def remove_stock(self):
        """Remove selected stock."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock to remove")
            return
        
        symbol = selected[0]
        if messagebox.askyesno("Confirm Remove", f"Remove {symbol} from watchlist?"):
            self.watchlist.remove(symbol)
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
        url = f"https://seekingalpha.com/symbol/{symbol}"
        try:
            webbrowser.open(url)
            self.status_label.config(text=f"Opened Seeking Alpha for {symbol}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Seeking Alpha: {e}")
    
    def on_double_click(self, event):
        """Handle double-click."""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            url = f"https://seekingalpha.com/symbol/{item}"
            webbrowser.open(url)
    
    def manual_refresh(self):
        """Manual refresh all data."""
        if not self.watchlist:
            messagebox.showinfo("Empty Watchlist", "Add some stocks to your watchlist first!")
            return
        
        self.status_label.config(text="Refreshing all stocks...")
        threading.Thread(target=self.refresh_all_stocks, daemon=True).start()
    
    def refresh_all_stocks(self):
        """Refresh all stocks."""
        for symbol in self.watchlist:
            self.root.after(0, lambda s=symbol: self.status_label.config(text=f"Updating {s}..."))
            self.update_stock_data(symbol)
            time.sleep(0.5)
        
        self.root.after(0, lambda: self.status_label.config(text="All stocks updated"))
    
    def update_stock_data(self, symbol: str):
        """Update stock data."""
        hist_data = self.stock_data.get_stock_data(symbol, "1mo")
        stock_info = self.stock_data.get_comprehensive_stock_info(symbol)
        
        if hist_data is None or hist_data.empty:
            error_values = (symbol, "Error", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error")
            self.root.after(0, lambda: self.update_table_row(symbol, *error_values))
            return
        
        try:
            current_price = stock_info['current_price']
            if current_price is None:
                current_price = float(hist_data['Close'].iloc[-1])
            
            previous_price = float(hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else current_price
            price_change = current_price - previous_price
            percent_change = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            rsi_series = self.rsi_calculator.calculate_rsi(hist_data['Close'])
            rsi = float(rsi_series.iloc[-1])
            
            if rsi > 70:
                rsi_status = "🔴 Overbought"
            elif rsi < 30:
                rsi_status = "🟢 Oversold"
            else:
                rsi_status = "🟡 Neutral"
            
            values = (
                symbol,
                f"${current_price:.2f}",
                f"${price_change:+.2f}",
                f"{percent_change:+.2f}%",
                f"{rsi:.1f}",
                rsi_status,
                stock_info['market_cap_formatted'],
                self.stock_data.format_volume(stock_info['daily_volume']),
                self.stock_data.format_volume(stock_info['avg_volume']),
                f"${stock_info['bid_price']:.2f}" if stock_info['bid_price'] > 0 else "N/A",
                f"${stock_info['ask_price']:.2f}" if stock_info['ask_price'] > 0 else "N/A",
                datetime.now().strftime("%H:%M:%S")
            )
            
            self.root.after(0, lambda: self.update_table_row(symbol, *values))
            
        except Exception as e:
            print(f"Error updating {symbol}: {e}")
            error_values = (symbol, "Error", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error")
            self.root.after(0, lambda: self.update_table_row(symbol, *error_values))
    
    def update_table_row(self, symbol: str, *values):
        """Update table row."""
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
        """Update loop."""
        while self.is_updating:
            if self.watchlist:
                self.root.after(0, lambda: self.status_label.config(text="Updating stocks..."))
                
                for symbol in self.watchlist.copy():
                    if not self.is_updating:
                        break
                    self.update_stock_data(symbol)
                    time.sleep(2)
                
                current_time = datetime.now().strftime("%H:%M:%S")
                self.root.after(0, lambda: self.last_update_label.config(text=f"Last update: {current_time}"))
                self.root.after(0, lambda: self.status_label.config(text="Ready"))
            
            time.sleep(self.update_interval)
    
    def load_watchlist(self):
        """Load watchlist."""
        try:
            if os.path.exists(self.watchlist_file):
                with open(self.watchlist_file, 'r') as f:
                    self.watchlist = json.load(f)
        except Exception as e:
            print(f"Error loading watchlist: {e}")
            self.watchlist = []
    
    def save_watchlist(self):
        """Save watchlist."""
        try:
            with open(self.watchlist_file, 'w') as f:
                json.dump(self.watchlist, f)
        except Exception as e:
            print(f"Error saving watchlist: {e}")
    
    def populate_initial_data(self):
        """Populate initial data."""
        for symbol in self.watchlist:
            loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "", "")
            self.tree.insert('', tk.END, iid=symbol, values=loading_values)
    
    def on_closing(self):
        """Handle closing."""
        self.is_updating = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.populate_initial_data()
        
        if not self.watchlist:
            status_msg = "Add stocks or import portfolio to start tracking"
            if CHARTS_AVAILABLE:
                status_msg += " with charts!"
            else:
                status_msg += "! Install matplotlib for charts."
            self.status_label.config(text=status_msg)
        
        self.root.mainloop()

def main():
    """Main function."""
    try:
        app = EnhancedRSITracker()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()
