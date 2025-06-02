#!/usr/bin/env python3
"""
Enhanced Working RSI Stock Tracker with Market Cap, Bid/Ask, Volume, and Volume Charts
Complete solution with all requested features
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime
import json
import os
import webbrowser
import csv

# Chart imports with fallback
CHARTS_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    plt.style.use('dark_background')
    CHARTS_AVAILABLE = True
except ImportError:
    pass

class RSICalculator:
    @staticmethod
    def calculate_rsi(prices, period=14):
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

class StockData:
    def __init__(self):
        self.cache = {}
        self.info_cache = {}
        self.cache_timeout = 300
        self.info_cache_timeout = 600  # Stock info changes less frequently
    
    def get_stock_data(self, symbol, period="5d"):
        cache_key = f"{symbol}_{period}"
        current_time = time.time()
        
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return data
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                self.cache[cache_key] = (data, current_time)
            
            return data if not data.empty else None
            
        except Exception as e:
            print(f"Error fetching {symbol} ({period}): {e}")
            return None
    
    def get_stock_info(self, symbol):
        """Get detailed stock information including market cap, bid/ask, volume"""
        current_time = time.time()
        
        if symbol in self.info_cache:
            info, timestamp = self.info_cache[symbol]
            if current_time - timestamp < self.info_cache_timeout:
                return info
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info:
                self.info_cache[symbol] = (info, current_time)
            
            return info
            
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def format_market_cap(self, market_cap):
        """Format market cap in readable format"""
        if not market_cap or market_cap == 0:
            return "N/A"
        
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.0f}M"
        else:
            return f"${market_cap:,.0f}"
    
    def format_volume(self, volume):
        """Format volume in readable format"""
        if not volume or volume == 0:
            return "0"
        
        if volume >= 1e9:
            return f"{volume/1e9:.2f}B"
        elif volume >= 1e6:
            return f"{volume/1e6:.1f}M"
        elif volume >= 1e3:
            return f"{volume/1e3:.0f}K"
        else:
            return f"{volume:,.0f}"
    
    def clear_cache(self):
        self.cache.clear()
        self.info_cache.clear()

class EnhancedRSITracker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced RSI Tracker - Market Cap, Volume & Bid/Ask")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1e1e1e')
        
        # Data handlers
        self.stock_data = StockData()
        self.rsi_calculator = RSICalculator()
        
        # Watchlist
        self.watchlist = []
        self.watchlist_file = "watchlist.json"
        self.load_watchlist()
        
        # Update control
        self.update_interval = 30
        self.is_updating = False
        self.update_thread = None
        
        # Chart variables
        self.selected_symbol = None
        self.current_canvas = None
        
        # Setup
        self.setup_styles()
        self.setup_ui()
        self.start_updates()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
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
        style.configure('TNotebook.Tab', background='#404040', foreground='white', padding=[15, 8])
        style.map('TNotebook.Tab', background=[('selected', '#00ff88')], foreground=[('selected', 'black')])
    
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        chart_status = "✅ Charts & Volume Working" if CHARTS_AVAILABLE else "⚠️ Charts Disabled"
        title_label = tk.Label(main_frame, 
                              text=f"📈 Enhanced RSI Tracker - Market Cap | Volume | Bid/Ask - {chart_status}", 
                              font=('Arial', 16, 'bold'),
                              bg='#1e1e1e', fg='#00ff88')
        title_label.pack(pady=(0, 10))
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Setup tabs
        self.setup_tracker_tab()
        if CHARTS_AVAILABLE:
            self.setup_charts_tab()
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def setup_tracker_tab(self):
        tracker_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(tracker_frame, text="📊 Enhanced Stock Tracker")
        
        # Controls
        control_frame = tk.Frame(tracker_frame, bg='#1e1e1e')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add stock
        add_frame = tk.Frame(control_frame, bg='#1e1e1e')
        add_frame.pack(side=tk.LEFT)
        
        tk.Label(add_frame, text="Add Stock:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.symbol_entry = tk.Entry(add_frame, width=10, font=('Arial', 10))
        self.symbol_entry.pack(side=tk.LEFT, padx=(5, 5))
        self.symbol_entry.bind('<Return>', self.add_stock_event)
        
        tk.Button(add_frame, text="Add", command=self.add_stock,
                 bg='#00ff88', fg='black', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        # Import
        import_frame = tk.Frame(control_frame, bg='#1e1e1e')
        import_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        tk.Button(import_frame, text="📁 Import Portfolio", command=self.import_portfolio,
                 bg='#8844ff', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        # Controls
        button_frame = tk.Frame(control_frame, bg='#1e1e1e')
        button_frame.pack(side=tk.RIGHT)
        
        if CHARTS_AVAILABLE:
            tk.Button(button_frame, text="📈 View Chart + Volume", command=self.view_chart,
                     bg='#ff8844', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Remove Selected", command=self.remove_stock,
                 bg='#ff4444', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Seeking Alpha", command=self.open_seeking_alpha,
                 bg='#ffaa00', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🔄 Refresh All", command=self.manual_refresh,
                 bg='#4488ff', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Status
        status_frame = tk.Frame(tracker_frame, bg='#1e1e1e')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Ready", 
                                    bg='#1e1e1e', fg='#cccccc', font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT)
        
        self.last_update_label = tk.Label(status_frame, text="", 
                                         bg='#1e1e1e', fg='#888888', font=('Arial', 9))
        self.last_update_label.pack(side=tk.RIGHT)
        
        # Enhanced Table with new columns
        self.setup_enhanced_table(tracker_frame)
        
        # Enhanced Legend
        legend_frame = tk.Frame(tracker_frame, bg='#1e1e1e')
        legend_frame.pack(pady=10)
        
        tk.Label(legend_frame, text="Enhanced Data Legend:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold')).pack()
        
        chart_info = "Select stock + View Chart for Price/RSI/Volume analysis" if CHARTS_AVAILABLE else "Charts disabled"
        legend_text = f"🔴 Overbought (>70)  🟡 Neutral (30-70)  🟢 Oversold (<30)  💰 Market Cap  📊 Volume  💱 Bid/Ask\n{chart_info}"
        tk.Label(legend_frame, text=legend_text,
                bg='#1e1e1e', fg='#cccccc', font=('Arial', 9)).pack()
    
    def setup_charts_tab(self):
        charts_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(charts_frame, text="📈 Price/RSI/Volume Charts")
        
        # Chart controls
        control_frame = tk.Frame(charts_frame, bg='#1e1e1e')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(control_frame, text="Symbol:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.chart_symbol_var = tk.StringVar()
        self.chart_symbol_combo = ttk.Combobox(control_frame, textvariable=self.chart_symbol_var, 
                                              width=8, font=('Arial', 10))
        self.chart_symbol_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        tk.Label(control_frame, text="Period:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.chart_period_var = tk.StringVar(value="3mo")
        self.period_combo = ttk.Combobox(control_frame, textvariable=self.chart_period_var, 
                                        values=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"], 
                                        width=8, font=('Arial', 10))
        self.period_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Auto-refresh on period change
        self.period_combo.bind('<<ComboboxSelected>>', self.on_period_changed)
        
        tk.Button(control_frame, text="📊 Load Price/RSI/Volume Chart", command=self.load_enhanced_chart,
                 bg='#00ff88', fg='black', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="🔄 Clear Cache", command=self.clear_cache_and_reload,
                 bg='#ff4488', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Chart area
        self.chart_frame = tk.Frame(charts_frame, bg='#1e1e1e')
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chart_message = tk.Label(self.chart_frame, 
                                     text="Select stock and period, then click Load Chart\nNow with Volume Analysis and Market Data!",
                                     bg='#1e1e1e', fg='#cccccc', font=('Arial', 12))
        self.chart_message.pack(expand=True)
    
    def on_period_changed(self, event=None):
        """Auto-reload when period changes"""
        if self.chart_symbol_var.get():
            print(f"Period changed to: {self.chart_period_var.get()}")
            self.status_label.config(text=f"Auto-refreshing enhanced chart for {self.chart_period_var.get()}...")
            self.root.after(200, self.load_enhanced_chart)
    
    def clear_cache_and_reload(self):
        self.stock_data.clear_cache()
        self.status_label.config(text="Cache cleared, reloading enhanced data...")
        self.load_enhanced_chart()
    
    def setup_enhanced_table(self, parent):
        table_frame = tk.Frame(parent, bg='#1e1e1e')
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Enhanced columns with Market Cap, Volume, Bid/Ask
        columns = ('Symbol', 'Price', 'Change', 'Change%', 'RSI', 'Status', 'MarketCap', 'Volume', 'Bid', 'Ask', 'Updated')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        headings = {
            'Symbol': ('Stock', 70),
            'Price': ('Price ($)', 80),
            'Change': ('Change ($)', 80), 
            'Change%': ('Change %', 80),
            'RSI': ('RSI', 60),
            'Status': ('RSI Status', 100),
            'MarketCap': ('Market Cap', 90),
            'Volume': ('Volume', 80),
            'Bid': ('Bid', 70),
            'Ask': ('Ask', 70),
            'Updated': ('Updated', 70)
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
    
    def on_stock_select(self, event):
        selected = self.tree.selection()
        if selected:
            self.selected_symbol = selected[0]
            if CHARTS_AVAILABLE:
                self.chart_symbol_var.set(self.selected_symbol)
                self.chart_symbol_combo['values'] = self.watchlist
    
    def on_tab_changed(self, event):
        if CHARTS_AVAILABLE:
            self.chart_symbol_combo['values'] = self.watchlist
            if self.selected_symbol:
                self.chart_symbol_var.set(self.selected_symbol)
    
    def view_chart(self):
        if not CHARTS_AVAILABLE:
            messagebox.showwarning("Charts Not Available", "Install matplotlib: pip3 install matplotlib")
            return
            
        if not self.selected_symbol:
            messagebox.showwarning("No Selection", "Please select a stock to view its enhanced chart")
            return
        
        self.notebook.select(1)  # Charts tab
        self.chart_symbol_var.set(self.selected_symbol)
        self.load_enhanced_chart()
    
    def load_enhanced_chart(self):
        if not CHARTS_AVAILABLE:
            return
            
        symbol = self.chart_symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showwarning("No Symbol", "Please select a stock symbol")
            return
        
        period = self.chart_period_var.get()
        
        # Clear previous chart
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Show loading
        loading_label = tk.Label(self.chart_frame, 
                               text=f"Loading enhanced {symbol} chart with volume for {period}...",
                               bg='#1e1e1e', fg='#cccccc', font=('Arial', 12))
        loading_label.pack(expand=True)
        self.root.update()
        
        try:
            # Fetch data and info
            data = self.stock_data.get_stock_data(symbol, period)
            stock_info = self.stock_data.get_stock_info(symbol)
            
            if data is None or data.empty:
                loading_label.destroy()
                tk.Label(self.chart_frame, 
                        text=f"No data available for {symbol} ({period})",
                        bg='#1e1e1e', fg='#ff4444', font=('Arial', 12)).pack(expand=True)
                return
            
            # Calculate RSI
            rsi_data = self.rsi_calculator.calculate_rsi(data['Close'])
            current_rsi = float(rsi_data.iloc[-1])
            
            # Get stock info for title
            market_cap = stock_info.get('marketCap', 0)
            formatted_market_cap = self.stock_data.format_market_cap(market_cap)
            company_name = stock_info.get('longName', symbol)
            
            # Create enhanced chart with 3 subplots: Price, RSI, Volume
            fig = Figure(figsize=(14, 10), facecolor='#1e1e1e')
            
            # Price chart (top)
            ax1 = fig.add_subplot(311, facecolor='#2d2d2d')
            ax1.plot(data.index, data['Close'], color='#00ff88', linewidth=2, label='Price')
            
            # Add moving averages if enough data
            if len(data) >= 20:
                ma20 = data['Close'].rolling(20).mean()
                ax1.plot(data.index, ma20, color='#ffaa00', linewidth=1, alpha=0.8, label='MA20')
            
            if len(data) >= 50:
                ma50 = data['Close'].rolling(50).mean()
                ax1.plot(data.index, ma50, color='#ff4488', linewidth=1, alpha=0.8, label='MA50')
            
            ax1.set_title(f'{company_name} ({symbol}) - Market Cap: {formatted_market_cap} | Period: {period} | {len(data)} points', 
                         color='white', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', color='white')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='white')
            
            # RSI chart (middle)
            ax2 = fig.add_subplot(312, facecolor='#2d2d2d')
            ax2.plot(data.index, rsi_data, color='#4488ff', linewidth=2, label='RSI')
            
            # RSI levels
            ax2.axhline(y=70, color='#ff4444', linestyle='--', alpha=0.8, label='Overbought (70)')
            ax2.axhline(y=30, color='#44ff44', linestyle='--', alpha=0.8, label='Oversold (30)')
            ax2.axhline(y=50, color='#888888', linestyle='-', alpha=0.6, label='Neutral (50)')
            
            # Fill zones
            ax2.fill_between(data.index, 70, 100, alpha=0.1, color='red')
            ax2.fill_between(data.index, 0, 30, alpha=0.1, color='green')
            
            ax2.set_title(f'RSI (Current: {current_rsi:.1f})', color='white', fontsize=12)
            ax2.set_ylabel('RSI', color='white')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors='white')
            
            # Volume chart (bottom)
            ax3 = fig.add_subplot(313, facecolor='#2d2d2d')
            
            # Color volume bars based on price movement
            colors = []
            for i in range(len(data)):
                if i == 0:
                    colors.append('#888888')  # Neutral for first bar
                else:
                    if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                        colors.append('#00ff88')  # Green for up days
                    else:
                        colors.append('#ff4444')  # Red for down days
            
            ax3.bar(data.index, data['Volume'], color=colors, alpha=0.7, width=0.8)
            
            # Add volume moving average
            if len(data) >= 20:
                vol_ma = data['Volume'].rolling(20).mean()
                ax3.plot(data.index, vol_ma, color='#ffaa00', linewidth=2, alpha=0.8, label='Volume MA20')
                ax3.legend(loc='upper left')
            
            # Format volume y-axis
            avg_volume = data['Volume'].mean()
            max_volume = data['Volume'].max()
            
            ax3.set_title(f'Volume - Avg: {self.stock_data.format_volume(avg_volume)} | Max: {self.stock_data.format_volume(max_volume)}', 
                         color='white', fontsize=12)
            ax3.set_ylabel('Volume', color='white')
            ax3.set_xlabel('Date', color='white')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(colors='white')
            
            # Format dates on all subplots
            for ax in [ax1, ax2, ax3]:
                if period in ['1d', '5d']:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                elif period in ['1mo']:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
            
            fig.autofmt_xdate(rotation=45)
            fig.tight_layout()
            
            # Remove loading message
            loading_label.destroy()
            
            # Create canvas
            self.current_canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update status with enhanced info
            rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            current_volume = data['Volume'].iloc[-1]
            volume_vs_avg = "Above Avg" if current_volume > avg_volume else "Below Avg"
            
            self.status_label.config(text=f"✅ Enhanced Chart: {symbol} ({period}) - RSI: {current_rsi:.1f} ({rsi_status}) | Volume: {self.stock_data.format_volume(current_volume)} ({volume_vs_avg})")
            
            print(f"✅ Enhanced chart loaded: {symbol} ({period}) - {len(data)} points - RSI: {current_rsi:.1f} - Volume: {self.stock_data.format_volume(current_volume)}")
            
        except Exception as e:
            loading_label.destroy()
            tk.Label(self.chart_frame, 
                    text=f"Error creating enhanced chart: {str(e)[:100]}",
                    bg='#1e1e1e', fg='#ff4444', font=('Arial', 11)).pack(expand=True)
            print(f"❌ Enhanced chart error: {e}")
    
    def import_portfolio(self):
        file_path = filedialog.askopenfilename(
            title="Import Portfolio",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            symbols = []
            if file_path.lower().endswith('.csv'):
                with open(file_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        for key, value in row.items():
                            if 'symbol' in key.lower() or 'ticker' in key.lower():
                                symbol = value.strip().upper()
                                if symbol and len(symbol) <= 5:
                                    symbols.append(symbol)
                                break
            else:
                with open(file_path, 'r') as file:
                    for line in file:
                        symbol = line.strip().upper()
                        if symbol and len(symbol) <= 5:
                            symbols.append(symbol)
            
            symbols = list(set(symbols))  # Remove duplicates
            
            if not symbols:
                messagebox.showerror("Import Error", "No valid symbols found")
                return
            
            added_count = 0
            for symbol in symbols:
                if symbol not in self.watchlist:
                    self.watchlist.append(symbol)
                    loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "")
                    self.tree.insert('', tk.END, iid=symbol, values=loading_values)
                    added_count += 1
            
            if added_count > 0:
                self.save_watchlist()
                threading.Thread(target=self.refresh_new_stocks, args=(symbols,), daemon=True).start()
                
            messagebox.showinfo("Import Complete", 
                              f"Imported {added_count} new stocks.\nSkipped {len(symbols) - added_count} duplicates.")
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Error importing file: {e}")
    
    def refresh_new_stocks(self, symbols):
        for symbol in symbols:
            if symbol in self.watchlist:
                self.update_enhanced_stock_data(symbol)
                time.sleep(2)  # Longer delay for enhanced data
    
    def add_stock_event(self, event):
        self.add_stock()
    
    def add_stock(self):
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            return
        
        if symbol in self.watchlist:
            messagebox.showwarning("Duplicate", f"{symbol} is already in watchlist!")
            return
        
        self.status_label.config(text=f"Validating {symbol} and fetching enhanced data...")
        self.root.update()
        
        data = self.stock_data.get_stock_data(symbol, "5d")
        if data is None or data.empty:
            messagebox.showerror("Invalid Symbol", f"Could not find data for {symbol}")
            self.status_label.config(text="Ready")
            return
        
        self.watchlist.append(symbol)
        self.symbol_entry.delete(0, tk.END)
        self.save_watchlist()
        
        loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "")
        self.tree.insert('', tk.END, iid=symbol, values=loading_values)
        
        threading.Thread(target=self.update_enhanced_stock_data, args=(symbol,), daemon=True).start()
        self.status_label.config(text=f"Added {symbol} - fetching enhanced data...")
    
    def remove_stock(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock to remove")
            return
        
        symbol = selected[0]
        if messagebox.askyesno("Confirm", f"Remove {symbol}?"):
            self.watchlist.remove(symbol)
            self.tree.delete(symbol)
            self.save_watchlist()
            self.status_label.config(text=f"Removed {symbol}")
    
    def open_seeking_alpha(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock")
            return
        
        symbol = selected[0]
        webbrowser.open(f"https://seekingalpha.com/symbol/{symbol}")
        self.status_label.config(text=f"Opened Seeking Alpha for {symbol}")
    
    def on_double_click(self, event):
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            webbrowser.open(f"https://seekingalpha.com/symbol/{item}")
    
    def manual_refresh(self):
        if not self.watchlist:
            messagebox.showinfo("Empty", "Add stocks first!")
            return
        
        self.status_label.config(text="Refreshing all enhanced data...")
        threading.Thread(target=self.refresh_all_enhanced_stocks, daemon=True).start()
    
    def refresh_all_enhanced_stocks(self):
        for symbol in self.watchlist:
            self.root.after(0, lambda s=symbol: self.status_label.config(text=f"Updating enhanced data for {s}..."))
            self.update_enhanced_stock_data(symbol)
            time.sleep(2)  # Longer delay for enhanced data
        
        self.root.after(0, lambda: self.status_label.config(text="All enhanced data updated"))
    
    def update_enhanced_stock_data(self, symbol):
        """Enhanced data update with Market Cap, Volume, Bid/Ask"""
        hist_data = self.stock_data.get_stock_data(symbol, "1mo")
        stock_info = self.stock_data.get_stock_info(symbol)
        
        if hist_data is None or hist_data.empty:
            error_values = (symbol, "Error", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error")
            self.root.after(0, lambda: self.update_table_row(symbol, *error_values))
            return
        
        try:
            # Basic price data
            current_price = float(hist_data['Close'].iloc[-1])
            previous_price = float(hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else current_price
            price_change = current_price - previous_price
            percent_change = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            # RSI calculation
            rsi_series = self.rsi_calculator.calculate_rsi(hist_data['Close'])
            rsi = float(rsi_series.iloc[-1])
            
            # RSI status
            if rsi > 70:
                rsi_status = "🔴 Overbought"
            elif rsi < 30:
                rsi_status = "🟢 Oversold"
            else:
                rsi_status = "🟡 Neutral"
            
            # Enhanced data from stock info
            market_cap = stock_info.get('marketCap', 0)
            formatted_market_cap = self.stock_data.format_market_cap(market_cap)
            
            # Volume data
            current_volume = hist_data['Volume'].iloc[-1] if 'Volume' in hist_data.columns else 0
            formatted_volume = self.stock_data.format_volume(current_volume)
            
            # Bid/Ask data
            bid_price = stock_info.get('bid', 0)
            ask_price = stock_info.get('ask', 0)
            
            bid_str = f"${bid_price:.2f}" if bid_price > 0 else "N/A"
            ask_str = f"${ask_price:.2f}" if ask_price > 0 else "N/A"
            
            # Enhanced values tuple
            values = (
                symbol,
                f"${current_price:.2f}",
                f"${price_change:+.2f}",
                f"{percent_change:+.2f}%",
                f"{rsi:.1f}",
                rsi_status,
                formatted_market_cap,
                formatted_volume,
                bid_str,
                ask_str,
                datetime.now().strftime("%H:%M:%S")
            )
            
            self.root.after(0, lambda: self.update_table_row(symbol, *values))
            
        except Exception as e:
            print(f"Error updating enhanced data for {symbol}: {e}")
            error_values = (symbol, "Error", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error")
            self.root.after(0, lambda: self.update_table_row(symbol, *error_values))
    
    def update_table_row(self, symbol, *values):
        try:
            self.tree.item(symbol, values=values)
        except tk.TclError:
            pass
    
    def start_updates(self):
        if not self.is_updating:
            self.is_updating = True
            self.update_thread = threading.Thread(target=self.enhanced_update_loop, daemon=True)
            self.update_thread.start()
    
    def enhanced_update_loop(self):
        while self.is_updating:
            if self.watchlist:
                self.root.after(0, lambda: self.status_label.config(text="Auto-updating enhanced data..."))
                
                for symbol in self.watchlist.copy():
                    if not self.is_updating:
                        break
                    self.update_enhanced_stock_data(symbol)
                    time.sleep(3)  # Longer delay for enhanced data
                
                current_time = datetime.now().strftime("%H:%M:%S")
                self.root.after(0, lambda: self.last_update_label.config(text=f"Last: {current_time}"))
                self.root.after(0, lambda: self.status_label.config(text="Enhanced data ready"))
            
            time.sleep(self.update_interval)
    
    def load_watchlist(self):
        try:
            if os.path.exists(self.watchlist_file):
                with open(self.watchlist_file, 'r') as f:
                    self.watchlist = json.load(f)
        except:
            self.watchlist = []
    
    def save_watchlist(self):
        try:
            with open(self.watchlist_file, 'w') as f:
                json.dump(self.watchlist, f)
        except:
            pass
    
    def populate_initial_data(self):
        for symbol in self.watchlist:
            loading_values = (symbol, "Loading...", "", "", "", "", "", "", "", "", "")
            self.tree.insert('', tk.END, iid=symbol, values=loading_values)
    
    def on_closing(self):
        self.is_updating = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=3)
        self.root.destroy()
    
    def run(self):
        self.populate_initial_data()
        
        if not self.watchlist:
            if CHARTS_AVAILABLE:
                self.status_label.config(text="Add stocks to start enhanced tracking with Market Cap, Volume & Bid/Ask!")
            else:
                self.status_label.config(text="Add stocks for enhanced tracking (install matplotlib for volume charts)")
        
        print("🚀 Starting Enhanced RSI Tracker...")
        print("✅ Enhanced features: Market Cap, Volume, Bid/Ask data")
        print("✅ Timeframes auto-refresh when changed!")
        if CHARTS_AVAILABLE:
            print("✅ Volume charts are working!")
        else:
            print("⚠️  Charts disabled - run: pip3 install matplotlib")
        
        self.root.mainloop()

def main():
    try:
        app = EnhancedRSITracker()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", f"Failed to start enhanced tracker: {e}")

if __name__ == "__main__":
    main()
