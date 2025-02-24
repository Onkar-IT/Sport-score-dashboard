import math
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, simpledialog, colorchooser, Tk, Canvas, Frame, BOTH, LEFT, RIGHT, Y, X, NW
import tkinter.font as tkFont
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import IsolationForest
import sqlite3
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


# Helper: Create a scrollable frame
def create_scrollable_frame(parent, bg=None):
    if bg is None:
        try:
            bg = parent.cget("background")
        except Exception:
            bg = "#f7f7f7"
    canvas = Canvas(parent, borderwidth=0, background=bg)
    frame = Frame(canvas, background=bg)
    vsb = tb.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor=NW)
    return frame


class DataVizApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sport Scope Dashboard")
        self.master.geometry("1700x950")

        # ------------------ Theme Definitions ------------------
        self.light_theme = {
            "bg": "#f0f0f0",
            "text": "#333333",
            "button": "#5bc0de",
            "chart_title": "#1a1a1a",
            "axis_label": "#1a1a1a"
        }
        self.dark_theme = {
            "bg": "#2b2b2b",
            "text": "#ffffff",
            "button": "#6c757d",
            "chart_title": "#ffffff",
            "axis_label": "#ffffff"
        }
        self.theme_color = self.light_theme["bg"]
        self.text_color = self.light_theme["text"]
        self.button_color = self.light_theme["button"]
        self.chart_title_color = self.light_theme["chart_title"]
        self.axis_label_color = self.light_theme["axis_label"]

        # ------------------ Defaults and Variables ------------------
        self.default_color_palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

        self.base_font_family = "Calibri"
        self.base_font_size = 12
        self.available_text_families = ["Arial", "Times New Roman", "Courier New", "Calibri", "Helvetica", "Verdana"]

        # UI Font settings
        self.font_family_var = tb.StringVar(value=self.base_font_family)
        self.font_size_var = tb.StringVar(value=str(self.base_font_size))
        self.chart_title_font_family = tb.StringVar(value="Calibri")
        self.chart_title_size = tb.StringVar(value="20")
        self.chart_title_bold = tb.BooleanVar(value=False)
        self.chart_title_italic = tb.BooleanVar(value=False)
        self.chart_title_underline = tb.BooleanVar(value=False)

        # Forecasting options
        self.prediction_var = tb.BooleanVar(value=False)
        self.forecast_model_var = tb.StringVar(value="Linear")
        self.forecast_horizon_var = tb.StringVar(value="5")
        self.conf_int_var = tb.BooleanVar(value=False)
        self.forecast_x = None
        self.forecast_y = None
        self.forecast_ci = None

        # Custom chart creator (for File & Data tab)
        self.custom_chart_type_var = tb.StringVar(value="Scatter")
        self.custom_chart_title_var = tb.StringVar()
        self.custom_chart_color_var = tb.StringVar()

        # Data and anomalies
        self.file_path = None
        self.data = None
        self.anomalies = None

        # Dashboard chart configuration storage
        self.dashboard_chart_configs = []

        # ------------------ Style Setup ------------------
        self.style = tb.Style()
        self.style.theme_use("flatly")
        self._update_style_fonts()
        self.update_colors()

        # ------------------ Layout Setup ------------------
        # Header
        self.header = tb.Frame(self.master, padding=10, bootstyle=INFO)
        self.header.pack(fill=X)
        header_label = tb.Label(self.header, text="Sport Scope Dashboard",
                                font=(self.base_font_family, 28, "bold"), bootstyle="inverse")
        header_label.pack(side=LEFT)
        self.theme_toggle_btn = tb.Button(self.header, text="Toggle Dark/Light",
                                          command=self.toggle_dark_light, bootstyle=SECONDARY)
        self.theme_toggle_btn.pack(side=RIGHT)

        # Main container: Three panes (Left Navigation, Center Content, Right Suggestions)
        self.main_pane = tb.Frame(self.master)
        self.main_pane.pack(fill=BOTH, expand=True, padx=10, pady=10)
        self.main_pane.columnconfigure(1, weight=1)
        self.main_pane.rowconfigure(0, weight=1)

        # Left Navigation Sidebar
        self.nav_frame = tb.Frame(self.main_pane, width=200, bootstyle=DARK)
        self.nav_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        self._build_nav_buttons()

        # Center Content Area (Pages)
        self.content_frame = tb.Frame(self.main_pane)
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame.rowconfigure(0, weight=1)
        self.content_frame.columnconfigure(0, weight=1)
        self.pages = {}
        self._build_pages()

        # Right Suggestions Panel (visible only on "File & Data")
        self.suggestions_frame = tb.Frame(self.main_pane, width=250)
        self.suggestions_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        self._build_suggestions_panel()

        # Status Bar
        self.status_bar = tb.Label(self.master, text="Welcome to Sport Scope Dashboard!", anchor="w", padding=5,
                                   bootstyle="secondary")
        self.status_bar.pack(fill=X, side="bottom")

        # Show default page
        self.show_page("File & Data")

    # ------------------ Style Methods ------------------
    def _update_style_fonts(self):
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family=self.font_family_var.get(), size=int(self.font_size_var.get()))
        self.style.configure("TButton", font=(self.font_family_var.get(), int(self.font_size_var.get())))
        self.style.configure("TLabel", font=(self.font_family_var.get(), int(self.font_size_var.get())))

    def update_colors(self):
        self.style.configure("TFrame", background=self.theme_color)
        self.style.configure("TLabelframe", background=self.theme_color, foreground=self.text_color)
        self.style.configure("TLabelframe.Label", background=self.theme_color, foreground=self.text_color)
        self.style.configure("TLabel", background=self.theme_color, foreground=self.text_color)
        self.style.configure("TButton", background=self.button_color, foreground=self.text_color)
        self.master.configure(background=self.theme_color)

    # ------------------ Toggle Dark/Light Mode ------------------
    def toggle_dark_light(self):
        if self.master.style.theme_use() == "flatly":
            self.master.style.theme_use("cyborg")
            self.theme_color = self.dark_theme["bg"]
            self.text_color = self.dark_theme["text"]
            self.button_color = self.dark_theme["button"]
            self.chart_title_color = self.dark_theme["chart_title"]
            self.axis_label_color = self.dark_theme["axis_label"]
            self.update_status("Switched to Dark Mode.")
        else:
            self.master.style.theme_use("flatly")
            self.theme_color = self.light_theme["bg"]
            self.text_color = self.light_theme["text"]
            self.button_color = self.light_theme["button"]
            self.chart_title_color = self.light_theme["chart_title"]
            self.axis_label_color = self.light_theme["axis_label"]
            self.update_status("Switched to Light Mode.")
        self.apply_settings()

    def update_status(self, message, error=False):
        self.status_bar.config(text=message, bootstyle="danger" if error else "secondary")

    def get_column_name(self, col):
        if self.data is None:
            return col
        if col in self.data.columns:
            return col
        col_lower = col.lower()
        for c in self.data.columns:
            if c.lower() == col_lower:
                return c
        return col

    def get_default_color_palette(self):
        if hasattr(self, "color_palette_entry"):
            cp = self.color_palette_entry.get().strip()
            if cp:
                return [c.strip() for c in cp.split(",") if c.strip()]
        return self.default_color_palette

    # ------------------ Navigation ------------------
    def _build_nav_buttons(self):
        btn_specs = [
            ("File & Data", lambda: self.show_page("File & Data")),
            ("File Converter", lambda: self.show_page("File Converter")),
            ("Forecasting", lambda: self.show_page("Forecasting")),
            ("Custom Dashboard", lambda: self.show_page("Custom Dashboard")),
            ("Settings", lambda: self.show_page("Settings")),
        ]
        for text, cmd in btn_specs:
            btn = tb.Button(self.nav_frame, text=text, command=cmd, width=20, bootstyle=PRIMARY)
            btn.pack(pady=8, padx=10, fill=X)

    def show_page(self, page_name):
        page = self.pages.get(page_name)
        if page:
            for p in self.pages.values():
                p.grid_remove()
            page.grid(sticky="nsew")
            page.tkraise()
            self.update_status(f"Switched to {page_name} page.")
        if page_name == "File & Data":
            self.suggestions_frame.grid()
        else:
            self.suggestions_frame.grid_remove()

    # ------------------ Pages Building ------------------
    def _build_pages(self):
        for page_name in ["File & Data", "File Converter", "Forecasting", "Custom Dashboard", "Settings"]:
            frame = tb.Frame(self.content_frame)
            frame.grid(row=0, column=0, sticky="nsew")
            self.pages[page_name] = frame

        self._build_file_data_page(self.pages["File & Data"])
        self._build_converter_page(self.pages["File Converter"])
        self._build_forecasting_page(self.pages["Forecasting"])
        self._build_dashboard_page(self.pages["Custom Dashboard"])
        self._build_settings_page(self.pages["Settings"])

    # ------------------ Page: File & Data ------------------
    def _build_file_data_page(self, parent):
        file_frame = tb.Labelframe(parent, text="File Upload", padding=10, bootstyle=INFO)
        file_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(file_frame, text="Upload your file:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(file_frame, text="Browse File", command=self.upload_file, bootstyle=SUCCESS).grid(row=0, column=1,
                                                                                                    padx=5, pady=5)
        col_frame = tb.Labelframe(parent, text="Column Selection", padding=10, bootstyle=INFO)
        col_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(col_frame, text="X-Axis Column:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.x_col_menu = tb.Combobox(col_frame, state="readonly")
        self.x_col_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(col_frame, text="Y-Axis Column:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.y_col_menu = tb.Combobox(col_frame, state="readonly")
        self.y_col_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.z_col_label = tb.Label(col_frame, text="Z-Axis Column (3D):")
        self.z_col_menu = tb.Combobox(col_frame, state="readonly")
        self.z_col_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.z_col_menu.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.z_col_label.grid_remove()
        self.z_col_menu.grid_remove()
        custom_frame = tb.Labelframe(parent, text="Custom Chart Creator", padding=10, bootstyle=INFO)
        custom_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(custom_frame, text="Chart Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.custom_chart_type_cb = tb.Combobox(custom_frame, state="readonly", textvariable=self.custom_chart_type_var,
                                                values=["Scatter", "Line", "Bar", "Area", "Bubble", "Pie", "Other"])
        self.custom_chart_type_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(custom_frame, text="Chart Title:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.custom_chart_title_entry = tb.Entry(custom_frame, textvariable=self.custom_chart_title_var)
        self.custom_chart_title_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(custom_frame, text="Custom Color (e.g., 'blue' or '#ff0000'):").grid(row=2, column=0, padx=5, pady=5,
                                                                                      sticky="w")
        self.custom_chart_color_entry = tb.Entry(custom_frame, textvariable=self.custom_chart_color_var)
        self.custom_chart_color_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(custom_frame, text="Generate Custom Chart", command=self.custom_chart, bootstyle=PRIMARY).grid(row=3,
                                                                                                                 column=0,
                                                                                                                 columnspan=2,
                                                                                                                 padx=5,
                                                                                                                 pady=10,
                                                                                                                 sticky="ew")
        custom_frame.columnconfigure(1, weight=1)

    # ------------------ Page: File Converter ------------------
    def _build_converter_page(self, parent):
        conv_frame = tb.Labelframe(parent, text="File Converter", padding=10, bootstyle=INFO)
        conv_frame.pack(fill="both", expand=True, padx=10, pady=5)
        tb.Label(conv_frame, text="Upload file to convert:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(conv_frame, text="Browse File", command=self.upload_file, bootstyle=SUCCESS).grid(row=0, column=1,
                                                                                                    padx=5, pady=5,
                                                                                                    sticky="w")
        tb.Label(conv_frame, text="Output Format:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_format_cb = tb.Combobox(conv_frame, state="readonly", values=["CSV", "Excel", "SQLite"])
        self.output_format_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(conv_frame, text="Convert File", command=self.convert_file, bootstyle=PRIMARY).grid(row=2, column=0,
                                                                                                      columnspan=2,
                                                                                                      padx=5, pady=10,
                                                                                                      sticky="ew")
        conv_frame.columnconfigure(1, weight=1)

    # ------------------ Page: Forecasting ------------------
    def _build_forecasting_page(self, parent):
        fc_frame = tb.Labelframe(parent, text="Forecasting Options", padding=10, bootstyle=INFO)
        fc_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.prediction_toggle = tb.Checkbutton(fc_frame, text="Enable Prediction", variable=self.prediction_var,
                                                command=self.toggle_prediction)
        self.prediction_toggle.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Label(fc_frame, text="Chart Type:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.chart_menu = tb.Combobox(fc_frame, state="readonly",
                                      values=["Scatter", "Line", "Bar", "Pie", "Area", "Bubble",
                                              "Waterfall", "Histogram", "Funnel", "Gantt", "Donut", "Radar",
                                              "Treemap", "Box Plot", "Clustered Bar", "Flowchart", "Heatmap",
                                              "Bullet Graph", "3D Scatter", "3D Surface", "3D Line", "3D Bubble",
                                              "Venn Diagram"])
        self.chart_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.chart_menu.bind("<<ComboboxSelected>>", self.update_z_axis_visibility)
        tb.Label(fc_frame, text="Forecast Model:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.forecast_model_cb = tb.Combobox(fc_frame, state="readonly", textvariable=self.forecast_model_var,
                                             values=["Linear", "Polynomial", "ARIMA"])
        self.forecast_model_cb.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(fc_frame, text="Forecast Horizon:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.forecast_horizon_entry = tb.Entry(fc_frame, textvariable=self.forecast_horizon_var)
        self.forecast_horizon_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.conf_int_cb = tb.Checkbutton(fc_frame, text="Show Confidence Interval", variable=self.conf_int_var)
        self.conf_int_cb.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        tb.Button(fc_frame, text="Visualize Forecast", command=self.generate_chart, bootstyle=PRIMARY).grid(row=5,
                                                                                                            column=0,
                                                                                                            columnspan=2,
                                                                                                            padx=5,
                                                                                                            pady=10,
                                                                                                            sticky="ew")
        fc_frame.columnconfigure(1, weight=1)

    # ------------------ Page: Custom Dashboard ------------------
    def _build_dashboard_page(self, parent):
        frame = tb.Frame(parent)
        frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        instructions = tb.Label(frame, text="Custom Dashboard Setup", font=(self.base_font_family, 16, "bold"))
        instructions.pack(pady=5)

        # Ask for number of charts
        count_frame = tb.Frame(frame)
        count_frame.pack(fill=X, pady=5)
        tb.Label(count_frame, text="How many charts do you need?").pack(side=LEFT, padx=5)
        self.chart_count_var = tb.StringVar(value="1")
        self.chart_count_entry = tb.Entry(count_frame, textvariable=self.chart_count_var, width=5)
        self.chart_count_entry.pack(side=LEFT, padx=5)
        tb.Button(count_frame, text="Generate Chart Options", command=self.generate_chart_options,
                  bootstyle=PRIMARY).pack(side=LEFT, padx=5)

        # Container frame for dynamic chart options – arranged in a grid.
        self.chart_options_frame = tb.Frame(frame)
        self.chart_options_frame.pack(fill=BOTH, expand=True, pady=10)

        # Create Dashboard button
        tb.Button(frame, text="Create Dashboard", command=self.create_dashboard, bootstyle=SUCCESS).pack(pady=10)

    def generate_chart_options(self):
        # Clear previous options
        for widget in self.chart_options_frame.winfo_children():
            widget.destroy()
        try:
            count = int(self.chart_count_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for chart count.")
            return

        self.dashboard_chart_configs = []
        cols = int(math.ceil(math.sqrt(count)))
        for col in range(cols):
            self.chart_options_frame.grid_columnconfigure(col, weight=1)

        for i in range(count):
            r = i // cols
            c = i % cols
            subframe = tb.Labelframe(self.chart_options_frame, text=f"Chart {i + 1} Options", padding=10,
                                     bootstyle=INFO)
            subframe.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            tb.Label(subframe, text="Chart Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            chart_type_var = tb.StringVar(value="Scatter")
            chart_type_cb = tb.Combobox(subframe, state="readonly", textvariable=chart_type_var,
                                        values=["Scatter", "Line", "Bar", "Pie", "Area", "Bubble", "Histogram"])
            chart_type_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            tb.Label(subframe, text="X Column:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            x_col_var = tb.StringVar()
            x_col_cb = tb.Combobox(subframe, state="readonly", textvariable=x_col_var,
                                   values=list(self.data.columns) if self.data is not None else [])
            x_col_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            tb.Label(subframe, text="Y Column:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
            y_col_var = tb.StringVar()
            y_col_cb = tb.Combobox(subframe, state="readonly", textvariable=y_col_var,
                                   values=list(self.data.columns) if self.data is not None else [])
            y_col_cb.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

            self.dashboard_chart_configs.append({
                "chart_type_var": chart_type_var,
                "x_col_var": x_col_var,
                "y_col_var": y_col_var
            })

    # New method to generate a chart figure without displaying it
    def generate_chart_figure(self, x_column, y_column, chart_type):
        # Use similar logic as in create_visualization, but return the figure instead of calling show()
        x_column = self.get_column_name(x_column)
        y_column = self.get_column_name(y_column)
        fig = None
        try:
            if chart_type == "Scatter":
                fig = px.scatter(self.data, x=x_column, y=y_column,
                                 title=f"Scatter: {x_column} vs {y_column}")
            elif chart_type == "Line":
                fig = px.line(self.data, x=x_column, y=y_column,
                              title=f"Line: {x_column} vs {y_column}")
            elif chart_type == "Bar":
                fig = px.bar(self.data, x=x_column, y=y_column,
                             title=f"Bar: {x_column} vs {y_column}")
            elif chart_type == "Pie":
                fig = px.pie(self.data, names=x_column, values=y_column,
                             title=f"Pie: {x_column} vs {y_column}")
            elif chart_type == "Area":
                fig = px.area(self.data, x=x_column, y=y_column,
                              title=f"Area: {x_column} vs {y_column}")
            elif chart_type == "Bubble":
                fig = px.scatter(self.data, x=x_column, y=y_column,
                                 title=f"Bubble: {x_column} vs {y_column}",
                                 size=self.data.index, color=self.data.index, color_continuous_scale='Viridis')
            elif chart_type == "Histogram":
                fig = px.histogram(self.data, x=x_column,
                                   title=f"Histogram: {x_column}")
            else:
                # If the chart type is not implemented, return an empty figure
                fig = go.Figure()
                fig.add_annotation(text=f"Chart type '{chart_type}' not implemented", showarrow=False)
            # Update layout for consistency
            fig.update_layout(title_font=dict(family=self.font_family_var.get(),
                                              size=int(self.font_size_var.get()) + 8,
                                              color=self.chart_title_color),
                              xaxis=dict(title_font=dict(color=self.axis_label_color)),
                              yaxis=dict(title_font=dict(color=self.axis_label_color)),
                              font=dict(color=self.axis_label_color),
                              margin=dict(l=60, r=60, t=60, b=60))
        except Exception as e:
            messagebox.showerror("Error", f"Error generating chart: {e}")
        return fig

    # Modified create_dashboard: Combine all charts into one dashboard using subplots
    def create_dashboard(self):
        if self.data is None:
            messagebox.showerror("Error", "Please upload a dataset in the File & Data tab first.")
            return

        num_charts = len(self.dashboard_chart_configs)
        if num_charts == 0:
            messagebox.showerror("Error", "No chart configurations available.")
            return

        # Calculate grid dimensions for the dashboard (roughly square)
        cols = int(math.ceil(math.sqrt(num_charts)))
        rows = int(math.ceil(num_charts / cols))

        # Create subplots with appropriate titles
        subplot_titles = [f"Chart {i + 1}" for i in range(num_charts)]
        combined_fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        chart_index = 0
        for config in self.dashboard_chart_configs:
            chart_type = config["chart_type_var"].get()
            x_col = config["x_col_var"].get()
            y_col = config["y_col_var"].get()
            if not x_col or not y_col:
                messagebox.showerror("Error", "Please select X and Y columns for all charts.")
                return
            # Generate individual chart figure
            fig = self.generate_chart_figure(x_col, y_col, chart_type)
            # Determine the subplot position
            row = chart_index // cols + 1
            col = chart_index % cols + 1
            # Add all traces from the individual figure to the subplot
            for trace in fig.data:
                combined_fig.add_trace(trace, row=row, col=col)
            chart_index += 1

        combined_fig.update_layout(height=rows * 400, width=cols * 600, title_text="Custom Dashboard")
        combined_fig.show()
        self.update_status("Custom dashboard created successfully.")

    # ------------------ Page: Settings ------------------
    def _build_settings_page(self, parent):
        canvas = Canvas(parent, background=self.theme_color)
        vsb = tb.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tb.Frame(canvas, padding=10)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill=BOTH, expand=True)
        vsb.pack(side="right", fill=Y)

        font_frame = tb.Labelframe(scroll_frame, text="UI Font Settings", padding=10, bootstyle=INFO)
        font_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(font_frame, text="UI Font Family:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.font_family_cb = tb.Combobox(font_frame, state="readonly", textvariable=self.font_family_var,
                                          values=self.available_text_families)
        self.font_family_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(font_frame, text="UI Font Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.font_size_cb = tb.Combobox(font_frame, state="readonly", textvariable=self.font_size_var,
                                        values=["10", "12", "14", "16", "18", "20"])
        self.font_size_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        font_frame.columnconfigure(1, weight=1)

        theme_frame = tb.Labelframe(scroll_frame, text="UI Theme Settings", padding=10, bootstyle=INFO)
        theme_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(theme_frame, text="Choose Background Color", command=self.choose_theme_color,
                  bootstyle=SECONDARY).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Text Color", command=self.choose_text_color, bootstyle=SECONDARY).grid(
            row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Button Color", command=self.choose_button_color, bootstyle=SECONDARY).grid(
            row=0, column=2, padx=5, pady=5, sticky="w")
        theme_frame.columnconfigure(0, weight=1)
        theme_frame.columnconfigure(1, weight=1)
        theme_frame.columnconfigure(2, weight=1)

        chart_frame = tb.Labelframe(scroll_frame, text="Output Chart Appearance", padding=10, bootstyle=INFO)
        chart_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(chart_frame, text="Choose Chart Title Color", command=self.choose_chart_title_color,
                  bootstyle=SECONDARY).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(chart_frame, text="Choose Axis Label Color", command=self.choose_axis_label_color,
                  bootstyle=SECONDARY).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Label(chart_frame, text="Default Color Palette (comma-separated):").grid(row=1, column=0, padx=5, pady=5,
                                                                                    sticky="w")
        self.color_palette_entry = tb.Entry(chart_frame)
        self.color_palette_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        chart_frame.columnconfigure(1, weight=1)

        btn_frame = tb.Frame(scroll_frame, padding=10)
        btn_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(btn_frame, text="Reset to Defaults", command=self.reset_settings, bootstyle=DANGER).grid(row=0,
                                                                                                           column=0,
                                                                                                           padx=5,
                                                                                                           pady=5,
                                                                                                           sticky="ew")
        tb.Button(btn_frame, text="Apply Settings", command=self.apply_settings, bootstyle=SUCCESS).grid(row=0,
                                                                                                         column=1,
                                                                                                         padx=5, pady=5,
                                                                                                         sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        about_frame = tb.Labelframe(scroll_frame, text="About", padding=10, bootstyle=INFO)
        about_frame.pack(fill="both", expand=True, padx=10, pady=5)
        about_text = (
            "This application is designed to help with data cleaning, forecasting, anomaly detection, and visualization.\n\n"
            "Features include:\n"
            "• File upload and column selection.\n"
            "• File conversion between CSV, Excel, and SQLite formats.\n"
            "• Forecasting using Linear, Polynomial, and ARIMA models with adjustable forecasting horizon and confidence intervals.\n"
            "• Custom chart creation with title, color, and formatting options.\n"
            "• Custom Dashboard creation where you can specify multiple charts to generate a dashboard view.\n\n"
            "Developed by: ONKAR SINGH P\nVersion 2.0\n"
            "This interface uses a modern three-pane design with a left navigation sidebar, central content area, and right suggestions panel."
        )
        tb.Label(about_frame, text=about_text, wraplength=600, justify="left").pack(padx=5, pady=5)

    # ------------------ Right Pane: Chart Suggestions ------------------
    def _build_suggestions_panel(self):
        sug_frame = tb.Labelframe(self.suggestions_frame, text="Chart Suggestions", padding=10, bootstyle=INFO)
        sug_frame.pack(fill="both", expand=True)
        self.sug_canvas = Canvas(sug_frame, background="#ffffff")
        self.sug_scrollbar = tb.Scrollbar(sug_frame, orient="vertical", command=self.sug_canvas.yview)
        self.sug_canvas.configure(yscrollcommand=self.sug_scrollbar.set)
        self.sug_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.sug_scrollbar.pack(side=RIGHT, fill=Y)
        self.sug_inner = Frame(self.sug_canvas, background="#ffffff")
        self.sug_canvas.create_window((0, 0), window=self.sug_inner, anchor=NW)
        self.sug_inner.bind("<Configure>",
                            lambda e: self.sug_canvas.configure(scrollregion=self.sug_canvas.bbox("all")))
        self.sug_label = tb.Label(self.sug_inner, text="Upload a dataset to see chart suggestions.",
                                  background="#ffffff")
        self.sug_label.pack(pady=10)

    def update_suggestions(self):
        if self.data is not None:
            self.display_suggestions()

    def display_suggestions(self):
        for widget in self.sug_inner.winfo_children():
            widget.destroy()
        all_cols = list(self.data.columns)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
        suggestions = []
        for x, y in itertools.permutations(all_cols, 2):
            if x in numeric_cols and y in numeric_cols:
                suggestions.append(("Scatter", x, y))
                suggestions.append(("Line", x, y))
                suggestions.append(("Bar", x, y))
                suggestions.append(("Bubble", x, y))
            elif x in categorical_cols and y in numeric_cols:
                suggestions.append(("Bar", x, y))
                suggestions.append(("Pie", x, y))
        for col in numeric_cols:
            suggestions.append(("Histogram", col, ""))
        unique_suggestions = list(dict.fromkeys(suggestions))[:10]
        if unique_suggestions:
            lbl = tb.Label(self.sug_inner, text="Chart Suggestions:", font=(self.base_font_family, 12, "bold"),
                           background="#ffffff")
            lbl.pack(pady=5)
            for chart, x, y in unique_suggestions:
                btn_text = f"{chart}: X = {x}" + (f", Y = {y}" if y else "")
                btn = tb.Button(self.sug_inner, text=btn_text,
                                command=lambda ch=chart, x=x, y=y: self.suggestion_clicked(ch, x, y), bootstyle=INFO)
                btn.pack(pady=2, fill="x", padx=5)
            self.update_status("Chart suggestions updated.")
        else:
            lbl = tb.Label(self.sug_inner, text="No suggestions available.", background="#ffffff")
            lbl.pack(pady=10)
            self.update_status("No chart suggestions available.")

    def suggestion_clicked(self, chart, x, y):
        self.x_col_menu.set(x)
        self.y_col_menu.set(y)
        self.chart_menu.set(chart)
        self.create_visualization(x, y, chart)

    # ------------------ File and Data Methods ------------------
    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if not self.file_path:
            self.update_status("File upload cancelled.")
            return
        try:
            if self.file_path.endswith(".csv"):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith(".xlsx"):
                self.data = pd.read_excel(self.file_path)
            self.update_dropdowns()
            self.update_suggestions()
            self.update_status(f"File loaded: {self.file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.update_status("Failed to load file.", error=True)

    def update_dropdowns(self):
        if self.data is not None:
            columns = list(self.data.columns)
            self.x_col_menu['values'] = columns
            self.y_col_menu['values'] = columns
            self.z_col_menu['values'] = columns

    def update_z_axis_visibility(self, event=None):
        selected_chart = self.chart_menu.get()
        if "3D" in selected_chart:
            self.z_col_label.grid()
            self.z_col_menu.grid()
        else:
            self.z_col_label.grid_remove()
            self.z_col_menu.grid_remove()

    # ------------------ File Converter ------------------
    def convert_file(self):
        if self.data is None:
            messagebox.showerror("Error", "Please upload a file in the File Converter section first.")
            return
        output_format = self.output_format_cb.get()
        if not output_format:
            messagebox.showerror("Error", "Please select an output format (CSV, Excel, or SQLite).")
            return
        file_path = filedialog.asksaveasfilename(defaultextension="",
                                                 filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"),
                                                            ("SQLite DB", "*.db")])
        if not file_path:
            self.update_status("File conversion cancelled.")
            return
        try:
            if output_format == "CSV":
                self.data.to_csv(file_path, index=False)
            elif output_format == "Excel":
                self.data.to_excel(file_path, index=False)
            elif output_format == "SQLite":
                conn = sqlite3.connect(file_path)
                self.data.to_sql("converted_data", conn, if_exists="replace", index=False)
                conn.close()
            messagebox.showinfo("File Converter", f"File converted and saved to {file_path}")
            self.update_status(f"File converted to {output_format} and saved.")
        except Exception as e:
            messagebox.showerror("Error", f"File conversion failed: {e}")
            self.update_status("File conversion failed.", error=True)

    # ------------------ Custom Chart Creator ------------------
    def custom_chart(self):
        x_column = self.get_column_name(self.x_col_menu.get())
        y_column = self.get_column_name(self.y_col_menu.get())
        if not x_column or not y_column:
            messagebox.showerror("Error", "Please select both X and Y columns.")
            return
        chart_type = self.custom_chart_type_var.get()
        if chart_type == "Other":
            chart_type = simpledialog.askstring("Custom Chart",
                                                "Enter your custom chart type (e.g., 'Scatter', 'Line', etc.):")
            if not chart_type:
                messagebox.showerror("Error", "No custom chart type provided.")
                return
        title_text = self.custom_chart_title_var.get()
        custom_color = self.custom_chart_color_var.get().strip()
        palette = [custom_color] if custom_color else self.get_default_color_palette()
        try:
            if chart_type.lower() == "scatter":
                fig = px.scatter(self.data, x=x_column, y=y_column,
                                 title=title_text or f"Custom Scatter: {x_column} vs {y_column}",
                                 color_discrete_sequence=palette)
            elif chart_type.lower() == "line":
                fig = px.line(self.data, x=x_column, y=y_column,
                              title=title_text or f"Custom Line: {x_column} vs {y_column}",
                              color_discrete_sequence=palette)
            elif chart_type.lower() == "bar":
                fig = px.bar(self.data, x=x_column, y=y_column,
                             title=title_text or f"Custom Bar: {x_column} vs {y_column}",
                             color_discrete_sequence=palette)
            elif chart_type.lower() == "area":
                fig = px.area(self.data, x=x_column, y=y_column,
                              title=title_text or f"Custom Area: {x_column} vs {y_column}",
                              color_discrete_sequence=palette)
            elif chart_type.lower() == "bubble":
                fig = px.scatter(self.data, x=x_column, y=y_column,
                                 title=title_text or f"Custom Bubble: {x_column} vs {y_column}",
                                 size=self.data.index, color=self.data.index, color_continuous_scale=palette)
            elif chart_type.lower() == "pie":
                fig = px.pie(self.data, names=x_column, values=y_column,
                             title=title_text or f"Custom Pie: {x_column} vs {y_column}")
            else:
                messagebox.showerror("Error", f"Custom chart type '{chart_type}' not implemented.")
                return
            fig.update_layout(title_font=dict(family=self.font_family_var.get(),
                                              size=int(self.font_size_var.get()) + 8,
                                              color=self.chart_title_color),
                              xaxis=dict(title_font=dict(color=self.axis_label_color)),
                              yaxis=dict(title_font=dict(color=self.axis_label_color)),
                              font=dict(color=self.axis_label_color),
                              margin=dict(l=60, r=60, t=60, b=60))
            fig.show()
            self.update_status("Custom chart generated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Custom chart failed: {e}")
            self.update_status("Custom chart generation failed.", error=True)

    # ------------------ Visualization & Forecasting ------------------
    def create_visualization(self, x_column, y_column, chart_type):
        # This method is still used for individual chart display
        x_column = self.get_column_name(x_column)
        y_column = self.get_column_name(y_column)
        try:
            fig = None
            if chart_type == "Scatter":
                fig = px.scatter(self.data, x=x_column, y=y_column,
                                 title=f"Scatter plot: {x_column} vs {y_column}")
            elif chart_type == "Line":
                fig = px.line(self.data, x=x_column, y=y_column,
                              title=f"Line plot: {x_column} vs {y_column}")
            elif chart_type == "Bar":
                fig = px.bar(self.data, x=x_column, y=y_column,
                             title=f"Bar chart: {x_column} vs {y_column}")
            elif chart_type == "Pie":
                fig = px.pie(self.data, names=x_column, values=y_column,
                             title=f"Pie chart: {x_column} vs {y_column}")
            elif chart_type == "Area":
                fig = px.area(self.data, x=x_column, y=y_column,
                              title=f"Area plot: {x_column} vs {y_column}")
            elif chart_type == "Bubble":
                fig = px.scatter(self.data, x=x_column, y=y_column,
                                 title=f"Bubble plot: {x_column} vs {y_column}",
                                 size=self.data.index, color=self.data.index, color_continuous_scale='Viridis')
            elif chart_type == "Waterfall":
                fig = px.waterfall(self.data, x=x_column, y=y_column,
                                   title=f"Waterfall plot: {x_column} vs {y_column}")
            elif chart_type == "Histogram":
                fig = px.histogram(self.data, x=x_column,
                                   title=f"Histogram: {x_column}")
            elif chart_type == "Box Plot":
                fig = px.box(self.data, y=y_column,
                             title=f"Box plot: {y_column}")
            elif chart_type == "Heatmap":
                fig = px.density_heatmap(self.data, x=x_column, y=y_column,
                                         title=f"Heatmap: {x_column} vs {y_column}")
            elif chart_type == "3D Scatter":
                z_column = self.get_column_name(self.z_col_menu.get())
                if not z_column:
                    messagebox.showerror("Error", "Please select a Z-Axis column for 3D charts.")
                    self.update_status("Missing Z-Axis for 3D Scatter.", error=True)
                    return
                fig = px.scatter_3d(self.data, x=x_column, y=y_column, z=z_column, title="3D Scatter")
            elif chart_type == "3D Bubble":
                z_column = self.get_column_name(self.z_col_menu.get())
                if not z_column:
                    messagebox.showerror("Error", "Please select a Z-Axis column for 3D charts.")
                    self.update_status("Missing Z-Axis for 3D Bubble.", error=True)
                    return
                fig = px.scatter_3d(self.data, x=x_column, y=y_column, z=z_column, size=self.data.index,
                                    title="3D Bubble")
            elif chart_type == "3D Surface":
                z_column = self.get_column_name(self.z_col_menu.get())
                if not z_column:
                    messagebox.showerror("Error", "Please select a Z-Axis column for 3D charts.")
                    self.update_status("Missing Z-Axis for 3D Surface.", error=True)
                    return
                fig = px.surface(self.data, x=x_column, y=y_column, z=z_column, title="3D Surface")
            else:
                messagebox.showinfo("Not Implemented", f"The chart type '{chart_type}' is not implemented.")
                self.update_status(f"Chart type '{chart_type}' not implemented.")
                return

            fig.update_layout(title_font=dict(family=self.font_family_var.get(),
                                              size=int(self.font_size_var.get()) + 8,
                                              color=self.chart_title_color),
                              xaxis=dict(title_font=dict(color=self.axis_label_color)),
                              yaxis=dict(title_font=dict(color=self.axis_label_color)),
                              font=dict(color=self.axis_label_color),
                              margin=dict(l=60, r=80, t=60, b=60))
            if self.prediction_var.get() and "Prediction" in self.data.columns and chart_type in ["Scatter", "Line",
                                                                                                  "Bubble"]:
                sorted_df = self.data.sort_values(by=x_column)
                pred_trace = go.Scatter(
                    x=sorted_df[x_column],
                    y=sorted_df["Prediction"],
                    mode="lines",
                    name="Fitted Prediction",
                    line=dict(color="red", width=2)
                )
                fig.add_trace(pred_trace)
                if self.forecast_x is not None and self.forecast_y is not None and len(self.forecast_x) > 0:
                    forecast_trace = go.Scatter(
                        x=self.forecast_x,
                        y=self.forecast_y,
                        mode="lines",
                        name="Forecast",
                        line=dict(color="red", width=2, dash="dash")
                    )
                    fig.add_trace(forecast_trace)
                    if self.forecast_ci is not None:
                        lower, upper = self.forecast_ci
                        fig.add_trace(go.Scatter(
                            x=list(self.forecast_x) + list(self.forecast_x[::-1]),
                            y=list(upper) + list(lower[::-1]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=True,
                            name="95% CI"
                        ))
            if self.anomalies is not None and chart_type in ["Scatter", "Line", "Bubble"]:
                anomaly_points = self.data.loc[self.anomalies]
                if not anomaly_points.empty:
                    anom_trace = go.Scatter(
                        x=anomaly_points[x_column],
                        y=anomaly_points[y_column],
                        mode="markers",
                        name="Anomalies",
                        marker=dict(color="black", size=10, symbol="x")
                    )
                    fig.add_trace(anom_trace)
            fig.show()
            self.update_status(f"{chart_type} chart generated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create visualization: {e}")
            self.update_status("Visualization failed.", error=True)

    def generate_chart(self):
        if self.data is None:
            messagebox.showerror("Error", "Please upload a dataset first.")
            self.update_status("No dataset loaded.", error=True)
            return
        chart_type = self.chart_menu.get()
        x_column = self.x_col_menu.get()
        y_column = self.y_col_menu.get()
        if not chart_type:
            messagebox.showerror("Error", "Please select a chart type.")
            self.update_status("Chart type not selected.", error=True)
            return
        if not x_column or not y_column:
            messagebox.showerror("Error", "Please select both X-Axis and Y-Axis columns.")
            self.update_status("X or Y column not selected.", error=True)
            return
        self.create_visualization(x_column, y_column, chart_type)

    # ------------------ Forecasting / Prediction ------------------
    def toggle_prediction(self):
        if self.prediction_var.get():
            if self.data is None:
                messagebox.showerror("Error", "Please upload a dataset first.")
                self.prediction_var.set(False)
                self.update_status("Prediction failed: no dataset loaded.", error=True)
                return
            x_column = self.x_col_menu.get()
            y_column = self.y_col_menu.get()
            if not x_column or not y_column:
                messagebox.showerror("Error", "Please select both X-Axis and Y-Axis columns for prediction.")
                self.prediction_var.set(False)
                self.update_status("Prediction failed: columns not selected.", error=True)
                return
            try:
                X_series = self.convert_series(self.data[x_column]).dropna()
                y_series = self.convert_series(self.data[y_column]).dropna()
                valid_mask = X_series.notna() & y_series.notna()
                if valid_mask.sum() == 0:
                    messagebox.showerror("Error", "No valid numeric data available for prediction.")
                    self.prediction_var.set(False)
                    self.update_status("Prediction failed: no valid numeric data.", error=True)
                    return

                X = X_series[valid_mask].values.reshape(-1, 1)
                y = y_series[valid_mask].values

                model_choice = self.forecast_model_var.get()
                forecast_horizon = int(
                    self.forecast_horizon_var.get()) if self.forecast_horizon_var.get().isdigit() else 5

                if model_choice == "Polynomial" and valid_mask.sum() >= 5:
                    poly = PolynomialFeatures(degree=3)
                    X_poly = poly.fit_transform(X)
                    model = LinearRegression()
                    model.fit(X_poly, y)
                    y_pred = model.predict(X_poly)
                    x_sorted = np.sort(X.ravel())
                    diff = np.median(np.diff(x_sorted)) if len(x_sorted) > 1 else 1
                    forecast_x = np.linspace(x_sorted[-1] + diff, x_sorted[-1] + forecast_horizon * diff,
                                             forecast_horizon)
                    forecast_X_poly = poly.transform(forecast_x.reshape(-1, 1))
                    forecast_y = model.predict(forecast_X_poly)
                    self.forecast_x = forecast_x
                    self.forecast_y = forecast_y
                    self.forecast_ci = None
                elif model_choice == "ARIMA" and len(y) > 10:
                    model = ARIMA(y, order=(1, 1, 1))
                    model_fit = model.fit()
                    y_pred = model_fit.fittedvalues
                    forecast_result = model_fit.get_forecast(steps=forecast_horizon)
                    forecast_output = forecast_result.predicted_mean
                    if self.conf_int_var.get():
                        conf_int = forecast_result.conf_int(alpha=0.05)
                        lower = conf_int.iloc[:, 0].values
                        upper = conf_int.iloc[:, 1].values
                        self.forecast_ci = (lower, upper)
                    else:
                        self.forecast_ci = None
                    self.forecast_x = np.arange(np.max(X) + 1, np.max(X) + forecast_horizon + 1)
                    self.forecast_y = forecast_output.values
                else:
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    self.forecast_x = np.array([])
                    self.forecast_y = np.array([])
                    self.forecast_ci = None

                self.data.loc[valid_mask, "Prediction"] = y_pred
                messagebox.showinfo("Prediction",
                                    f"Prediction complete for '{y_column}' using '{x_column}' with {model_choice} model.")
                self.update_status("Prediction completed successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")
                self.update_status("Prediction failed.", error=True)
                self.prediction_var.set(False)
        else:
            messagebox.showinfo("Prediction", "Prediction mode is now disabled.")
            self.update_status("Prediction mode disabled.")

    # ------------------ Export Predictions ------------------
    def export_predictions(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Export", f"Data exported to {file_path}")
                self.update_status(f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                self.update_status("Export failed.", error=True)

    # ------------------ Helper: Convert Series ------------------
    def convert_series(self, series):
        s = pd.to_numeric(series, errors='coerce')
        if s.isna().all():
            codes, _ = pd.factorize(series)
            s = pd.Series(codes, index=series.index)
        return s

    # ------------------ Settings Methods ------------------
    def apply_settings(self):
        self._update_style_fonts()
        self.update_colors()
        messagebox.showinfo("Settings", "Settings applied successfully.")
        self.update_status("Settings applied.")

    def choose_theme_color(self):
        color_code = colorchooser.askcolor(title="Choose Background Color")
        if color_code[1]:
            self.theme_color = color_code[1]
            self.apply_settings()

    def choose_text_color(self):
        color_code = colorchooser.askcolor(title="Choose Text Color")
        if color_code[1]:
            self.text_color = color_code[1]
            self.apply_settings()

    def choose_button_color(self):
        color_code = colorchooser.askcolor(title="Choose Button Color")
        if color_code[1]:
            self.button_color = color_code[1]
            self.apply_settings()

    def choose_chart_title_color(self):
        color_code = colorchooser.askcolor(title="Choose Chart Title Color")
        if color_code[1]:
            self.chart_title_color = color_code[1]
            messagebox.showinfo("Settings", "Chart title color updated.")

    def choose_axis_label_color(self):
        color_code = colorchooser.askcolor(title="Choose Axis Label Color")
        if color_code[1]:
            self.axis_label_color = color_code[1]
            messagebox.showinfo("Settings", "Axis label color updated.")

    def reset_settings(self):
        self.font_family_var.set(self.base_font_family)
        self.font_size_var.set(str(self.base_font_size))
        self.theme_color = self.light_theme["bg"]
        self.text_color = self.light_theme["text"]
        self.button_color = self.light_theme["button"]
        self.chart_title_color = self.light_theme["chart_title"]
        self.axis_label_color = self.light_theme["axis_label"]
        self.default_color_palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
        if hasattr(self, "color_palette_entry"):
            self.color_palette_entry.delete(0, tb.END)
        self.apply_settings()
        messagebox.showinfo("Settings", "Settings reset to defaults.")


if __name__ == "__main__":
    app = tb.Window(themename="flatly")
    DataVizApp(app)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("Application closed.")
