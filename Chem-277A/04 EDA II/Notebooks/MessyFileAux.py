# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 01:33:56 2025

@author: MMH_user
"""
#import re
import os                           # for navigating through folders/harddiscs etc
import ipywidgets as widgets
from IPython.display import display, clear_output
import inspect
from IPython import get_ipython


def ShowInteractiveTable(df, editable_col = "Example", user_globals = None):
    
    
    if user_globals is None:
        try:
            ip = get_ipython()
            if ip is not None:
                user_globals = ip.user_global_ns
        except Exception:
            user_globals = None

        if user_globals is None:
            # fallback to caller globals (useful if called from plain python)
            caller_frame = inspect.currentframe().f_back
            user_globals = caller_frame.f_globals if caller_frame is not None else {}

    
    header_style = "font-weight:bold; text-decoration:underline; padding-right:10px;"

    # Default widths – use first N to match your layout, fallback to equal split if more columns
    default_widths = [80, 320, 380]  # for first 3 columns
    total_default = sum(default_widths)
    remaining_space = 1040 - total_default  # total width = columns + run + output
    if len(df.columns) > 3:
        extra_width = remaining_space // (len(df.columns) - 3)
        col_widths = default_widths + [extra_width] * (len(df.columns) - 3)
    else:
        col_widths = default_widths[:len(df.columns)]

    # Build dynamic header
    header_items = []
    for col, w in zip(df.columns, col_widths):
        name = f"{col} (editable)" if col == editable_col else col
        header_items.append(widgets.HTML(f"<div style='width:{w}px;{header_style}'>{name}</div>"))
    header_items.append(widgets.HTML(f"<div style='width:60px;{header_style}'>Run</div>"))
    header_items.append(widgets.HTML(f"<div style='flex:1;{header_style}'>Output</div>"))  # flexible width
    header = widgets.HBox(header_items)

    # Build rows
    rows = []
    for _, row in df.iterrows():
        cell_widgets = []
        for col, w in zip(df.columns, col_widths):
            if col == editable_col:
                # Editable code cell
                code_box = widgets.Textarea(
                    value=str(row[col]),
                    layout=widgets.Layout(width=f'{w}px', height='50px'),
                    continuous_update=False
                )
                cell_widgets.append(code_box)
            else:
                # Static HTML cell
                cell_widgets.append(
                    widgets.HTML(f"<div style='width:{w}px'>{row[col]}</div>")
                )

        # Output area
        output = widgets.Output(
            layout=widgets.Layout(
                flex='1',
                min_height='50px',
                max_height='300px',  # prevent infinite growth
                overflow_y='auto',
                overflow_x='auto',
                border='1px solid #ddd',
                padding='4px',
                white_space='pre-wrap',  # wrap long lines
                word_break='break-word'
                )
            )
        # Run button
        run_button = widgets.Button(
            description="▶",
            button_style='success',
            tooltip="Run example",
            layout=widgets.Layout(width='40px')
        )

        # Define run behavior
        g = user_globals

        def on_click(b, code_widget=code_box, out_area=output, g=g):
            with out_area:
                clear_output()
                if code_widget is None:
                    print(f"No editable column '{editable_col}' found for this row.")
                    return
                try:
                    # Execute in the chosen globals namespace
                    exec(code_widget.value, g)
                except Exception as e:
                    print(f"Error: {e}")

        run_button.on_click(on_click)

        # Build row
        row_box = widgets.HBox(cell_widgets + [run_button, output])
        rows.append(row_box)

    # Combine header + rows
    table = widgets.VBox([header] + rows)
    display(table)
    
    
def FindMyFile(filename: str, ServerHardDiscPath: str = r"c:\Users\MMH_user\Desktop") -> str:
    """
    finds file of name "filename" anywhere in "ServerHardDiscPath" and returns complete path
    """
    for r,d,f in os.walk(ServerHardDiscPath):
        for files in f:
             if files == filename: #for example "MessyFile.xlsx"
                 file_name_and_path =  os.path.join(r,files)
                 return file_name_and_path