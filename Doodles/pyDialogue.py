import tkinter as tk
from tkinter import filedialog, ttk

def askDIR():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    DIR_path = filedialog.askdirectory()
    return DIR_path

def askFILE():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    FILE_path = filedialog.askopenfilename()
    return FILE_path

def askFILES():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    FILE_path = filedialog.askopenfilenames()
    return FILE_path