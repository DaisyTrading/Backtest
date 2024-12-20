import os
import warnings
import webbrowser

import matplotlib.pyplot as plt
import mpld3
import pandas as pd
import pyfolio as pf

warnings.filterwarnings("ignore")

# Load the performance DataFrame from dma.pickle
performance = pd.read_pickle("dma.pickle")

# Generate Pyfolio Tearsheet (returns analysis only)
pf.create_returns_tear_sheet(returns=performance['returns'])


def save_and_open_plots_as_html(fig, file_name="tearsheet.html"):
    """
    Save the current Matplotlib plots to an HTML file and display it in the browser.

    Parameters:
        fig: The current Matplotlib figure object.
        file_name: Name of the HTML file to save the plots (default 'tearsheet.html').
    """
    # Ensure the figure object is passed
    if fig is None:
        fig = plt.gcf()

    # Save plots as an HTML file
    html_file_path = file_name
    with open(html_file_path, "w") as f:
        html_content = mpld3.fig_to_html(fig)
        f.write("<html><head><title>Pyfolio Tearsheet</title></head><body>")
        f.write("<h1 style='text-align: center;'>Pyfolio Tearsheet</h1>")
        f.write(html_content)
        f.write("</body></html>")

    # Automatically open the HTML file in the browser
    webbrowser.open("file://" + os.path.abspath(html_file_path))


# Get the current figure and save it as an HTML file
fig = plt.gcf()
save_and_open_plots_as_html(fig, "tearsheet.html")
