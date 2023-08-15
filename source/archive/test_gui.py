

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np

# Define the layout of the GUI
layout = [
    [sg.Text('Input 1'), sg.Input(key='input1')],
    [sg.Text('Input 2'), sg.Input(key='input2')],
    [sg.Text('Input 3'), sg.Input(key='input3')],
    [sg.Text('Input 4'), sg.Input(key='input4')],
    [sg.Text('Input 5'), sg.Input(key='input5')],
    [sg.Button('Plot')]
]

# Create the window
window = sg.Window('Plot GUI', layout)

# Event loop to process GUI events
while True:
    event, values = window.read()

    # Handle events
    if event == sg.WINDOW_CLOSED:
        break
    elif event == 'Plot':
        # Get user inputs
        input1 = float(values['input1'])
        input2 = float(values['input2'])
        input3 = float(values['input3'])
        input4 = float(values['input4'])
        input5 = float(values['input5'])

        # Perform calculations based on the inputs
        x = np.linspace(0, 10, 100)
        y = input1 * x**2 + input2 * x + input3 * np.sin(input4 * x + input5)

        # Generate the plot
        plt.figure()
        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Output Plot')

        # Display the plot
        plt.show()

# Close the window
window.close()




# from flask import Flask, render_template, request
# import matplotlib.pyplot as plt
# import numpy as np

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/plot', methods=['POST'])
# def plot():
#     # Get user inputs from the form
#     input1 = float(request.form['input1'])
#     input2 = float(request.form['input2'])
#     input3 = float(request.form['input3'])
#     input4 = float(request.form['input4'])
#     input5 = float(request.form['input5'])

#     # Perform calculations based on the inputs
#     x = np.linspace(0, 10, 100)
#     y = input1 * x**2 + input2 * x + input3 * np.sin(input4 * x + input5)

#     # Generate the plot
#     plt.figure()
#     plt.plot(x, y)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Output Plot')

#     # Save the plot as HTML
#     plot_html = 'plot.html'
#     plt.savefig(plot_html)

#     # Pass the plot HTML to the template
#     return render_template('plot.html', plot=plot_html)

# if __name__ == '__main__':
#     app.run(debug=True)


'''

Here's what you need to do:

1. Create a new directory for your project and navigate into it.
2. Create a new Python file (e.g., `app.py`) and paste the above code into it.
3. Create a new directory named `templates` and inside it, create two HTML files: `index.html` and `plot.html`.
4. In `index.html`, create a form with input fields for your variables and a submit button. Ensure that the form's action points to `/plot`.
5. In `plot.html`, display the plot using an `<img>` tag, passing the plot HTML file name as the `src` attribute.
6. Install Flask by running `pip install flask`.
7. Start the Flask development server by running `python app.py`.
8. Open your web browser and go to `http://localhost:5000` to access the GUI.

This is a basic example to get you started. You can enhance the GUI by customizing the HTML templates, adding validation to the form inputs, and handling more complex interactions with the user. 
'''