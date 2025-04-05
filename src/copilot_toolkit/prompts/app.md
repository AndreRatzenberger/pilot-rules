# Create a static web site

>Analyzes input data and creates a web site dedicated to the optimal representation of the input data

## Prompt


"An agent that generates a static html app based on the requirements and the input_data. "
"The input_data is the content of a file that contains the data to be displayed in the app."
"The final app should offer an upload so the user can upload a file that contains input_data to render."
"The app should present the content of the input file as if designed by a professional UX designer and dedicated to the data in the input file."
"For example, if the input data is a story, the app should present the story as if it is a dedicated story app."
"If for example the input data is a list of products, the app should present the products as if it is a dedicated product app.",
"The output is a dictionary of kv pairs with the keys being the relative filepaths and the values the content. see definition."
"The expected name of the input data file should be part of the output. see definition"
"app requirements: static web app with html+css+js, data upload, elegant, professional, color-coded, modern, dark mode"
"data-optimized presentation. markdown should be displayed as beautiful markdown, code as beautiful code and so on."
"output definition

The output consist always of three files:

a html file
a css file
a js file"
