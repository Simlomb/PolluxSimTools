# luvoir_simtools
simulation tools for the LUVOIR Surveyor STDT based on https://github.com/tumlinson/luvoir_simtools

(1) you must have bokeh 0.12 installed to use these tools 
    http://bokeh.pydata.org/en/latest/docs/installation.html

(2) This tool has been tested with Python 2.7.15rc1 there might be some incompatibilities with python 3.


With this new tool the calculations are quite easy. You would need to have python and install bokeh (https://bokeh.pydata.org/en/latest/docs/installation.html) and this should be all (you can contact me if you have issues).


1) In this package you will have all the web tools (ETC) that you already know, and you can run them as follows:

- once you are in the main directory (in  a terminal), you can locally run the web etc for pollux and lumos by typing:

bokeh serve --show pollux_etc/

or

bokeh serve --show lumos_etc/

A window should pop up in your browser with the usual ETC interface. Remember that you can only use one of them at a time.


2) In addition, in this package there is also a new python script. This allows more freedom in the input to provide. You can run the new script, by typing in the main directory in the terminal:

python main_pollux.py -h

the option -h lists all the options available.

For example if you want to have an M3 Dwarf of mag = 7 and exposure time 0.5hr you do:

python main_pollux.py -i mdwarf2 -m 7 -e 0.5

A series of calculation will start and once they are done a plot will pop up. Once you close it, it will be saved in pollux_tools/plots and the s/n file will be saved in pollux_tool/files

If you do not want to have the S/N but you want to have the 2d image, you do:

python main_pollux.py -i mdwarf2 -m 7 -e 0.5 -n 0 -d 1

The saving of the image and file will be a bit longer as the dimensions are higher.

You can also decide to have the image or the s/n or both in the photon counting mode of the detector.

python main_pollux.py -i mdwarf2 -m 7 -e 0.5 -p 1

the exposure time in input will be the total exposure time, and the exposure time of each frame will be automatically calculated and printed on the terminal.
More info can be found in POLLUX_simulator.pdf
