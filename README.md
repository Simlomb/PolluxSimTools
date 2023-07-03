# luvoir_simtools
simulation tools for the POLLUX instrument of the HWO based on https://github.com/tumlinson/luvoir_simtools

## HOW TO INSTALL

Anaconda usage for installation is highly advised. After having installed anaconda, you can do this:
```
conda config --add channels http://ssb.stsci.edu/astroconda
```
```
conda create -n test python=2
```
```
conda activate test
```
```
conda install numpy astropy pysynphot pyyaml scipy specutils bokeh matplotlib dotdict
```
```
git clone https://github.com/Simlomb/PolluxSimTools.git
```
```
cd PolluxSimTools
```
```
git checkout origin/PolluxSimTool2023
```

Please check that you are in the correct branch (PolluxSimTool2023) by doing

```
git branch
```

The PolluxSimTool2023 should be the one highlighted.

Please add these lines with the correct path in which you have the pollux simulato to your .bashrc like:

```
export PYSYN_CDBS=/path_to_simulator/PolluxSimTools/data_reference/cdbs
```
  
Please add also the path to the simulator to your PYTHONPATH in the .bashrc like:
    
```
export PYTHONPATH=$PYTHONPATH:/path_to_simulator/PolluxSimTools/
```

## HOW TO RUN IT

1) In this package you will have all the web tools (ETC), and you can run them as follows:

- once you are in the main directory (in  a terminal), you can locally run the web etc for pollux by typing:
```
bokeh serve --show pollux_etc/
```

A window should pop up in your browser with the usual ETC interface. Remember that for this ETC only the 8 m aperture option is available.


2) In addition, in this package there is also a new python script. This allows more freedom in the input to provide. You can run the new script, by typing in the main directory in the terminal:
```
python main_pollux.py -h
```
the option -h lists all the options available.


For example if you want to have an M3 Dwarf of mag = 7 and exposure time 0.5hr you do:
```
python main_pollux.py -i mdwarf2 -m 7 -e 0.5
```

A series of calculation will start and once they are done a plot will pop up. Once you close it, it will be saved in pollux_tools/plots and the s/n file will be saved in pollux_tool/files
the exposure time in input will be the total exposure time, and the exposure time of each frame will be automatically calculated and printed on the terminal.
More info can be found in POLLUX_simulator.pdf

UPDATED: 
A 2D image simulator possibility was available in the previous version but not anymore as the current design is not mature enough.



## IMPORTANT NOTES:

(1) you must have bokeh (tested with 0.12) installed to use these tools 
    (https://docs.bokeh.org/en/latest/docs/first_steps/installation.html)

(2) This tool has been tested with Python 2.7.15rc1 it will not work with python 3. So after having created the environment with python 2 in conda remember to always activate it before using the tool, as explained above (conda activate test).

If you are not using anaconda, and you have an error related with pysynphot, please try to install pysynphot with the package manager.


