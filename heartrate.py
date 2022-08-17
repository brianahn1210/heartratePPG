import heartpy as hp
import matplotlib.pyplot as plt
import Data_Parsing_Pneumonia as gd
#download heartpy
#https://github.com/paulvangentcom/heartrate_analysis_python

#installation by
#python -m pip install heartpy
#or
#python setup.py install

#load the PPG signal
#data = gd.sendppg()
data, timer = hp.load_exampledata(2)
#and visualise
plt.figure(figsize=(12,4))
plt.plot(data)
plt.show()

#run the analysis
wd, m = hp.process(data, sample_rate = 100.0)

#display measures computed (just bpm in this case)
for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
    break
print(type(m))
# #call plotter
# hp.plotter(wd, m)