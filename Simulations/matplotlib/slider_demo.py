# https://matplotlib.org/3.1.1/gallery/widgets/slider_demo.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)

l, = plt.plot(t, s, lw=2)

axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
axamp = plt.axes([0.25, 0.15, 0.65, 0.03])

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0)
samp = Slider(axamp, 'Amp', 0.1, 10.0)

def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()

sfreq.on_changed(update)
samp.on_changed(update)

plt.show()
