import json

history_path = 'outputs/ResUNet.json'

history_file = open(history_path, "r")
history = history_file.read()
history_file.close()

history = json.loads(history)

import matplotlib.pyplot as plt
plt.plot(history['train'])
plt.plot(history['valid'])
plt.legend(['train loss', 'valid loss'])
plt.show()