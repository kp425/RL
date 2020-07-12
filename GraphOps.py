import math
import matplotlib.pyplot as plt
import statistics
from IPython.display import clear_output

class Datapoint:
    def __init__(self, name):
        self.x = []
        self.y = []
        self.name = name

    def collect(self, x_,y_):
        self.x.append(x_)
        self.y.append(y_)

    def avg_x(self,from_index = None , to_index = None):
        x = self.x[from_index:to_index]
        return statistics.mean(x)
    
    def avg_y(self,from_index = None , to_index = None):
        y = self.y[from_index:to_index]
        return statistics.mean(y)

def plot(data, graphs_per_row = 2, figsize = (10,10), gap=(0.2,0.5)):  #data is list of Datapoints
    
    ncols = graphs_per_row
    nrows = int(math.ceil(len(data)/ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=figsize)

    # grid = plt.GridSpec(nrows, ncols, wspace=gap[0], hspace=gap[1])

    if len(axes.shape) == 1:
        counter = 0
        for x in range(nrows):
            if counter == len(data):
                break
            tup = data[counter]
            axes[x].title.set_text(tup.name)
            axes[x].plot(tup.x,tup.y)  
            counter += 1            
    else:
        counter = 0
        for x in range(nrows):
            for y in range(ncols):
                if counter == len(data):
                    break
                tup = data[counter]
                axes[x][y].title.set_text(tup.name)
                axes[x][y].plot(tup.x,tup.y)
                counter += 1   
    plt.show()


def live_plot(data, graphs_per_row = 2, figsize = (10,10), gap=(0.2,0.5)):
    plot(data, graphs_per_row = graphs_per_row, figsize = figsize, gap= gap)
    clear_output(True)


'''
Example Usage

policy_dp = Datapoint("policy_loss")
value_dp = Datapoint("value_loss")
entropy_dp = Datapoint("entropy_loss")
loss_dp = Datapoint("loss_dp")
counter = 0


for loop :
    policy_dp.collect(counter, policy_loss.numpy())
    value_dp.collect(counter, value_loss.numpy())
    entropy_dp.collect(counter, entropy_loss.numpy())
    oss_dp.collect(counter, loss.numpy())
    counter += 1
    live_plot([policy_dp, value_dp, entropy_dp, loss_dp])

'''
