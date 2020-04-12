class PlotTheCurve():
    def __init__(self, cycle_count, lr_min, lr_max, step_size):
        self.cycle_count = cycle_count
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size
        
    def setX(self,cycle,cycle_count):
        list1 = []
        mincycleval, maxcycleval = cycle
        step_size = maxcycleval-mincycleval
        minim = mincycleval
        for i in range(mincycleval,(step_size * self.cycle_count)+1):
            if i-minim==step_size:
                list1.append(i)
                minim = i
            list1.append(i)
        return list1[:step_size*self.cycle_count]
        
        
    def setY(self,minval, maxval, cycle, step_size):
        list2 = []
        for i in range(cycle):
            list2.append(minval)
            list2.append(maxval)
            list2.append(minval)
        return list2[:cycle*step_size]
    
    
    def plot(self):
        import matplotlib.pyplot as plt 

        cycle_count=self.cycle_count
        # x axis values 
        x = self.setX((1,3),cycle_count)
        #print("X ", x)

        # Set the minimum and maximum values of LR
        lr_min = self.lr_min
        lr_max = self.lr_max
        step_size = self.step_size
        # corresponding y axis values 
        y = self.setY(lr_min,lr_max,cycle_count,step_size) 
        #print("Y ", y)
        # plotting the points  
        plt.plot(x, y, color='green') 

        # naming the x axis 
        plt.xlabel('Iterations (10e4)') 
        # naming the y axis 
        plt.ylabel('LR Range') 

        # giving a title to my graph 
        plt.title('Triangular schedule') 

        # function to show the plot 
        plt.show()
        
