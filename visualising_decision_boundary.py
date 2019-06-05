import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression

# Create a grid of points
xres = 100
yres = 100
xx = np.linspace(xmin,xmax,xres)
yy = np.linspace(ymin,ymax,yres)
grid = np.zeros((xres*yres,2))
for i in range(yres):
    grid[i*xres:(i+1)*xres,0] = xx
for i in range(yres):
    for j in range(xres):
        grid[i*xres+j,1] = yy[i]
        
# Make a prediction.
# You can use any classifier that you want 

clf = LogisticRegression()
clf.fit(X,y)
preds = clf.predict(grid)

# Then plot with the following: 
# You can change the colors by adjusting the variable colors using RGB. 
# The color map makes a smooth transition from colors[0] to colors[1] to colors[2]. 
# These will correspond with prediction=0 to prediction=0.5 to prediction=1.0.
colors = [(0.5, 0.5, 0.5), (0.65, 0.65, 0.65), (0.8, 0.8, 0.8)]
cm = LinearSegmentedColormap.from_list('mycolors', colors, N=100)    
plt.scatter(grid[:,0], grid[:,1], c=preds, cmap=cm)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
