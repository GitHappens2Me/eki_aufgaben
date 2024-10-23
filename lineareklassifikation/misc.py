from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


class Plotter:
    def __init__(self,  coords_x1, coords_x2, class_y) -> None:

        self.fig = plt.figure()
        self.ax2d = self.fig.add_subplot(1,2,1)
        self.ax3d = self.fig.add_subplot(1,2,2,projection='3d')
        
        self.coords_x1 = coords_x1
        self.coords_x2 = coords_x2
        self.class_y = class_y

        self.min_x = np.min(coords_x1)
        self.max_x = np.max(coords_x1)

        self.min_y = np.min(coords_x2)
        self.max_y = np.max(coords_x2)

        plt.ion()
        plt.show()


    def scatter(self):
        self.ax2d.scatter(self.coords_x1,self.coords_x2, c = list(map(lambda x: 'r' if x== 1 else 'b', self.class_y)))
        self.ax3d.scatter(self.coords_x1, self.coords_x2, self.class_y,c = list(map(lambda x: 'r' if x== 1 else 'b', self.class_y)))

    def getMeshgrid(self,  limit_x = 2, limit_y = 2):


        x1, x2 = np.meshgrid(np.linspace(self.min_x,self.max_x,50), np.linspace(self.min_y,self.max_y,50))
        return x1, x2

    def contour(self,x1,x2,z):
        
        
        blue_indices = (self.class_y < 0.0)
        red_indices = (self.class_y > 0.0)
        self.ax2d.contourf(x1, x2,  np.clip(z, -1, 1), levels=np.linspace(-1,1,20),cmap="coolwarm", alpha=.2)
        self.ax2d.contour(x1, x2,  np.clip(z, -1, 1), levels=[0.0], colors=["k"])
        self.ax2d.plot(self.coords_x1[blue_indices], self.coords_x2[blue_indices], 'bo')
        self.ax2d.plot(self.coords_x1[red_indices], self.coords_x2[red_indices], 'ro')


        self.ax2d.set_xlim((self.min_x,self.max_x))
        self.ax2d.set_ylim((self.min_y,self.max_y))

        self.ax3d.set_xlim((self.min_x,self.max_x))
        self.ax3d.set_ylim((self.min_y,self.max_y))
        self.ax3d.set_zlim((np.min(z),np.max(z)))

        # x1, x2 = np.meshgrid(np.linspace(-1.5,1.5,50), np.linspace(-1.5,1.5,50))
        # z = w[0] + w[1] * x1 + w[2] * x2 + w[3] * (x1 ** 2) + w[4] * (x2 ** 2) + w[5] * x1 * x2
        self.ax3d.plot_surface(x1, x2, z, cmap=cm.YlGn, alpha=0.2)  # Plot contour curves
        self.ax3d.contourf(x1, x2, np.clip(z, -1, 1)*0.01, levels=[-1*0.01,0,1*0.01],colors=["b","r"], alpha=.3)
        self.ax3d.contour(x1, x2, z, levels=[0.0], colors=["k"])


# def plot(model_fun, coords_x1, coords_x2, class_y, x1, x2, z):

#     fig = plt.figure()
#     ax2d = fig.add_subplot(1,2,1)
#     ax3d = fig.add_subplot(1,2,2,projection='3d')

#     # x1, x2 = np.meshgrid(np.linspace(-2,2,50), np.linspace(-2,2,50))
#     # z = w[0] + w[1] * x1 + w[2] * x2 + w[3] * (x1 ** 2) + w[4] * (x2 ** 2) + w[5] * x1 * x2
#     # z = np.clip(z, -1, 1)
#     # print(np.min(z), np.max(z))
#     blue_indices = (class_y < 0.0)
#     red_indices = (class_y > 0.0)
#     ax2d.contourf(x1, x2, np.clip(z, -1, 1), levels=np.linspace(-1,1,20),cmap="coolwarm", alpha=.2)
#     ax2d.contour(x1, x2, np.clip(z, -1, 1), levels=[0.0], colors=["k"])
#     ax2d.plot(coords_x1[blue_indices], coords_x2[blue_indices], 'bo')
#     ax2d.plot(coords_x1[red_indices], coords_x2[red_indices], 'ro')


#     ax2d.set_xlim((-1.5,1.5))
#     ax2d.set_ylim((-1.5,1.5))

#     ax3d.set_xlim((-1.5,1.5))
#     ax3d.set_ylim((-1.5,1.5))
#     ax3d.set_zlim((-5,1))

#     # x1, x2 = np.meshgrid(np.linspace(-1.5,1.5,50), np.linspace(-1.5,1.5,50))
#     # z = w[0] + w[1] * x1 + w[2] * x2 + w[3] * (x1 ** 2) + w[4] * (x2 ** 2) + w[5] * x1 * x2
#     ax3d.plot_surface(x1, x2, z, cmap=cm.YlGn, alpha=0.2)  # Plot contour curves
#     ax3d.contourf(x1, x2, np.clip(z, -1, 1)*0.01, levels=[-1*0.01,0,1*0.01],colors=["b","r"], alpha=.3)
#     ax3d.contour(x1, x2, z, levels=[0.0], colors=["k"])
#     #plt.show()
#     # pause = input("Press Enter to continue...")
#     # x1_est = np.random.normal(0, 0.5, 200) 
#     # x2_est = np.random.normal(0, 0.5, 200)
#     # y_est = fun(x1_est,x2_est,w)
#     # ax3d.scatter(x1_est,x2_est,y_est, c = list(map(lambda x: 'm' if x> 0 else 'c', y_est)))