import numpy as np
from numba import jit
from numpy import cos, sin, dot


class Ellipsoid:

    def __init__(self, coordinates, coefficients, ellipsoidRatio, value, eulerAngles):
        self.coordinates = np.asarray(coordinates)
        self.eulerAngles = np.asarray(eulerAngles)
        self.coefficients = np.asarray(coefficients)
        self.ellipsoidRatio = ellipsoidRatio
        self.value = value
        self.R = self.culculateREuler(eulerAngles)

    def rotate(self, eulerAngles):
        self.eulerAngles = eulerAngles
        self.R = self.culculateREuler(self.eulerAngles)


    def _solveTheEquation(self, coordinates):
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        x0, y0, z0 = self.coordinates
        a, b, c, = self.coefficients
        result = ((x - x0)**2)/a**2 + ((y - y0)**2)/b**2 + ((z - z0)**2)/c**2
        indices = np.nonzero(z < z0/2)
        result[indices] = 0
        return result

    def _convertCoordinstes(self, coordinates):
        coordinates -= self.coordinates
        self.rotateCoordinates(coordinates, self.R)
        coordinates += self.coordinates

    def indicesOfPlacing(self, volume, voxelSize):
        volume = np.copy(volume)
        j, i, k = np.meshgrid(np.arange(volume.shape[0]), np.arange(volume.shape[1]), np.arange(volume.shape[2]))
        indices = np.stack((i, j, k), axis=-1).reshape((-1, 3))
        coordinates = indices*voxelSize
        self._convertCoordinstes(coordinates)
        result = self._solveTheEquation(coordinates)
        mask = (result >= 1)*(result <= self.ellipsoidRatio)
        indices = indices[np.nonzero(mask)]
        return indices[:, 0], indices[:, 1], indices[:, 2]

    def placingInVolume(self, volume, voxelSize):
        indices = self.indicesOfPlacing(volume, voxelSize)
        volume[indices] = self.value
        return volume

    @staticmethod
    def culculateREuler(eulerAngles):
        alpha, beta, gamma = eulerAngles

        Rz_ = np.array([
            [cos(alpha),    -sin(alpha),    0           ],
            [sin(alpha),    cos(alpha),     0           ],
            [0,             0,              1           ]
        ])

        Rx = np.array([
            [1,             0,              0           ],
            [0,             cos(beta),      -sin(beta)  ],
            [0,             sin(beta),      cos(beta)   ]
        ])

        Rz = np.array([
            [cos(gamma),    -sin(gamma),    0           ],
            [sin(gamma),    cos(gamma),     0           ],
            [0,             0,              1           ]
        ])

        Rzx = dot(Rz, Rx)
        R = dot(Rzx, Rz_)
        return R

    @staticmethod
    @jit(nopython=True, cache=True)
    def rotateCoordinates(coordinates, R):
        for i in range(coordinates.shape[0]):
            coordinates[i] = dot(R, coordinates[i])


class Hyperboloid(Ellipsoid):

    def _solveTheEquation(self, coordinates):
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        x0, y0, z0 = self.coordinates
        a, b, c, = self.coefficients
        result = - ((x - x0)**2)/a**2 - ((y - y0)**2)/b**2 + ((z - z0)**2)/c**2
        indices = np.nonzero(z > z0)
        result[indices] = 0
        indices = np.nonzero(z < 20)
        result[indices] = 0
        return result

if __name__ == '__main__':
    voxelSize = 0.4
    volumeShape = np.array((128, 128, 128))
    volumeSize = volumeShape*voxelSize
    volume = np.zeros(volumeShape)
    leftVentricle = Ellipsoid((51.2/2, 51.2/2, 51.2/2.5), (5, 8, 10), 2, 100, (0, np.pi/3, 0))
    volume = leftVentricle.placingInVolume(volume, voxelSize)
    leftVentricle = Hyperboloid((51.2/2, 51.2/2.5, 51.2/1.15), (3.2, 3.2, 10), 4, 100, (0, 0, 0))
    volume = leftVentricle.placingInVolume(volume, voxelSize)


    from PyQt5 import QtGui
    from PyQt5.uic import loadUi
    from pyqtgraph.Qt import mkQApp
    from pyqtgraph import GradientEditorItem, makeRGBA
    from pyqtgraph.opengl import GLBoxItem, GLVolumeItem

    levels = [
        np.min(volume),
        np.max(volume)
    ]

    def gradientChanged():
        global volume, volumeItem, gradientEditor, levels
        indices = np.nonzero(volume == 0)
        listTicks = gradientEditor.listTicks()
        listTicks[0][0].color.setAlpha(0)
        for tick in listTicks[1:]:
            tick[0].color.setAlpha(30)
        lut = gradientEditor.getLookupTable(255)
        volumeColors = np.asarray([makeRGBA(data=slice, lut=lut, levels=levels)[0] for slice in volume])
        # volumeColors[indices, 3] = 0
        volumeItem.setData(volumeColors)

    mkQApp()
    mainWindow = loadUi("UI/volume_visualization.ui")

    volumeItem = GLVolumeItem(None, sliceDensity=5)
    volumeItem.scale(*[voxelSize]*3)
    gradientEditor = GradientEditorItem(orientation='right')
    gradientEditor.sigGradientChangeFinished.connect(gradientChanged)
    mainWindow.graphicsView.addItem(gradientEditor)
    volumeViewWidget = mainWindow.openGLWidget
    volumeViewWidget.addItem(volumeItem)

    volumeViewWidget.setCameraPosition(distance=volumeSize[0]*3)
    volumeViewWidget.pan(*volumeSize/2)

    space_box = GLBoxItem()
    space_box.setSize(*volumeSize)
    # mainWindow.openGLWidget.addItem(space_box)

    mainWindow.show()
    QtGui.QApplication.exec()