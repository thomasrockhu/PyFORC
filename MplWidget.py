
from PyQt5 import QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.backend_managers import ToolManager
import matplotlib.backend_tools as mb
from matplotlib.figure import Figure
import matplotlib.collections as mc
import matplotlib.path as mp
import matplotlib.widgets as mw
import numpy as np


class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, toolbar_loc='bottom'):
        super().__init__(parent=parent)
        self.sublayout = QtWidgets.QVBoxLayout(self)
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        self.figurecanvas = FigureCanvas(self.fig)
        # self.toolmanager = self.figurecanvas.manager.toolmanager

        if toolbar_loc == 'bottom':
            self.sublayout.addWidget(self.figurecanvas)
            self.sublayout.addWidget(CustomToolbar(self.figurecanvas, self))
        elif toolbar_loc == 'top':
            self.sublayout.addWidget(CustomToolbar(self.figurecanvas, self))
            self.sublayout.addWidget(self.figurecanvas)
        else:
            raise ValueError('Invalid toolbar location: {}'.format(toolbar_loc))

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()
        return


class CustomToolbar(NavigationToolbar2QT):

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)

        self.axes = parent.axes

        ## PyQt way of adding actions
        self.addSeparator()
        lasso_action = self.addAction(QtGui.QIcon('lasso.png'), 'lasso', self.lasso_select)
        lasso_action.setCheckable(True)
        rectangle_action = self.addAction(QtGui.QIcon('rectangle.png'), 'rectangle', self.rectangle_select)
        self._actions.update({'lasso': lasso_action, 'rectangle': rectangle_action})
        rectangle_action.setCheckable(True)
        self._update_buttons_checked()

        return

    # def add_tool(self, *args, **kwargs):
    #     self.toolmanager.add_tool(*args, **kwargs)

    def lasso_select(self):
        if self._active == 'LASSO':
            self.lasso = None
        else:
            self._active = 'LASSO'
            self.lasso = mw.LassoSelector(self.axes, onselect=self.lasso_callback)
        self._update_buttons_checked()
        return

    def rectangle_select(self):
        if self._active == 'RECTANGLE':
            self.lasso = None
        else:
            self._active = 'RECTANGLE'
            self.lasso = mw.RectangleSelector(self.axes, onselect=self.rectangle_callback)
        self._update_buttons_checked()
        return

    def lasso_callback(self, verts):
        path = mp.Path(verts)
        # self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        # self.fc[:, -1] = self.alpha_other
        # self.fc[self.ind, -1] = 1
        # self.collection.set_facecolors(self.fc)
        print('lasso_callback!')
        self.canvas.draw_idle()

    def rectangle_callback(self, eclick, erelease):
        # path = mp.Path(verts)
        print('rectangle_callback')
        self.canvas.draw_idle()

    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        self._actions['pan'].setChecked(self._active == 'PAN')
        self._actions['zoom'].setChecked(self._active == 'ZOOM')
        self._actions['lasso'].setChecked(self._active == 'LASSO')
        self._actions['rectangle'].setChecked(self._active == 'RECTANGLE')
        return