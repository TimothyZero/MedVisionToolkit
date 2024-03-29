# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import SimpleITK as sitk

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

import sys

from medtk.data.visulaize.vtk_tools import getRenderOfSrcImageWithClip, getRenderofSeg
from medtk.data.visulaize import MedicalImageViewer


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.typeComboBox1 = QtWidgets.QComboBox(self.centralwidget)
        self.typeComboBox1.setObjectName("typeComboBox1")
        self.gridLayout.addWidget(self.typeComboBox1, 0, 0, 1, 1)

        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.gridLayout.addWidget(self.label1, 0, 1, 1, 1)
        self.label1.setObjectName("label")

        self.lineEdit1 = QtWidgets.QLineEdit(self.centralwidget)
        self.gridLayout.addWidget(self.lineEdit1, 0, 2, 1, 5)
        self.lineEdit1.setObjectName("lineEdit")

        self.pushButton1 = QtWidgets.QToolButton(self.centralwidget)
        self.gridLayout.addWidget(self.pushButton1, 0, 7, 1, 1)
        self.pushButton1.setObjectName("pushButton")

        self.showbotten1 = QtWidgets.QToolButton(self.centralwidget)
        self.gridLayout.addWidget(self.showbotten1, 0, 8, 1, 1)
        self.showbotten1.setObjectName("showbotten")

        # seg

        self.typeComboBox2 = QtWidgets.QComboBox(self.centralwidget)
        self.gridLayout.addWidget(self.typeComboBox2, 1, 0, 1, 1)
        self.typeComboBox2.setObjectName("typeComboBox2")

        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.gridLayout.addWidget(self.label2, 1, 1, 1, 1)
        self.label2.setObjectName("label")

        self.lineEdit2 = QtWidgets.QLineEdit(self.centralwidget)
        self.gridLayout.addWidget(self.lineEdit2, 1, 2, 1, 5)
        self.lineEdit2.setObjectName("lineEdit")

        self.pushButton2 = QtWidgets.QToolButton(self.centralwidget)
        self.gridLayout.addWidget(self.pushButton2, 1, 7, 1, 1)
        self.pushButton2.setObjectName("pushButton")

        self.showbotten2 = QtWidgets.QToolButton(self.centralwidget)
        self.gridLayout.addWidget(self.showbotten2, 1, 8, 1, 1)
        self.showbotten2.setObjectName("showbotten")

        self.widget = QVTKRenderWindowInteractor(self.centralwidget)
        self.gridLayout.addWidget(self.widget, 2, 0, 5, 9)
        self.widget.setObjectName("widget")

        self.typeComboBoxList = [self.typeComboBox1, self.typeComboBox2]
        self.lineEditList = [self.lineEdit1, self.lineEdit2]
        self.showbottenList = [self.showbotten1, self.showbotten2]

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.typeComboBox1.addItems(['nii', 'dcm'])
        self.label1.setText(_translate("MainWindow", "SrcPath"))
        self.pushButton1.setText(_translate("MainWindow", "..."))
        self.showbotten1.setText(_translate("MainWindow", "show"))

        self.typeComboBox2.addItems(['nii', 'dcm'])
        self.label2.setText(_translate("MainWindow", "SegPath"))
        self.pushButton2.setText(_translate("MainWindow", "..."))
        self.showbotten2.setText(_translate("MainWindow", "show"))


class SimpleView(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton1.clicked.connect(lambda: self.choosefile(0))
        self.ui.pushButton2.clicked.connect(lambda: self.choosefile(1))
        self.ui.showbotten1.clicked.connect(lambda: self.showimg(0))
        self.ui.showbotten2.clicked.connect(lambda: self.showimg(1))

        self.RenderWindow = self.ui.widget.GetRenderWindow()
        self.renWinInteractor = self.RenderWindow.GetInteractor()

        # self.RenderWindow.SetSize(450, 300)

        camera = vtk.vtkCamera()
        pos = [0, 0, 1, 1]

        self.render = vtk.vtkRenderer()
        self.render.SetBackground(0.8, 0.8, 0.8)
        self.render.SetActiveCamera(camera)
        self.render.SetViewport(*pos)

        self.RenderWindow.AddRenderer(self.render)
        self.m1 = MedicalImageViewer(self.RenderWindow, self.renWinInteractor, self.render)

        class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

            def __init__(self, parent, m1, *args, **kwargs):
                super(KeyPressInteractorStyle).__init__(*args, **kwargs)
                self.parent = vtk.vtkRenderWindowInteractor()
                if parent is not None:
                    self.parent = parent

                print('key press')
                self.AddObserver("KeyPressEvent", m1.keypressFun)

        self.renWinInteractor.SetInteractorStyle(KeyPressInteractorStyle(None, m1=self.m1))

    def choosefile(self, index):
        if self.ui.typeComboBoxList[index].currentText() == 'nii':
            img_file_path = QFileDialog.getOpenFileName(self)
            self.ui.lineEditList[index].setText(img_file_path[0])

        if self.ui.typeComboBoxList[index].currentText() == 'dcm':
            img_file_path = QFileDialog.getExistingDirectory(self)
            self.ui.lineEditList[index].setText(img_file_path)

    def showimg(self, index):
        spacing = (1.0, 1.0, 1.0)
        if index == 0:
            img_file_path = self.ui.lineEditList[index].text()
            if self.ui.typeComboBoxList[index].currentText() == 'nii':
                itk_img = sitk.ReadImage(img_file_path)
                numpyImage = sitk.GetArrayFromImage(itk_img)
                spacing = itk_img.GetSpacing()

            if self.ui.typeComboBoxList[index].currentText() == 'dcm':
                reader = sitk.ImageSeriesReader()
                dicom = reader.GetGDCMSeriesFileNames(img_file_path)
                reader.SetFileNames(dicom)
                image = reader.Execute()
                numpyImage = sitk.GetArrayFromImage(image)
                spacing = image.GetSpacing()
            print(numpyImage.shape)
            print(spacing)

            # self.render, _, _ = getRenderOfSrcImageWithClip(render=self.render,
            #                                                 renWinInteractor=self.renWinInteractor,
            #                                                 numpyImage_src=numpyImage,
            #                                                 spacing=spacing)
            #
            # self.RenderWindow.AddRenderer(self.render)
            # self.renWinInteractor.SetInteractorStyle(
            #     vtk.vtkInteractorStyleTrackballCamera())
            self.m1.addSrc(numpyImage, spacing)
            self.m1.addGrayScaleSliderToRender()
            self.m1.addSliceToRender()
            self.m1.addCropSliderToRender()
            #
            # self.renWinInteractor.SetInteractorStyle(
            #     vtk.vtkInteractorStyleTrackballCamera())

            self.render.ResetCamera()
            self.RenderWindow.Render()
            self.renWinInteractor.Start()

        if index == 1:
            img_file_path = self.ui.lineEditList[index].text()
            if self.ui.typeComboBoxList[index].currentText() == 'nii':
                itk_img = sitk.ReadImage(img_file_path)
                numpyImage = sitk.GetArrayFromImage(itk_img)
                spacing = itk_img.GetSpacing()

            if self.ui.typeComboBoxList[index].currentText() == 'dcm':
                reader = sitk.ImageSeriesReader()
                dicom = reader.GetGDCMSeriesFileNames(img_file_path)
                reader.SetFileNames(dicom)
                image = reader.Execute()
                numpyImage = sitk.GetArrayFromImage(image)
                spacing = image.GetSpacing()

            # self.render = getRenderofSeg(render=self.render,
            #                              renWinInteractor=self.renWinInteractor,
            #                              renWin=self.RenderWindow,
            #                              numpyImage_segs=[numpyImage],
            #                              spacing=spacing)
            # self.RenderWindow.AddRenderer(self.render)
            self.m1.addSeg(numpyImage, spacing)
            self.RenderWindow.Render()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = SimpleView()
    ui.show()
    sys.exit(app.exec_())
