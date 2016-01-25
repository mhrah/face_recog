#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <QFile>
#include <QDir>
#include<dir.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/ml.h>
#include <cstdlib>
#include<QFile>
#include <ctype.h>
#include <cstdlib>
#include <QTextStream>
#include <QtCore>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <strings.h>
#include <string.h>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
//Mat training_labels,hogfeat;
QString filename;
using namespace cv;

using namespace std;
Mat training_labels,hogfeat;
int  countNu=0;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    for (int  k= 0;k  < 40; k++) {

        filename=QString("dataset\\train\\%1").arg(k);
        QString p=filename;

       QDir dir(p);

       dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
           dir.setSorting(QDir::Size | QDir::Reversed);
           QFileInfoList list = dir.entryInfoList();
        QString fpath;
        Mat temp(list.size(),1,CV_16U);
        Mat feat;
         for (int i = 0; i < list.size(); i++){
             temp.at<LONG>(i,0)=k;
             QFileInfo finfo=list.at(i);
             fpath =finfo.absoluteFilePath();

             Mat m=imread(fpath.toStdString(),1);
             Mat ImageInput=m.clone();
            cv::resize(m,ImageInput,Size(32,32));

             try{


             waitKey(1);
             }catch(...){}

             HOGDescriptor hog;
             vector<float> ders;
             vector<Point> locs;
            
            
            //Change the Block and cell size to achieve optimization
            
             hog.blockSize=Size(16,16);
             hog.blockStride=Size(8,8);
             hog.cellSize=Size(8,8);
             hog.winSize=Size(32,32);

             hog.compute(ImageInput,ders,Size(0,0),Size(0,0),locs);
             feat.create(1,ders.size(),CV_32FC1);
             for (int j = 0; j < ders.size(); j++)
                 feat.at<float>(0,j)=ders.at(j);
              hogfeat.push_back(feat);
                 feat.release();

         }
         training_labels.push_back(temp);

    }try{
    cout<<"data samples are:("<<hogfeat.rows<<","<<hogfeat.cols<<")"<<endl;
    cout<<"data samples are:("<<training_labels.rows<<","<<training_labels.cols<<")"<<endl;
    FileStorage file1("svm_fe.xml",FileStorage::WRITE);
    file1<<"hogfeat"<<hogfeat;
    file1.release();
    FileStorage file2("svm_la.xml",FileStorage::WRITE);
    file2<<"labels"<<training_labels;
    file2.release();
    }
    catch(...){}

}

void MainWindow::on_pushButton_2_clicked()
{

    FileStorage file1("svm_fe.xml",FileStorage::READ);
     file1["hogfeat"]>>hogfeat;
     file1.release();
     FileStorage file2("svm_la.xml",FileStorage::READ);
     file2["labels"]>>training_labels;
     file2.release();
     Mat tr;
     training_labels.convertTo(tr,CV_32FC1);
     CvSVMParams params;
     params.svm_type=CvSVM::C_SVC;

      params.gamma=3;
      params.kernel_type=CvSVM::C;


    CvSVM svm;

    cout<<"initialize svm done datd samples:"<<hogfeat.rows<<"\n"<<"number of features each sample:"<<hogfeat.cols<<"\n"<<"training_lables:("<<training_labels.rows<<","<<training_labels.cols<<")"<<endl;

    svm.train(hogfeat,tr,Mat(),Mat(),params);
     cout<<"training...."<<endl;
     svm.save("svm_orb.xml");
     cout<<"svm rady...."<<endl;

}

void MainWindow::on_pushButton_3_clicked()
{
    double result[42][42];
    ui->tableWidget->setRowCount(41);
        ui->tableWidget->setColumnCount(41);
        ui->tableWidget->setHorizontalHeaderLabels(QString("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,result").split(","));
        ui->tableWidget->setVerticalHeaderLabels(QString("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,result").split(","));
        double sum=0,right=0;
        for(int r=0;r<=41;r++)
                for(int k=0;k<=41;k++)
                {
                    result[k][r]=0;
                    ui->tableWidget->setItem(k,r,new QTableWidgetItem(tr("%1").arg(result[k][r])));
                    ui->tableWidget->show();
                }

    CvSVMParams params;
    params.svm_type=CvSVM::C_SVC;
    params.kernel_type=CvSVM::C;
    params.gamma=3;
    CvSVM svm;
    svm.load("svm_orb.xml");Mat m;
    int i=0;
    for (int ii = 0; ii < 40; ++ii) {
        filename =QString("dataset\\test\\%1").arg(ii);

        QString p=filename;
       QDir dir(p);

       dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
           dir.setSorting(QDir::Size | QDir::Reversed);
           QFileInfoList list = dir.entryInfoList();
        QString fpath;

        for (int jj = 0; jj < list.size(); ++jj) {
            QFileInfo fInfo=list.at(jj);
            fpath =fInfo.absoluteFilePath();
             m=imread(fpath.toStdString(),1);
            Mat ImageInput=m.clone();
           cv::resize(m,ImageInput,Size(32,32));


            imshow("test",ImageInput);
            waitKey(1);

            Mat feat;
            HOGDescriptor hog;
            vector<float> ders;
            vector<Point> locs;
            hog.blockSize=Size(16,16);
            hog.blockStride=Size(8,8);
            hog.cellSize=Size(8,8);
            hog.winSize=Size(32,32);
            hog.compute(ImageInput,ders,Size(0,0),Size(0,0),locs);
            feat.create(1,ders.size(),CV_32FC1);
            for(int j=0;j<ders.size();j++){
             feat.at<float>(0,j)=ders.at(j);
            }
             i=svm.predict(feat);

             cout<<i<<endl;
             sum++;
            int j=ii+1;
            int response=i+1;
                         if(response==j)right++;
              result[j][response]+=1;
              result[41][response]+=1;
              result[j][41]+=1;
                         ui->tableWidget->setItem(j-1,response-1,new QTableWidgetItem(tr("%1").arg(result[j][response])));
                         ui->tableWidget->setItem(40,response-1,new QTableWidgetItem(tr("%1").arg(result[response][response]/result[41][response])));
                         ui->tableWidget->setItem(j-1,40,new QTableWidgetItem(tr("%1").arg(result[j][j]/result[j][41])));
                          ui->tableWidget->setItem(40,40,new QTableWidgetItem(tr("%1").arg(right/sum)));
                         ui->tableWidget->resizeColumnsToContents();
                         ui->tableWidget->resizeRowsToContents();
                         ui->tableWidget->show();
        }

    }


}

void MainWindow::on_pushButton_4_clicked()
{
    QFile f("\\tabel_4.csv");

       if (f.open(QFile::WriteOnly | QFile::Truncate))
       {
           QTextStream data( &f );
           QStringList strList;
          ///put column headers
           strList <<"\" ... \" ";
           for( int c = 0; c < ui->tableWidget->columnCount(); ++c )
           {
               strList <<
                       "\" " +
                       ui->tableWidget->horizontalHeaderItem(c)->data(Qt::DisplayRole).toString() +
                       "\" ";
           }
           data << strList.join(",") << "\n";



           for( int r = 0; r < ui->tableWidget->rowCount(); ++r )
           {
               strList.clear();
               strList <<
                       "\" " +
                       ui->tableWidget->horizontalHeaderItem(r)->data(Qt::DisplayRole).toString() +
                       "\" ";
               for( int c = 0; c < ui->tableWidget->columnCount(); ++c )
               {
                   strList << "\" "+ui->tableWidget->item( r, c )->text()+"\" ";
               }
               data << strList.join( "," )+"\n";
           }
           f.close();
       }

}
