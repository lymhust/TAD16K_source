/*
 * evaluate.h
 *
 *  Created on: Nov 9, 2016
 *      Author: tengfeixing
 */

#ifndef EVALUATE_H_
#define EVALUATE_H_

#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <utility>

#include "util.h"
using namespace std;
using namespace cv;

extern "C" 
{
bool sort_asc(double a, double b);
int matchImagePair(string baseDir, string listFile, string detFile, string annotFile, double confidenceThreshold, double matchScoreThreshold, double& recall ,double& precision);
double evaluate(const char *bDir, const char *lFile, const char *aFile, const char *dFile, const char *rFile, double matchScoreThreshold);
}

bool sort_asc(double a, double b)
{
	return (a < b);
}
int matchImagePair(string baseDir, string listFile, string detFile, string annotFile, double confidenceThreshold, double matchScoreThreshold, double& recall ,double& precision)
{
//#define MATCH_IMAGE_PAIR_SHOW

	ifstream imagelistfile(listFile.c_str());
	ifstream detectionfile(detFile.c_str());
	ifstream annotationfile(annotFile.c_str());
	if (!imagelistfile || !detectionfile || !annotationfile)
	{
		cout << "open file failed." << endl;
		return -1;
	}
	string detstr, annotstr, imagename;
	vector<MatchPair> vMatchPairs, vMatchPairs_image;

	int annotobjnum_total(0);
	int detobjnum_total(0);
	int true_positives_total = 0, false_positives_total = 0, false_negtives_total = 0;
	while(getline(imagelistfile, imagename))
	{
		getline(detectionfile, detstr);
		getline(annotationfile, annotstr);
		if (detstr != imagename || annotstr != imagename)
		{
			cout << "unmatching image pair: " << imagename << " " << annotstr << ", " << detstr << endl;
			return -1;
		}

		vMatchPairs_image.clear();
		getline(detectionfile, detstr);
		getline(annotationfile, annotstr);
		int detobjnum = atoi(detstr.c_str());
		int annotobjnum = atoi(annotstr.c_str());
		annotobjnum_total += annotobjnum;

		vector<Annotation> annotobjs, detobjs;
		for (int i=0; i<annotobjnum; i++)
		{
			getline(annotationfile, annotstr);
			vector<string> vstrs = str_split(annotstr, ' ');
			Rect rect(atoi(vstrs[0].c_str()), atoi(vstrs[1].c_str()), atoi(vstrs[2].c_str()), atoi(vstrs[3].c_str()));
			if(rect.width <= 19 || rect.height <= 19)//neglect small annotations
			{
				//continue;
			}
			double score = atof(vstrs[4].c_str());
			Annotation obj;
			obj.rect = rect;
			obj.confidence = score;
			annotobjs.push_back(obj);
		}
		for (int i=0; i<detobjnum; i++)
		{
			getline(detectionfile, detstr);
			vector<string> vstrs = str_split(detstr, ' ');
			Rect rect(atoi(vstrs[0].c_str()), atoi(vstrs[1].c_str()), atoi(vstrs[2].c_str()), atoi(vstrs[3].c_str()));
			double score = atof(vstrs[4].c_str());
			if (score < confidenceThreshold)
			{
				continue;
			}
			//neglect special detections, e.g. small, improper aspect ratio...
			if (rect.width < 30 || rect.height < 30
					//|| (double)rect.width/rect.height < 0.5 || (double)rect.width/rect.height > 2
					|| rect.y + rect.height/2 > 1300
				)
			{
				//continue;
			}
			Annotation obj;
			obj.rect = rect;
			obj.confidence = score;
			detobjs.push_back(obj);
		}
		detobjnum_total += detobjs.size();

		Mat matchMat = Mat::zeros(annotobjs.size(), detobjs.size(), CV_32F);
		for (int i=0; i<annotobjs.size(); i++)
		{
			for (int j=0; j<detobjs.size(); j++)
			{
				double iou_score = IOU(detobjs[j].rect, annotobjs[i].rect);
				matchMat.at<float>(i, j) = iou_score;
			}
		}

		for(int i=0; i<annotobjs.size(); i++)
		{
			double matchScore = 0;
			Point maxLoc;
			minMaxLoc(matchMat, 0, &matchScore, 0, &maxLoc);

			if (matchScore >= matchScoreThreshold)
			{
				for(int y=0; y<matchMat.rows; y++)
				{
					matchMat.at<float>(y, maxLoc.x) = 0;
				}
				for(int x=0; x<matchMat.cols; x++)
				{
					matchMat.at<float>(maxLoc.y, x) = 0;
				}

				MatchPair matchpair;
				matchpair.rect_annot = annotobjs[maxLoc.y].rect;
				matchpair.rect_det = detobjs[maxLoc.x].rect;
				matchpair.score = detobjs[maxLoc.x].confidence;

				vMatchPairs_image.push_back(matchpair);
				vMatchPairs.push_back(matchpair);

				annotobjs[maxLoc.y].confidence = 0;
				detobjs[maxLoc.x].confidence = 0;
			}
			else
			{
				break;
			}
		}
		for (int i=0; i<annotobjs.size(); )
		{
			if (annotobjs[i].confidence == 0)
			{
				annotobjs.erase(annotobjs.begin() + i);
			}
			else
			{
				i++;
			}
		}
		for (int i=0; i<detobjs.size(); )
		{
			if (detobjs[i].confidence == 0)
			{
				detobjs.erase(detobjs.begin() + i);
			}
			else
			{
				i++;
			}
		}

		true_positives_total += vMatchPairs_image.size();
		false_positives_total += detobjs.size();
		false_negtives_total += annotobjs.size();

#ifdef MATCH_IMAGE_PAIR_SHOW
		Mat image = imread(baseDir + imagename);
		resize(image, image, Size(1024,1024));
		cout << "-------false------" << endl;
		for (int i=0; i<detobjs.size(); i++)
		{
			rectangle(image, detobjs[i].rect, Scalar(255, 0, 0), 2);
			cout << detobjs[i].rect << endl;
		}
		cout << "-------miss------" << endl;
		for (int i=0; i<annotobjs.size(); i++)
		{
			rectangle(image, annotobjs[i].rect, Scalar(0, 0, 255), 2);
			cout << annotobjs[i].rect << endl;
		}
		cout << "-------MatchPairs------" << endl;
		for (int i=0; i<vMatchPairs_image.size(); i++)
		{
			rectangle(image, vMatchPairs_image[i].rect_annot, Scalar(0, 0, 0), 1);
			rectangle(image, vMatchPairs_image[i].rect_det, Scalar(0, 255, 255), 1);
			cout << vMatchPairs_image[i].rect_annot << " " << vMatchPairs_image[i].rect_det << endl;
		}
		cout << endl;

		cout << "annotate objs: " << annotobjnum << endl;
		cout << "detect objs:   " << detobjnum << endl;
		cout << "match objs:    " << vMatchPairs_image.size() << endl;
		cout << "false postive: " << detobjs.size() << endl;
		cout << "false negtive: " << annotobjs.size() << endl;
		cout << imagename << endl;
		cout << endl;
		namedWindow("image", 0);
		//imshow("image", image);
		if (detobjs.size() || annotobjs.size()/**/)
		{
			string dstDir = "/home/tengfeixing/lenovo/UbuntuShare/TrafficSign/tsinghua_100K/benchmark_annotated/other_check/";
			//resize(image, image, Size(1024,1024));

			//false detection
			char message[1024];
			sprintf(message, "false:%d", detobjs.size());
			putText(image,message, Point(image.cols/2, image.rows/2), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 255, 255), 9);
			putText(image,message, Point(image.cols/2, image.rows/2), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 0, 255), 4);

			//miss detection
			sprintf(message, "miss:%d", annotobjs.size());
			putText(image,message, Point(image.cols/2, image.rows/2+50), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 255, 255), 9);
			putText(image,message, Point(image.cols/2, image.rows/2+50), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 0, 255), 4);

			//imwrite(dstDir+imagename, image);
			imshow("image", image);
			waitKey(0);
		}
		//waitKey(0);
#endif
	}

	recall = (double)true_positives_total / annotobjnum_total;
	precision = (double)true_positives_total / detobjnum_total;


	return 0;
}
double evaluate(const char *bDir, const char *lFile, const char *aFile, const char *dFile, const char *rFile, double matchScoreThreshold)
{
	string baseDir(bDir);
	string listFile(lFile);
	string annotFile(aFile);
	string detFile(dFile);
	string rocFileName(rFile);

	ifstream detectionfile(detFile.c_str());
	ofstream rocfile(rocFileName.c_str());
	vector<double> vConfidence;
	string str;
	while(getline(detectionfile, str))
	{
		getline(detectionfile, str);
		int detobjnum = atoi(str.c_str());
		for (int i=0; i<detobjnum; i++)
		{
			getline(detectionfile, str);
			vector<string> vstrs = str_split(str, ' ');
			double score = atof(vstrs[4].c_str());
			vConfidence.push_back(score);
		}
	}
	sort(vConfidence.begin(), vConfidence.end(), sort_asc);
	cout << "vConfidence.size(): " << vConfidence.size() << endl;
	double confidence(0), recall(0), precision(0);
	double confidence_pre(0);
	vector<pair<double, double> > vPrPts;
	for (int i=0; i<vConfidence.size(); i++)
	{
		confidence = vConfidence[i];
		if (confidence <= confidence_pre)
		{
			continue;
		}
		matchImagePair(baseDir, listFile, detFile, annotFile, confidence, matchScoreThreshold, recall, precision);
		if (confidence_pre<0.1)
		{
			confidence_pre = confidence + 0.001;
		}
		else if (confidence_pre<0.99)
		{
			confidence_pre = confidence + 0.01;
		}
		else if (confidence_pre<0.999)
		{
			confidence_pre = confidence + 0.001;
		}
		else
		{
			confidence_pre = confidence + 0.0001;
		}

		vPrPts.push_back(make_pair(precision, recall));
		cout << precision << " " << recall << " " << confidence << endl;
		rocfile << precision << " " << recall << " " << confidence << endl;
		//break;
	}
	double mAP(0);
	for (int i=0; i<(int)vPrPts.size() -1; i++)
	{
		mAP += (vPrPts[i].first + vPrPts[i+1].first) / 2 * (vPrPts[i].second - vPrPts[i+1].second);
	}
	mAP += vPrPts.back().first * vPrPts.back().second;
	cout << "mAP: " << mAP << endl;
	//rocfile << "mAP: " << mAP << endl;
	detectionfile.close();
	rocfile.close();
	return mAP;
}


#endif /* EVALUATE_H_ */
