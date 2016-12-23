/*
 * util.h
 *
 *  Created on: Nov 9, 2016
 *      Author: tengfeixing
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

struct Annotation
{
	string labelname;
	Rect rect;
	double confidence;
};

struct MatchPair{
	Rect rect_annot;
	Rect rect_det;
	double score;
};
/*
 * 将string用delimiter分割开成若干个sub string
 * */
vector<string> str_split(string str, char delimiter)
{
	vector<string> vstrs;
	while(str.length())
	{
		int index = str.find(delimiter);
		if (index != string::npos)
		{
			vstrs.push_back(str.substr(0, index));
			str = str.substr(index+1);
		}
		else
		{
			vstrs.push_back(str);
			break;
		}
	}
	return vstrs;
}

/*
 * 计算两个矩形的交集
 */
Rect getRectIntersection(Rect rect1, Rect rect2)
{
	Rect rectIntersection;
	int left, top, right, bottom;
	left = max(rect1.x, rect2.x);
	top = max(rect1.y, rect2.y);
	right = min(rect1.x+rect1.width, rect2.x+rect2.width);
	bottom = min(rect1.y+rect1.height, rect2.y+rect2.height);

	rectIntersection.x = left;
	rectIntersection.y = top;
	rectIntersection.width = max(0, right -left);
	rectIntersection.height = max(0, bottom - top);

	return rectIntersection;
}

double IOU(Rect rect1, Rect rect2)
{
	Rect overlap = getRectIntersection(rect1, rect2);
	if (overlap.width == 0 || overlap.height == 0)
	{
		return 0;
	}

	return (double)overlap.width*overlap.height / (rect1.width*rect1.height + rect2.width*rect2.height - overlap.width*overlap.height);
}

double points_distance(Point pt1, Point pt2)
{
	return sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
}
#endif /* UTIL_H_ */
