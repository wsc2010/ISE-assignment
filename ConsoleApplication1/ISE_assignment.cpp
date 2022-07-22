
#include <iostream>
#include "highgui/highgui.hpp"
#include "opencv.hpp"
#include "core/core.hpp"

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <string>

using namespace cv;
using namespace std;



Mat RGBtoGrey(Mat RGB)
{
	Mat Grey = Mat::zeros(RGB.size(), CV_8UC1);
	for (int i = 0; i < RGB.rows; i++)
	{
		for (int j = 0; j < RGB.cols * 3; j = j + 3)
		{
			Grey.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
		}
	}
	return Grey;
}

Mat GreytoBinary(Mat Grey, int thresh)
{
	Mat BinaryImage = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			if (Grey.at<uchar>(i, j) > thresh) {
				BinaryImage.at<uchar>(i, j) = 255;
			}
		}
	}
	return BinaryImage;
}

Mat Invert(Mat Grey)
{
	Mat InversionImage = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			InversionImage.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
		}
	}
	return InversionImage;
}

Mat Darken(Mat Grey, int Thresh)
{
	Mat DarkenImage = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			if (Grey.at<uchar>(i, j) <= Thresh)
			{
				DarkenImage.at<uchar>(i, j) = Grey.at<uchar>(i, j);
			}
			else
			{
				DarkenImage.at<uchar>(i, j) = Thresh;
			}
		}
	}
	return DarkenImage;
}


Mat StepFunction(Mat Grey, int Thresh1, int Thresh2)
{
	Mat DarkenImage = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			if (Grey.at<uchar>(i, j) >= Thresh1 && Grey.at<uchar>(i, j) <= Thresh2)
			{
				DarkenImage.at<uchar>(i, j) = 255;
			}
		}
	}
	return DarkenImage;
}

//convolution 3x3 mask max
Mat MaxMask(Mat Grey, int neighbor)
{
	// to apply a dynamic max mask into the image using exclude border 
	Mat MaxImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbor; i < Grey.rows - neighbor; i++) // Exclude the rows top/down
	{
		for (int j = neighbor; j < Grey.cols - neighbor; j++) // Exclude column right / left 
		{
			int max = 0;
			for (int ii = -neighbor; ii <= neighbor; ii++) // to get the neighbors
			{
				for (int jj = -neighbor; jj <= neighbor; jj++) // to get the neighbors
				{
					// my task is to find the max out of the neighbors 
					if (Grey.at<uchar>(i + ii, j + jj) > max)
						max = Grey.at<uchar>(i + ii, j + jj);
				}
			}
			MaxImg.at<uchar>(i, j) = max;
		}
	}
	return MaxImg;
}

Mat MinMask(Mat Grey, int neighbor)
{
	// to apply a dynamic min mask into the image using exclude border 
	Mat MinImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbor; i < Grey.rows - neighbor; i++) // Exclude the rows top/down
	{
		for (int j = neighbor; j < Grey.cols - neighbor; j++) // Exclude column right / left 
		{
			int min = 256;
			for (int ii = -neighbor; ii <= neighbor; ii++) // to get the neighbors
			{
				for (int jj = -neighbor; jj <= neighbor; jj++) // to get the neighbors
				{
					// my task is to find the max out of the neighbors 
					if (Grey.at<uchar>(i + ii, j + jj) < min)
						min = Grey.at<uchar>(i + ii, j + jj);
				}
			}
			MinImg.at<uchar>(i, j) = min;
		}
	}
	return MinImg;
}

Mat Average(Mat Grey, int neighbor)
{
	// to sum up all the 9 (3*3) and divided them by 9 to get the output pixel
	// Mask 3*3 AVG 
	Mat AvgImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbor; i < Grey.rows - neighbor; i++) // Exclude the rows top/down
	{
		for (int j = neighbor; j < Grey.cols - neighbor; j++) // Exclude column right / left 
		{
			int sum = 0;
			for (int ii = -neighbor; ii <= neighbor; ii++) // to get the neighbors
			{
				for (int jj = -neighbor; jj <= neighbor; jj++) // to get the neighbors
				{
					sum += Grey.at<uchar>(i + ii, j + jj);
				}
			}
			AvgImg.at<uchar>(i, j) = sum / ((2 * neighbor + 1)* (2 * neighbor + 1)); // get the average
		}
	}
	return AvgImg;
}

Mat EdgeLaplacian(Mat Grey)
{
	Mat NewImage = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 1; i < Grey.rows - 1; i++)
	{
		for (int j = 1; j < Grey.cols - 1; j++)
		{
			int total = 0;
			int offset = 1;

			total = (-4 * Grey.at<uchar>(i, j) + Grey.at<uchar>(i + offset, j) + Grey.at<uchar>(i - offset, j) + Grey.at<uchar>(i, j + offset) + Grey.at<uchar>(i, j - offset));
			if (total <= 0)
			{
				total = 0;
			}
			else if (total >= 255)
			{
				total = 255;
			}
			NewImage.at<uchar>(i, j) = total;
		}
	}
	return NewImage;
}

Mat EqualizeHistogram(Mat Grey)
{
	// sort out all dark, bright , low contrast , high contrast issues 
	Mat EqImage = Mat::zeros(Grey.size(), CV_8UC1);
	// counting for pixels 0->255
	int count[256] = { 0 };
	// how to count 
	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			count[Grey.at<uchar>(i, j)]++;
		}
	}
	// how do I calculate the probability 
	float prob[256] = { 0.0 };
	// to write the code which is able to fill the prob = count/ total number of pixels 
	for (int i = 0; i < 256; i++)
		prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);

	// calculate the accumulative prob
	float accprob[256] = { 0.0 };
	// write the code which is calculating accpro
	// exp: accprob [2] = accprob[0] + accprob[1] + accprob[2]
	accprob[0] = prob[0];
	for (int i = 1; i < 256; i++)
		accprob[i] = prob[i] + accprob[i - 1];

	int newpixel[256] = { 0 };
	// calculate newpixel = 255 * accprob
	for (int i = 0; i < 256; i++)
		newpixel[i] = 255 * accprob[i];

	// I want to fill EqImge by replacing the old pixels with new pixel that I have 
	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			EqImage.at<uchar>(i, j) = newpixel[Grey.at<uchar>(i, j)];
		}
	}
	return EqImage;
}

Mat SobelVEdges(Mat Grey, int thresh)
{
	//-1     1
	//-2  *  2
	//-1     1
	// if the difference is more than 50 make it white
	// absolute function abs()
	Mat VedgeImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++)
	{
		for (int j = 1; j < Grey.cols - 1; j++)
		{
			int LeftSide = -1 * Grey.at<uchar>(i - 1, j - 1) - 2 * Grey.at<uchar>(i, j - 1) - 1 * Grey.at<uchar>(i + 1, j - 1);
			int RightSide = Grey.at<uchar>(i - 1, j + 1) + 2 * Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1);
			if (abs(LeftSide + RightSide) > thresh)
				VedgeImg.at<uchar>(i, j) = 255;
		}
	}
	return VedgeImg;
}

Mat SobelHEdges(Mat Grey, int thresh)
{
	//-1  -2   -1
	//     *  
	//1    2     1
	// if the difference is more than 50 make it white
	// absolute function abs()
	Mat HedgeImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++)
	{
		for (int j = 1; j < Grey.cols - 1; j++)
		{
			int TopSide = -1 * Grey.at<uchar>(i - 1, j - 1) - 2 * Grey.at<uchar>(i - 1, j) - 1 * Grey.at<uchar>(i - 1, j + 1);
			int DownSide = Grey.at<uchar>(i + 1, j - 1) + 2 * Grey.at<uchar>(i + 1, j) + Grey.at<uchar>(i + 1, j + 1);
			if (abs(TopSide + DownSide) > thresh)
				HedgeImg.at<uchar>(i, j) = 255;
		}
	}


	return HedgeImg;
}

Mat Sobel(Mat Grey, int thresh)
{
	Mat SobelImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++)
	{
		for (int j = 1; j < Grey.cols - 1; j++)
		{
			int LeftSide = -1 * Grey.at<uchar>(i - 1, j - 1) - 2 * Grey.at<uchar>(i, j - 1) - 1 * Grey.at<uchar>(i + 1, j - 1);
			int RightSide = Grey.at<uchar>(i - 1, j + 1) + 2 * Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1);
			int TopSide = -1 * Grey.at<uchar>(i - 1, j - 1) - 2 * Grey.at<uchar>(i - 1, j) - 1 * Grey.at<uchar>(i - 1, j + 1);
			int DownSide = Grey.at<uchar>(i + 1, j - 1) + 2 * Grey.at<uchar>(i + 1, j) + Grey.at<uchar>(i + 1, j + 1);
			int Vedge = abs(LeftSide + RightSide);
			int Hedge = abs(TopSide + DownSide);

			if (Vedge + Hedge >= thresh)
				SobelImg.at<uchar>(i, j) = 255;
		}
	}
	return SobelImg;
}

Mat Dilation(Mat Grey, int Neighbor)
{
	Mat NewImage = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = Neighbor; i < Grey.rows - Neighbor; i++)
	{
		for (int j = Neighbor; j < Grey.cols - Neighbor; j++)
		{
			for (int ii = -Neighbor; ii <= Neighbor; ii++)
			{
				for (int jj = -Neighbor; jj <= Neighbor; jj++)
				{
					if (Grey.at<uchar>(i + ii, j + jj) == 255)
					{
						NewImage.at<uchar>(i, j) = 255;
						break;
					}
				}
			}

		}
	}
	return NewImage;
}

Mat HoriDilation(Mat Grey, int Neighbor)
{
	Mat NewImage = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = Neighbor; i < Grey.rows - Neighbor; i++)
	{
		for (int j = Neighbor; j < Grey.cols - Neighbor; j++)
		{
			for (int jj = -Neighbor; jj <= Neighbor; jj++)
			{
				if (Grey.at<uchar>(i, j + jj) == 255)
				{
					NewImage.at<uchar>(i, j) = 255;
					break;
				}
			}
		}
	}
	return NewImage;
}

Mat VertDilation(Mat Grey, int Neighbor)
{
	Mat NewImage = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = Neighbor; i < Grey.rows - Neighbor; i++)
	{
		for (int j = Neighbor; j < Grey.cols - Neighbor; j++)
		{
			for (int ii = -Neighbor; ii <= Neighbor; ii++)
			{
				if (Grey.at<uchar>(i + ii, j) == 255)
				{
					NewImage.at<uchar>(i, j) = 255;
					break;
				}
			}
		}
	}
	return NewImage;
}

Mat Erosion(Mat Grey, int Neighbor)
{
	Mat NewImage = Grey.clone();

	for (int i = Neighbor; i < Grey.rows - Neighbor; i++)
	{
		for (int j = Neighbor; j < Grey.cols - Neighbor; j++)
		{
			for (int ii = -Neighbor; ii <= Neighbor; ii++)
			{
				for (int jj = -Neighbor; jj <= Neighbor; jj++)
				{
					if (Grey.at<uchar>(i + ii, j + jj) == 0)
					{
						NewImage.at<uchar>(i, j) = 0;
						break;
					}
				}
			}

		}
	}
	return NewImage;
}

Mat HoriErosion(Mat Grey, int Neighbor)
{
	Mat NewImage = Grey.clone();

	for (int i = Neighbor; i < Grey.rows - Neighbor; i++)
	{
		for (int j = Neighbor; j < Grey.cols - Neighbor; j++)
		{
			for (int jj = -Neighbor; jj <= Neighbor; jj++)
			{
				if (Grey.at<uchar>(i, j + jj) == 0)
				{
					NewImage.at<uchar>(i, j) = 0;
					break;
				}
			}
		}
	}
	return NewImage;
}

Mat VertErosion(Mat Grey, int Neighbor)
{
	Mat NewImage = Grey.clone();

	for (int i = Neighbor; i < Grey.rows - Neighbor; i++)
	{
		for (int j = Neighbor; j < Grey.cols - Neighbor; j++)
		{
			for (int ii = -Neighbor; ii <= Neighbor; ii++)
			{
				if (Grey.at<uchar>(i + ii, j) == 0)
				{
					NewImage.at<uchar>(i, j) = 0;
					break;
				}
			}
		}
	}
	return NewImage;
}

Mat Laplacian(Mat Grey, int thresh)
{
	//     1
	// 1  -4  1  
  //       1
	  // if the addition of all the values is more than 50 make it white 
	  // to find the edges in all the directions 
	Mat LaplcianImage = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++)
	{
		for (int j = 1; j < Grey.cols - 1; j++)
		{
			int sum = -4 * Grey.at<uchar>(i, j) + Grey.at<uchar>(i - 1, j) + Grey.at<uchar>(i + 1, j) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i, j + 1);
			if (abs(sum) > thresh)
				LaplcianImage.at<uchar>(i, j) = 255;
		}
	}
	return LaplcianImage;
}

// OTSU Function  - will calculate what is the best TH to split Bg/Fg
int magnfiedOTSU(Mat Grey)
{
	int count[256] = { 0 }; // 0 -> 255
   // write the code which is able to count the pixels 
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			count[Grey.at<uchar>(i, j)]++;

	double prob[256] = { 0.0 };
	// write the code to prob = count / total number of pixels 
	for (int i = 0; i < 256; i++)
		prob[i] = (double)count[i] / (double)(Grey.rows * Grey.cols);

	double theta[256] = { 0.0 };
	// accprob[3] = prob[3] + prob[2]+ prob[1] + prob[0];
	theta[0] = prob[0];
	for (int i = 1; i < 256; i++)
		theta[i] = prob[i] + theta[i - 1];

	// caculate meu - accumulative i * prob[i] 
	float meu[256] = { 0.0 };
	for (int i = 1; i < 256; i++)
		meu[i] = i * prob[i] + meu[i - 1];

	// calculate sigma 
	float Sigma[256] = { 0.0 };
	// apply the OTSU formula to calcualte Sigma
	for (int i = 0; i < 256; i++)
		Sigma[i] = pow((meu[255] * theta[i] - meu[i]), 2) / (theta[i] * (1 - theta[i])); // magnified otsu 

	// find the i which has the Maximum sigma 
	float MaxSigma = -1;
	int OTSUVal = 0;
	for (int i = 0; i < 256; i++)
	{
		if (Sigma[i] > MaxSigma)
		{
			MaxSigma = Sigma[i];
			OTSUVal = i;
		}
	}
	// magnified otsu
	return OTSUVal + 30;
}


float calcWhiteRatio(Mat binaryImg)
{
	return(countNonZero(binaryImg)/(float)(binaryImg.rows*binaryImg.cols));
}


Mat CCL(Mat inputImg) 
{
	Mat blob;
	blob = inputImg.clone();
	vector<vector<Point>> segments;
	vector<Vec4i> hierarchy1;
	findContours(inputImg, segments, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));


	Mat dst = Mat::zeros(inputImg.size(), CV_8UC3);
	if (!segments.empty()) {
		for (int i = 0; i < segments.size(); i++) {
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, segments, i, colour, -1, 8, hierarchy1);
		}
	}
	return dst;
}


Mat cropBorders(Mat binaryPlate)
{
	int * rowWhiteCount = new int[binaryPlate.rows];
	memset(rowWhiteCount, 0, binaryPlate.rows * 4);

	for (int i = 0; i < binaryPlate.rows; i++)
	{
		for (int j = 0; j < binaryPlate.cols; j++)
		{
			if (binaryPlate.at<uchar>(i, j) == 255)
			{
				rowWhiteCount[i]++;
			}
		}
	}

	int startRow = 0;
	int endRow = binaryPlate.rows - 1;
	for (int i = binaryPlate.rows*0.2; i >= 0; i--)
	{
		if (rowWhiteCount[i] / (float)binaryPlate.cols < 0.15) // candidate start point
		{
			startRow = i;
		}
		if (rowWhiteCount[i] / (float)binaryPlate.cols > 0.8) // if scan long white line start row = prev
		{
			if (startRow > i + 1)
			{
				break;
			}
			startRow = i + 1;
			break;
		}
	}

	for (int i = binaryPlate.rows*0.8; i < binaryPlate.rows; i++)
	{
		if (rowWhiteCount[i] / (float)binaryPlate.cols < 0.15)
		{
			endRow = i;
		}
		if (rowWhiteCount[i]/ (float)binaryPlate.cols > 0.8)
		{
			if (endRow < i - 1)
			{
				break;
			}
			endRow = i - 1;
			break;
		}
	}


	int * colWhiteCount = new int[binaryPlate.cols];
	memset(colWhiteCount, 0, binaryPlate.cols * 4);

	for (int j = 0; j < binaryPlate.cols; j++)
	{
		for (int i = 0; i < binaryPlate.rows; i++)
		{
			if (binaryPlate.at<uchar>(i, j) == 255)
			{
				colWhiteCount[j]++;
			}
		}
	}

	int startCol = 0;
	int endCol = binaryPlate.cols - 1;
	for (int j = binaryPlate.cols*0.25; j >= 0 ; j--)
	{
		if (colWhiteCount[j] / (float)binaryPlate.rows < 0.15)
		{
			startCol = j;
		}
		if (colWhiteCount[j] / (float)binaryPlate.rows > 0.79)
		{
			if (startCol > j + 1)
			{
				break;
			}
			startCol = j + 1;
			break;
		}
	}

	for (int j = binaryPlate.cols*0.75; j < binaryPlate.cols; j++)
	{
		if (colWhiteCount[j] / (float)binaryPlate.rows < 0.15)
		{
			endCol = j;
		}
		if (colWhiteCount[j] / (float)binaryPlate.rows > 0.79)
		{
			if (endCol < j - 1)
			{
				break;
			}
			endCol = j - 1;
			break;
		}
	}

	Mat croppedImg = binaryPlate(Rect(Point(startCol, startRow), Point(endCol + 1, endRow + 1)));

	return croppedImg;
}

vector<Mat> splitLayers(Mat binaryPlate)
{
	vector<Mat> plateLayers;

	int * rowWhiteCount = new int[binaryPlate.rows];
	memset(rowWhiteCount, 0, binaryPlate.rows * 4);

	for (int i = 0; i < binaryPlate.rows; i++)
	{
		for (int j = 0; j < binaryPlate.cols; j++)
		{
			if (binaryPlate.at<uchar>(i, j) == 255)
			{
				rowWhiteCount[i]++;
			}
		}
	}

	int splitRow = 0;
	for (int i = binaryPlate.rows*0.4; i < binaryPlate.rows*0.6; i++)
	{
		if (rowWhiteCount[i] / (float)binaryPlate.cols < 0.15)
		{
			splitRow = i;
			break;
		}
	}

	if (splitRow > 0) //split required
	{
		Mat topLayer = binaryPlate(Rect(0, 0, binaryPlate.cols, splitRow));
		plateLayers.push_back(topLayer);
		Mat botLayer = binaryPlate(Rect(0, splitRow, binaryPlate.cols, binaryPlate.rows - splitRow));
		plateLayers.push_back(botLayer);

		imshow("topLayer", plateLayers[0]);
		imshow("botLayer", plateLayers[1]);

		return plateLayers;
	}
	plateLayers.push_back(binaryPlate);
	return plateLayers;
}

vector<Mat> textSegment(vector<Mat> plateLayers)
{
	vector<Mat> segmentText;

	for (int p = 0; p < plateLayers.size(); p++)
	{
		float totalRatio = 0;

		int * colWhiteRatio = new int[plateLayers[p].cols];
		memset(colWhiteRatio, 0, plateLayers[p].cols * 4);

		for (int j = 0; j < plateLayers[p].cols; j++)
		{
			for (int i = 0; i < plateLayers[p].rows; i++)
			{
				if (plateLayers[p].at<uchar>(i, j) == 255)
				{
					colWhiteRatio[j]++;
				}
			}
			totalRatio += (colWhiteRatio[j] / (float)plateLayers[p].rows);
		}

		float avgRatio = totalRatio / plateLayers[p].cols;

		int startCol = 0;
		int splitCol = 0;
		bool inText = false;
		for (int j = 1; j < plateLayers[p].cols; j++)
		{
			if ((colWhiteRatio[j] / (float)plateLayers[p].rows) > (avgRatio / 2) && !inText)
			{
				startCol = j - 1;
				inText = true;
			}

			if ((colWhiteRatio[j] / (float)plateLayers[p].rows) < (avgRatio / 2) && inText)
			{
				splitCol = j;
				segmentText.push_back(plateLayers[p](Rect(startCol, 0, splitCol - startCol, plateLayers[p].rows)));
				startCol = splitCol;
				inText = false;
			}

			if (j == plateLayers[p].cols - 1 && inText)
			{
				splitCol = j;
				segmentText.push_back(plateLayers[p](Rect(startCol, 0, splitCol - startCol, plateLayers[p].rows)));
				startCol = splitCol;
				inText = false;
			}
		}
	}

	//for (int i = 0; i < segmentText.size(); i++)
	//{
	//	imshow(to_string(i), segmentText[i]);
	//}
	
	return segmentText;
}

vector<String> OCR(vector<Mat> plateLayers)
{
	vector<String> textOut;
	

	auto     numOfConfigs = 1;
	auto     **configs = new char *[numOfConfigs];
	configs[0] = (char *) "letters.txt";

	tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
	ocr->Init(NULL, "eng", tesseract::OEM_TESSERACT_LSTM_COMBINED, configs, numOfConfigs, nullptr, nullptr, false);

	ocr->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	for (auto i = plateLayers.begin(); i != plateLayers.end(); ++i)
	{
		Mat invertedImg = Invert(*i);
		copyMakeBorder(invertedImg, invertedImg, 5, 5, 5, 5, BORDER_CONSTANT, 255);

		imshow("invertedImg", invertedImg);

		ocr->SetImage((uchar*)invertedImg.data, invertedImg.size().width, invertedImg.size().height, invertedImg.channels(), invertedImg.step1());

		String outText = ocr->GetUTF8Text();
		outText.erase(std::remove(outText.begin(), outText.end(), '\n'), outText.end());

		textOut.push_back(outText);

	}
	ocr->End();
	delete ocr;

	return textOut;
}

void horiVertProjection(Mat binaryPlate)
{
	int * rowWhiteRatio = new int[binaryPlate.rows];
	memset(rowWhiteRatio, 0, binaryPlate.rows * 4);

	for (int i = 0; i < binaryPlate.rows; i++)
	{
		for (int j = 0; j < binaryPlate.cols; j++)
		{
			if (binaryPlate.at<uchar>(i, j) == 255)
			{
				rowWhiteRatio[i]++;
			}
		}
		cout << rowWhiteRatio[i] << endl;
		cout << "HoriwhiteRatio: " << rowWhiteRatio[i]/(float)binaryPlate.cols << endl;
	}

	Mat HoriProj = Mat::zeros(binaryPlate.size(), CV_8UC1);
	for (int i = 0; i < binaryPlate.rows; i++)
	{
		for (int j = 0; j < rowWhiteRatio[i]; j++)
		{
			HoriProj.at<uchar>(i, j) = 255;
		}
	}
	imshow("HoriProj", HoriProj);



	int * colWhiteRatio = new int[binaryPlate.cols];
	memset(colWhiteRatio, 0, binaryPlate.cols * 4);

	for (int j = 0; j < binaryPlate.cols; j++)
	{
		for (int i = 0; i < binaryPlate.rows; i++)
		{
			if (binaryPlate.at<uchar>(i, j) == 255)
			{
				colWhiteRatio[j]++;
			}
		}
		cout << colWhiteRatio[j] << endl;
		cout << "VertwhiteRatio: " << colWhiteRatio[j] / (float)binaryPlate.rows << endl;
	}

	Mat VertProj = Mat::zeros(binaryPlate.size(), CV_8UC1);
	for (int j = 0; j < binaryPlate.cols; j++)
	{
		for (int i = 0; i < colWhiteRatio[j]; i++)
		{
			VertProj.at<uchar>(i, j) = 255;
		}
	}
	cout << "Total rows: " << binaryPlate.rows << endl;
	cout << "Total cols: " << binaryPlate.cols << endl;

	imshow("VertProj", VertProj);
}

String textRecognition(Mat plate)
{
	imshow("plate", plate);

	Mat binaryImg = GreytoBinary(plate, magnfiedOTSU(plate));
	imshow("binImg", binaryImg);

	Mat croppedImg = cropBorders(binaryImg);
	imshow("croppedImg", croppedImg);

	horiVertProjection(binaryImg); // show vert/hori projections

	//Mat invertedImg = Invert(croppedImg);
	//imshow("invertedImg", invertedImg);

	vector<Mat> plateLayers = splitLayers(croppedImg);
	vector<String> outText = OCR(plateLayers);

	//vector<Mat> plateSegments = textSegment(plateLayers); // testing text Segmentation
	//vector<String> outTextSegmented = OCR(plateSegments);  // testing text Segmentation

	String txt = "";
	for (int i = 0; i < outText.size(); i++)
	{
		txt += outText.at(i);
	}

	cout << "Output Text: " << txt << endl;

	return txt;
}

Rect CCLAndFindCountour(Mat inputImg, Mat greyImg)
{
	Mat blob;
	blob = inputImg.clone();
	vector<vector<Point>> segments;
	vector<Vec4i> hierarchy1;
	findContours(inputImg, segments, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));


	Mat dst = Mat::zeros(inputImg.size(), CV_8UC3);
	if (!segments.empty()) {
		for (int i = 0; i < segments.size(); i++) {
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, segments, i, colour, -1, 8, hierarchy1);
		}
	}

	Rect rect;
	Rect plateRect;
	Scalar black = CV_RGB(0, 0, 0);
	Mat plate;

	String ocrText = "";


	for (int seg = 0; seg < segments.size(); seg++)
	{
		rect = boundingRect(segments[seg]);
		if (rect.width < 40 || rect.height > 70 ||
			rect.y < inputImg.rows*0.3 || rect.br().y > inputImg.rows*(1 - 0.02) || rect.x < inputImg.cols*0.22 || rect.br().x > inputImg.cols*(1 - 0.15) ||
			(rect.br().x > inputImg.cols*0.55 && rect.br().y < inputImg.rows*0.6) || ((rect.x + rect.width) > inputImg.cols*0.65 && rect.y < inputImg.rows*0.72) ||
			(rect.x < inputImg.cols*0.42 && (rect.y + rect.height) > inputImg.rows*0.7) ||
			rect.height / (float)rect.width > 0.9 || rect.height / (float)rect.width < 0.001 ||
			rect.area() < 1000 || rect.area() > 7000)
		{
			drawContours(blob, segments, seg, black, -1, 8, hierarchy1);
		}
		else
		{
			Mat plateCandidate = inputImg(rect);
			cout << calcWhiteRatio(plateCandidate) << endl;
			if (calcWhiteRatio(plateCandidate) <= 0.77)
			{
				drawContours(blob, segments, seg, black, -1, 8, hierarchy1);
			}
			else
			{
				Mat plate = greyImg(rect);


				plateRect = rect;
			}
		}
	}

	imshow("seg", blob);

	return plateRect;
}


void plateExtraction1(Mat image) {
	Mat greyImage = RGBtoGrey(image);
	imshow("greyImage", greyImage);

	Mat eqImage = EqualizeHistogram(greyImage); 

	Mat blurImage = Average(eqImage, 1);
	imshow("blurImage", blurImage);

	Mat verticalSobel = SobelVEdges(blurImage, 150);
	imshow("verticalSobel", verticalSobel);

	Mat dilationVImg = Dilation(verticalSobel, 1);
	imshow("dilationVVVded", dilationVImg);

	Mat erosionVImg = Erosion(dilationVImg, 2);
	imshow("erosionVVVded", erosionVImg);

	Mat dilationV22Img = Dilation(erosionVImg, 2);
	imshow("dilationVVV222ded", dilationV22Img);

	Mat dilationV33Img = Dilation(dilationV22Img, 2);
	imshow("dilationVVV333ded", dilationV33Img);

	Mat dilationV44Img = Dilation(dilationV33Img, 1);
	imshow("dilationVVV333ded", dilationV44Img);

	//Mat blob = CCLAndFindCountour(dilationV44Img, greyImage);
	//imshow("plate", blob);
}

void plateExtraction2(Mat image)
{
	Mat greyImage = RGBtoGrey(image);
	imshow("greyImage", greyImage);

	Mat eqImage = EqualizeHistogram(greyImage);
	imshow("eqImage", eqImage);

	Mat blurImage = Average(eqImage, 2);

	Mat sobelImg = SobelVEdges(blurImage, 50);
	imshow("sobelImg", sobelImg);

	
	Mat dilationImg = Dilation(sobelImg, 3);
	imshow("dilationImg", dilationImg);

	Mat erosionImg = Erosion(dilationImg, 1);
	imshow("erosionImg", erosionImg);

	//Mat blob = CCLAndFindCountour(erosionImg, greyImage);
	//imshow("plate", blob);
}

void plateExtraction3(Mat image)
{
	Mat greyImage = RGBtoGrey(image);
	imshow("greyImg", greyImage);

	Mat dst = greyImage.clone();

	fastNlMeansDenoising(greyImage, dst, 13, 7, 21);
	imshow("denoise", dst);

	Mat sobelImg = SobelVEdges(dst, 170);
	imshow("sobel", sobelImg);

	Mat dilatedImg = Dilation(sobelImg, 2);
	imshow("dilated", dilatedImg);

	Mat dilated2Img = Dilation(dilatedImg, 2);
	imshow("dilated2", dilated2Img);

	Mat erodedImg = Erosion(dilated2Img, 3);

	Mat dilatedErodeImg = Dilation(erodedImg, 3);
	imshow("dilatedErodeImg", dilatedErodeImg);

	Mat eroded2Img = Erosion(dilatedErodeImg, 2);

	Mat eroded3Img = Erosion(eroded2Img, 1);

	imshow("cclseg", CCL(eroded3Img));

	//Mat blob = CCLAndFindCountour(eroded3Img, greyImage);
	//imshow("plate", blob);
}

void plateExtraction4(Mat image)
{
	Mat greyImage = RGBtoGrey(image);
	imshow("greyImg", greyImage);

	Mat dst = greyImage.clone();

	GaussianBlur(greyImage, dst, Size(7, 7), 0, 0);
	imshow("dst", dst);

	Mat sobelImg = SobelVEdges(dst, 100);
	imshow("sobel", sobelImg);

	Mat dilateImg = Dilation(sobelImg, 1);

	Mat dilate2Img = Dilation(dilateImg, 2);
	imshow("dilate2Img", dilate2Img);

	Mat erodedImg = Erosion(dilate2Img, 2);
	imshow("erodedImg", erodedImg);
}

void plateExtraction5(Mat image)
{
	Mat greyImage = RGBtoGrey(image);
	imshow("greyImg", greyImage);

	Mat dst = greyImage.clone();

	bilateralFilter(greyImage, dst, 5, 150, 150);
	imshow("dst", dst);

	Mat sobelImg = SobelVEdges(dst, 100);
	imshow("sobel", sobelImg);

	Mat dilateImg = Dilation(sobelImg, 2);
	imshow("dilateImg", dilateImg);

	Mat erodedImg = Erosion(dilateImg, 1);
	imshow("erodedImg", erodedImg);

	Mat eroded2Img = Erosion(erodedImg, 2);
	imshow("eroded2Img", eroded2Img);

	Mat dilate2Img = Dilation(eroded2Img, 1);
	imshow("dilate2Img", dilate2Img);

	Mat dilate3Img = Dilation(dilate2Img, 2);
	imshow("dilate3Img", dilate3Img);

	Mat eroded3Img = Erosion(dilate3Img, 3);
	imshow("eroded3Img", eroded3Img);

	Mat dilate4Img = Dilation(eroded3Img, 4);
	imshow("dilate4Img", dilate4Img);

	imshow("cclseg", CCL(dilate4Img));

	//Mat blob = CCLAndFindCountour(dilate4Img, greyImage);
	//imshow("plate", blob);

}

void plateExtraction3Copy(Mat image)
{
	Mat greyImage = RGBtoGrey(image);
	imshow("greyImg", greyImage);

	Mat dst = greyImage.clone();

	fastNlMeansDenoising(greyImage, dst, 13, 7, 21);
	imshow("denoise", dst);

	Mat sobelImg = SobelVEdges(dst, 170);
	imshow("sobel", sobelImg);

	Mat dilatedImg = Dilation(sobelImg, 2);
	imshow("dilated", dilatedImg);

	Mat dilated2Img = Dilation(dilatedImg, 2);
	imshow("dilated2", dilated2Img);

	Mat erodedImg = Erosion(dilated2Img, 3);

	Mat dilatedErodeImg = Dilation(erodedImg, 3);
	imshow("dilatedErodeImg", dilatedErodeImg);

	Mat eroded2Img = Erosion(dilatedErodeImg, 2);

	Mat eroded5Img = VertErosion(eroded2Img, 2);

	//Mat eroded3Img = Erosion(eroded2Img, 1);

	//Mat eroded4Img = Erosion(eroded3Img, 1);

	//Mat eroded5Img = VertErosion(eroded4Img, 1);

	imshow("cclseg", CCL(eroded5Img));

	//Mat blob = CCLAndFindCountour(eroded5Img, greyImage);
	//imshow("plate", blob);
}

void plateExtraction6(Mat image)
{
	Mat greyImage = RGBtoGrey(image);
	imshow("greyImg", greyImage);

	Mat dst = greyImage.clone();

	fastNlMeansDenoising(greyImage, dst, 10, 7, 21);
	//imshow("denoise", dst);

	Mat sobelImg = SobelVEdges(dst, 150);
	imshow("sobel", sobelImg);


	Mat dilateImg = Dilation(sobelImg, 2);
	//imshow("dilateImg", dilateImg);

	Mat dilateHoriImg = HoriDilation(dilateImg, 6);
	//imshow("dilateHoriImg", dilateHoriImg);


	Mat erodeImg = Erosion(dilateHoriImg, 7);
	//imshow("erodeImg", erodeImg);

	Mat erodeHImg = HoriErosion(erodeImg, 8);
	//imshow("erodeHImg", erodeHImg);

	Mat DilateVImg = VertDilation(erodeHImg, 8);
	//imshow("DilateVImg", DilateVImg);

	Mat DilateHori2Img = HoriDilation(DilateVImg, 10); //10 or 11
	//imshow("DilateHori2Img", DilateHori2Img);
	

	//imshow("cclseg", CCL(DilateHori2Img));

	Rect plateRect = CCLAndFindCountour(DilateHori2Img, greyImage);
	Mat plate = greyImage(plateRect);

	Mat drawnImg = image.clone();
	rectangle(drawnImg, plateRect, Scalar(0, 255, 0), 1.5, LINE_8);


	String ocrText = textRecognition(plate);
	putText(drawnImg, ocrText, plateRect.tl(), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 255, 0), 2);

	imshow("annotated img", drawnImg);

	//imshow("seg", CCLAndFindCountour(DilateHori2Img, greyImage));
}

int main()
{
	//Mat image; // opencv code
	//image = imread("C:\\Users\\wsc2010\\Desktop\\APU_degree_Y2_sem_2\\ISE\\1.jpg");

	cv::String path("C:\\Users\\wsc2010\\Desktop\\APU_degree_Y2_sem_2\\ISE\\Dataset\\*.jpg"); //select only jpg
	vector<cv::String> fn;
	//vector<cv::Mat> data;
	cv::glob(path, fn, true);
	cout << fn.size();
	for (size_t i = 0; i < fn.size(); i++)
	{
		cv::Mat im = cv::imread(fn[i]);
		if (im.empty()) continue; //only proceed if sucsessful
		cout << fn[i] << endl;
		plateExtraction6(im);
		waitKey();
		destroyAllWindows();
		//data.push_back(im);
	}
}
