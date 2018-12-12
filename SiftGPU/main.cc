
#include <SiftGPU.h>

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <GL/gl.h>

using namespace std;
using namespace cv;

class GpuFeatureDetector{

    enum InitStatus{
        INIT_OK,
        INIT_IS_NOT_SUPPORT,
        INIT_VERIFY_FAILED
    };

public:
    GpuFeatureDetector() = default;
    ~GpuFeatureDetector() {
        if(m_siftGpuDetector) delete m_siftGpuDetector;
        if(m_siftGpuMatcher)  delete m_siftGpuMatcher;
    }
    InitStatus create(){
        m_siftGpuDetector = new SiftGPU();

        char* myargv[4] = {"-fo","-1","-v","1"};
        m_siftGpuDetector->ParseParam(4,myargv);
        // Set edge threshold, dog threshold

        if(m_siftGpuDetector->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED){
            cerr << "SiftGPU is not supported!" << endl;
            return InitStatus::INIT_IS_NOT_SUPPORT;
        }

        m_siftGpuMatcher = new SiftMatchGPU();
        m_siftGpuMatcher->VerifyContextGL();

        m_maxMatch = 4096;

        return INIT_OK;
    }

    void detectAndCompute(const Mat &img,Mat &descriptors,vector<KeyPoint> &kpts){

        assert(img.channels() == 3); // RGB

        m_siftGpuDetector->RunSIFT(img.cols,img.rows,img.data,GL_RGB,GL_UNSIGNED_BYTE);
        auto num1 = m_siftGpuDetector->GetFeatureNum();

        vector<float> des(128 * num1);
        vector<SiftGPU::SiftKeypoint> keypoints(num1);
        m_siftGpuDetector->GetFeatureVector(&keypoints[0],&des[0]);

        // Trans to Mat
        Mat m(des);
        descriptors = m.reshape(1,num1).clone();

        for(const SiftGPU::SiftKeypoint &kp : keypoints){
            KeyPoint t(kp.x,kp.y,kp.s,kp.o);
            kpts.push_back(t);
        }     
    }

    void transToRootSift(const cv::Mat &siftFeature,cv::Mat &rootSiftFeature){
        for(int i = 0; i < siftFeature.rows; i ++){
            // Conver to float type
            Mat f;
            siftFeature.row(i).convertTo(f,CV_32FC1);

            normalize(f,f,1,0,NORM_L1); // l1 normalize
            sqrt(f,f); // sqrt-root  root-sift
            rootSiftFeature.push_back(f);
        }
    }

    int gpuMatch(const Mat &des1,const Mat &des2){
        
        m_siftGpuMatcher->SetDescriptors(0,des1.rows,des1.data);
        m_siftGpuMatcher->SetDescriptors(1,des2.rows,des2.data);

        int (*match_buf)[2] = new int[m_maxMatch][2];
        
        auto matchNum = m_siftGpuMatcher->GetSiftMatch(m_maxMatch,match_buf);
        
        delete[] match_buf;
        
        return matchNum;
    }

    int gpuMatch(const Mat &des1,const Mat &des2,vector<DMatch>& matches){
        m_siftGpuMatcher->SetDescriptors(0,des1.rows,(float*)des1.data);
        m_siftGpuMatcher->SetDescriptors(1,des2.rows,(float*)des2.data);

        int (*match_buf)[2] = new int[m_maxMatch][2];
        
        auto matchNum = m_siftGpuMatcher->GetSiftMatch(m_maxMatch,match_buf);

        for(int i = 0 ;i  < matchNum; i ++) {
            DMatch dm(match_buf[i][0],match_buf[i][1],0);
            matches.push_back(dm);
        }

        delete[] match_buf;
        return matchNum;
    }
private:
    SiftGPU *m_siftGpuDetector;
    SiftMatchGPU *m_siftGpuMatcher;

    int m_maxMatch;
};

int main()
{

    /////////////////////////////////////////////////////////////////////
    ///
    /// Opencv extract sift
    ///
    ///////////////////////////////////////////////////////////////////

    // Read image  
    auto detector = cv::xfeatures2d::SIFT::create();

    Mat des;
    vector<KeyPoint> kpts;

    string file1 = "/home/liqiang/Documents/shared/8.jpg";
    auto t = getTickCount();
    auto img = imread(file1);
    detector->detectAndCompute(img,noArray(),kpts,des);
    auto end = static_cast<double>(getTickCount() - t) / getTickFrequency();
    cout << "OpenCV get sift consume:" << end << endl;
    cout << "count:" << kpts.size() << endl;


    ////////////////////////////////////////////////////////////////
    ///
    /// SiftGPU extract sift
    ///
    ///////////////////////////////////////////////////////////////
    // Declare sift and initlize
    SiftGPU sift;
    char* myargv[4] = {"-fo","-1","-v","0"};
    //char* myargv[5] = { "-m", "-s", "-unpa", "1"};
    //char* myargv[4] = {"-fo", "-1", "-cuda", "0"};
    sift.ParseParam(4,myargv);

    // Check hardware is support siftGPU
    int support = sift.CreateContextGL();
    if(support != SiftGPU::SIFTGPU_FULL_SUPPORTED){
        cerr << "SiftGPU is not supported!" << endl;
        return 2;
    }

    auto img1 = imread("/home/liqiang/Documents/shared/3.jpg");
    auto img2 = imread("/home/liqiang/Documents/shared/4.jpg");
    auto img3 = imread("/home/liqiang/Documents/shared/5.jpg");
    auto img4 = imread("/home/liqiang/Documents/shared/6.jpg");
    auto img5 = imread("/home/liqiang/Documents/shared/7.jpg");

    auto f = [&sift](Mat &img,vector<float> &des,vector<SiftGPU::SiftKeypoint> &kpts){
        
        auto t = getTickCount();
        sift.RunSIFT(img.cols,img.rows,img.data,GL_RGB,GL_UNSIGNED_BYTE);
        auto num1 = sift.GetFeatureNum();
        
        des.resize(128 * num1);
        kpts.resize(num1);
        sift.GetFeatureVector(&kpts[0],&des[0]);
        cout << "=======================================" << endl;
        cout << "width x height : " << img.cols << "x" << img.rows << endl;
        cout << "Features count:" << num1 << endl;
        cout << "Extract features,consume:" << static_cast<double>(getTickCount() - t) / getTickFrequency() << endl;
    };


    vector<float> des1,des2,des3,des4,des5;
    vector<SiftGPU::SiftKeypoint> kpts1,kpts2,kpts3,kpts4,kpts5;

    f(img1,des1,kpts1);
    f(img2,des2,kpts2);
    f(img3,des3,kpts3);
    f(img4,des4,kpts4);
    f(img5,des5,kpts5);


    SiftMatchGPU matcher;
    matcher.VerifyContextGL();

    matcher.SetDescriptors(0,kpts1.size(),&des1[0]);
    matcher.SetDescriptors(1,kpts2.size(),&des2[0]);

    int (*match_buf)[2] = new int[kpts1.size()][2];
    t = getTickCount();
    int num_match = matcher.GetSiftMatch(kpts1.size(), match_buf);
    cout << "====================================" << endl;
    cout << "Match keypoints count:" << num_match << endl;
    end = static_cast<double>(getTickCount() - t) / getTickFrequency();

    cout << "Match,consume:" << end << endl;


    ////////////////////////////////////////////////////////////////////
    ///
    /// Test class GpuFeatureDetector
    ///
    ///////////////////////////////////////////////////////////////////

    GpuFeatureDetector fp;
    fp.create();

    Mat des11,des22;
    vector<KeyPoint> kpts11,kpts22;

    fp.detectAndCompute(img1,des11,kpts11);
    fp.detectAndCompute(img2,des22,kpts22);

    vector<DMatch> matches;
    t = getTickCount();
    auto matcheNum = fp.gpuMatch(des11,des22,matches);
    cout << "gpu matche:" <<  static_cast<double>(getTickCount() - t) / getTickFrequency() << endl;
    cout << "gpu match count:" << matcheNum << endl;

    Mat matchImg;
    drawMatches(img1,kpts11,img2,kpts22,matches,matchImg);
    imshow("gpu matches",matchImg);
    

    //////////////////////////////////////////////////////////////////////
    ///
    /// OpenCV extract sift and match
    ///
    ////////////////////////////////////////////////////////////////////
    Mat des111,des222;
    vector<KeyPoint> kpts111,kpts222;
    detector->detectAndCompute(img1,noArray(),kpts111,des111);
    detector->detectAndCompute(img2,noArray(),kpts222,des222);

    auto ov_matcher = DescriptorMatcher::create("FlannBased");
    const float minRatio = 0.8;
    const int k = 2;

    vector<vector<DMatch>> knnMatches;
    vector<DMatch> betterMatches;

    t = getTickCount();
    ov_matcher->knnMatch(des111, des222, knnMatches, k);

    for (size_t i = 0; i < knnMatches.size(); i++) {
        const DMatch &bestMatch = knnMatches[i][0];
        const DMatch &betterMatch = knnMatches[i][1];

        float distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < minRatio)
            betterMatches.push_back(bestMatch);
    }
    cout << "Opencv Match:" << static_cast<double>(getTickCount() - t) / getTickFrequency() << endl;

    Mat ovMatchImg;
    drawMatches(img1,kpts111,img2,kpts222,betterMatches,ovMatchImg);

    imshow("opencv matches",ovMatchImg);

    waitKey();


    return 0;
}
