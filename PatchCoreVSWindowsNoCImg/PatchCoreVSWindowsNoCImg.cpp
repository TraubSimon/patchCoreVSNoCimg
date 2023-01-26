#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <math.h>       /* exp, ceil */
#include <string>
#include <iomanip>
#include <map>
#include <fstream>
#include <chrono>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/optional_debug_tools.h" 
#include "CImg.h"
using namespace cimg_library;



constexpr int N_NEIGBRHS = 3;
constexpr float PERCENTILE2 = 1;
constexpr int N_CORESET_SAMPLES = 10;

constexpr int N_PIXELS = 4900;
constexpr int PATCH_WIDTH = 224;
constexpr int PATCH_HEIGHT = 224;
constexpr int N_FEATURES_L0 = 16;
constexpr int N_FEATURES_L1 = 24;
constexpr int N_FEATURES_L2 = 24;
constexpr int N_FEATURES_L3 = 40;
constexpr int N_FEATURES_L4 = 40;
constexpr int N_FEATURES_L5 = 40;
constexpr int N_FEATURES_L6 = 48;
constexpr int N_FEATURES_L7 = 48;
constexpr int N_FEATURES_L8 = 96;
constexpr int N_FEATURES_L9 = 96;
constexpr int N_FEATURES_L10 = 96;

int INPUT_IMAGE_WIDTH = 1024;
int INPUT_IMAGE_HEIGHT = 1024;
int INPUT_IMAGE_CHANNELS = 3;

const char* INPUT_IMAGE_PATH_C = "C:\\SensoPart\\AnomalyDetection\\Datasets\\MVTec\\cable\\train\\good\\000trn.png";
const char* INPUT_IMAGE_PATH = "C:\\SensoPart\\AnomalyDetection\\Board\\Code\\inference\\data\\000trn.bmp";
const char* TFLITE_MODEL_PATH = "C:\\SensoPart\\AnomalyDetection\\model_tmp\\withCoresetAndResultScaler.tflite";
//const char* INPUT_IMAGE_PATH = R"(C:\Users\Paul.Hilt\untracked_desktop\source\Data\MVTec_Anomaly_Dataset\cable\train\good\000trn.png)";
//const char* TFLITE_MODEL_PATH = R"(C:\Users\Paul.Hilt\untracked_desktop\source\tensorflow_visor\anomaly_experiments\model_tmp\MobilenetV3_AnomalyDtc_logs_cmp_model_float.tflite)";


struct twoDImgSize
{
    void print(std::string prefix) const
    {
        printf("%s: height: %i and width: %i \n", prefix.c_str(), height, width);
    }
    int height;
    int width;
};


float getXYC(std::vector<float> image, unsigned int x, unsigned int y, unsigned int c, int& imageHeight, int imageWidth, int nChannels)
{
    int idx = c + (x * nChannels) + (y * imageWidth * nChannels);
    return image[idx];
}



int loadTfLiteModel(std::vector<std::vector<Eigen::MatrixXf>>& coresets,
    std::vector<Eigen::MatrixXf>& fMaps,
    std::vector<float>& resultScalers,
    const char* sModelPath,
    const std::vector<std::vector<float>> imagePatches)
    /*
    * @brief:   1. loads tflite model in the model
    *           2. builds an interpreter with the tflite InterpreterBuilder
    *           3. Allocates the Memory for in and output
    *           4. prints input information to console
    *           5. prints output informations to console
    *           6. loads the coreset out of the interpreter
    *           7. loads the fmaps out of the interpreter
    *               7.1 generates for each patch in imagePatches the output of the net
    *               7.2 saves the output of the net in the fMap
    *
    * @param OUT coresets: vector(k) of vector(n) of Eigen::matrices: k: num of layers n: num of coreset samples, Matrices with shape (nPixel, nFeatures) as OUTPUT
    * @param OUT fMaps: vector(k) of Matrices with shape (nPixel, nFeatures) as OUTPUt
    * @param OUT resultScalers: vector<float>(k) which are stored in the model and loaded for the anomaly score as OUTPUT
    * @param IN sModelPath: path to tflite Model
    * @param IN imagePatches: vector with CImgs. num of CImgs= nPatches, CImg shape: (patchHeight, patchWidth, 1, nChannels)
    *
    */
{
    printf("\n DEBUG: loadTfLiteModel() opend \n");
    printf(" DEBUG: Inputs: coresets: vector(%i) < vector(%i) < MatrixXf(rows: %i, cols: %i)>> \n", coresets.size(), coresets[0].size(), coresets[0][0].rows(), coresets[0][0].cols());
    printf(" DEBUG: Inputs: fMaps: vector(%i) < MatrixXf(rows: %i, cols: %i)> \n", fMaps.size(), fMaps[0].rows(), fMaps[0].cols());
    printf(" DEBUG: Inputs: imagePatches: vector(%i) < CImg(height: %i, width: %i)> \n", imagePatches.size(), PATCH_HEIGHT, PATCH_WIDTH);
    printf(" DEBUG: Inputs: resultScalers:vector(%i) \n", resultScalers.size());

    // load the Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(sModelPath);
    if (model == nullptr)
    {
        fprintf(stderr, "failed to load the model \n");
        exit(-1);
    }

    // Build the interprteter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        fprintf(stderr, "Failed to initiate the interpreter \n");
        exit(-1);
    }
    // allocate memory
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Failed to allocate tensor \n");
        exit(-1);
    }

    // INPUT
    std::vector<int> input = interpreter->inputs();
    printf("INPUT \n");
    for (int inputIdx = 0; inputIdx < input.size(); inputIdx++)
    {
        int numOfDims = interpreter->tensor(input[inputIdx])->dims->size;
        TfLiteType type = interpreter->tensor(input[inputIdx])->type;
        TfLiteIntArray* dims = interpreter->tensor(input[inputIdx])->dims;
        printf("  input %i with type %s and dimension: [", inputIdx, TfLiteTypeGetName(type));
        for (int dimIdx = 0; dimIdx < numOfDims; dimIdx++)
        {
            printf(" %i ", dims->data[dimIdx]);
        }
        printf("]\n");
    }

    // OUTPUT
    printf("\nOUTPUT \n");
    std::vector<int> output = interpreter->outputs();
    for (int outputIdx = 0; outputIdx < output.size(); outputIdx++)
    {
        int numOfDims = interpreter->tensor(output[outputIdx])->dims->size;
        TfLiteType type = interpreter->tensor(output[outputIdx])->type;
        TfLiteIntArray* dims = interpreter->tensor(output[outputIdx])->dims;
        printf("  output %i with type %s and dimension: [", outputIdx, TfLiteTypeGetName(type));
        for (int dimIdx = 0; dimIdx < numOfDims; dimIdx++)
        {
            printf(" %i ", dims->data[dimIdx]);
        }
        printf("]\n");
    }


    // get the properties of the layers
    int fMapIdcs[] = { 0, 1 };
    int coresetIdcs[] = { 2, 3 };
    int resultScalerIdcs[] = { 4, 5 };
    const size_t nLayers = fMaps.size();

    // load the coresets
    for (size_t layerIdx = 0; layerIdx < nLayers; layerIdx++)
    {
        TfLiteIntArray* csetDim = interpreter->tensor(output[coresetIdcs[layerIdx]])->dims;
        auto nCrsetSmpls = csetDim->data[0];
        auto nPixels = csetDim->data[1];
        auto nFeatures = csetDim->data[2];
        float* cSet = interpreter->typed_output_tensor<float>(coresetIdcs[layerIdx]);
        size_t idx = 0;


        for (size_t smplIdx = 0; smplIdx < nCrsetSmpls; smplIdx++)
        {
            for (size_t pixelIdx = 0; pixelIdx < nPixels; pixelIdx++)
            {
                for (size_t featurIdx = 0; featurIdx < nFeatures; featurIdx++)
                {
                    idx = (smplIdx * nPixels * nFeatures) + (pixelIdx * nFeatures) + featurIdx;
                    coresets[layerIdx][smplIdx](pixelIdx, featurIdx) = cSet[idx];
                }
            }
        }
    }
    printf("\n DEBUG: Coresets loaded \n");



    for (size_t layerIdx = 0; layerIdx < nLayers; layerIdx++)
    {
        TfLiteIntArray* fMapDim = interpreter->tensor(output[fMapIdcs[layerIdx]])->dims;

        auto layerWidth = fMapDim->data[1];
        auto layerHeight = fMapDim->data[2];
        auto nFeatures = fMapDim->data[3];
        auto nLayerPixels = layerWidth * layerHeight;


        size_t netInputIdx = 0;
        size_t outputIdx = 0;
        size_t fMapIdx = 0;

        for (size_t ptchIdx = 0; ptchIdx < imagePatches.size(); ptchIdx++)
        {
            float* netInput = interpreter->typed_input_tensor<float>(0);
            std::vector<float> curPatch = imagePatches[ptchIdx];

            // generate net output for the curPatch
            interpreter->typed_tensor<float>(input[0]);
            for (size_t elemIdx = 0; elemIdx < curPatch.size(); elemIdx++)
            {
                netInput[elemIdx] = curPatch[elemIdx];
            }


            if (interpreter->Invoke() != kTfLiteOk) {
                printf("Failed to run model! \n");
            }

            // load the fMap
            float* netOutput = interpreter->typed_output_tensor<float>(fMapIdcs[layerIdx]);
            for (size_t pxlIdx = 0; pxlIdx < nLayerPixels; pxlIdx++)
            {
                fMapIdx = pxlIdx + (ptchIdx * nLayerPixels);
                for (size_t ftrIdx = 0; ftrIdx < nFeatures; ftrIdx++)
                {
                    outputIdx = ftrIdx + (pxlIdx * nFeatures);
                    fMaps[layerIdx](fMapIdx, ftrIdx) = netOutput[outputIdx];
                }
            }
        }
    }
    printf("\n DEBUG: fMaps laoded \n");


    // load the result scalers
    for (size_t layerIdx = 0; layerIdx < nLayers; layerIdx++)
    {
        float* netOutput = interpreter->typed_output_tensor<float>(resultScalerIdcs[layerIdx]);
        resultScalers.push_back(netOutput[0]);
    }

    printf("\n DEBUG: loadTfLiteModel() closed \n");

}


Eigen::MatrixXf distanceL1(const std::vector<Eigen::MatrixXf>& coreset, const Eigen::MatrixXf& fMap)
/*
 * @brief calculates L1 Distance between every corset sample and the fMap
 *
 * @param IN coresets: vector(n) of coresets: n=nCoresetSamples and Matrices of shape (row: nPixels, col: nFeatures)
 * @param IN fMaps: Matrix of Shape (row: nPixels, col: nFeatures)
 * @return: L1 Distance as Matrix of shape(row: nCoresetSamples, nPixels)
*/
{
    printf("\n DEBUG: distanceL1() opend \n");
    printf(" DEBUG: Inputs: coreset: vector(%i) < MatrixXf(rows: %i, cols: %i)> \n", coreset.size(), coreset[0].rows(), coreset[0].cols());
    printf(" DEBUG: Inputs: fMap: MatrixXf(rows: %i, cols: %i)> \n", fMap.rows(), fMap.cols());


    int nCoresetSamples = coreset.size();

    Eigen::MatrixXf distMat(nCoresetSamples, fMap.rows());
    for (size_t cSetIdx = 0; cSetIdx < nCoresetSamples; cSetIdx++)
    {
        Eigen::MatrixXf diff = coreset[cSetIdx] - fMap;
        Eigen::MatrixXf absDiff = diff.array().abs();
        Eigen::VectorXf distVec(fMap.rows());
        distVec = absDiff.rowwise().sum();
        distMat.row(cSetIdx) = distVec;
    }

    printf("\n DEBUG: distanceL1() closed \n");
    return distMat;
}

Eigen::MatrixXf distanceL2(const std::vector<Eigen::MatrixXf>& coreset, const Eigen::MatrixXf& fMap)
/*
 * @brief calculates L2 Distance between every corset sample and the fMap
 *
 * @param IN coreset: vector(n): n=nCoresetSamples and Matrices of shape (row: nPixels, col: nFeatures)
 * @param IN fMaps: Matrix of Shape (row: nPixels, col: nFeatures)
 * @return: L2 Distance as Matrix of shape(row: nCoresetSamples, nPixels)
*/
{
    printf("\n DEBUG: distanceL2() opend \n");
    printf(" DEBUG: Inputs: coreset: vector(%i) < MatrixXf(rows: %i, cols: %i)> \n", coreset.size(), coreset[0].rows(), coreset[0].cols());
    printf(" DEBUG: Inputs: fMap: MatrixXf(rows: %i, cols: %i)> \n", fMap.rows(), fMap.cols());
    int nCoresetSamples = coreset.size();

    Eigen::MatrixXf distMat(fMap.rows(), nCoresetSamples);
    for (size_t cSetIdx = 0; cSetIdx < nCoresetSamples; cSetIdx++)
    {
        Eigen::MatrixXf diff = coreset[cSetIdx] - fMap;
        Eigen::MatrixXf sqrDiff = diff.array().square();
        Eigen::VectorXf distVec(fMap.rows());
        distVec = sqrDiff.rowwise().sum();
        distVec = distVec.array().sqrt();
        distMat.col(cSetIdx) = distVec;
    }
    printf("\n DEBUG: distanceL2() closed \n");
    return distMat;
}

Eigen::MatrixXf colMin(const Eigen::MatrixXf& mat)
/*
 * @brief calculate for every column (pixelDimension) the minimal value over all samples in mat (nCoresetSamples)
 *
 * @param IN mat: Matrix with shape: (row: nCoresetSamples, col: nPixels)
 * @return vector with all columnwise minmums: VectorXf of shape (size: nPixels)
*/
{
    printf("\n DEBUG: colMin() opend \n");
    printf(" DEBUG: Inputs: mat: MatrixXf(rows: %i, cols: %i)> \n", mat.rows(), mat.rows());
    printf("\n DEBUG: colMin() closed \n");

    return mat.colwise().minCoeff();
}

Eigen::MatrixXf kNN(const Eigen::MatrixXf& mat, const int k)
/*
 * @brief calculates for every column (pixelDimension) the k minimal values over all samples in mat (nCoresetSamples)
 *
 * @param IN mat: Matrix with shape: (row: nCoresetSamples, col: nPixels)
 * @param IN k: number of coreset Smaples to take
 * @return Matrix with all columnwise k minimal values: Matrix of shape (row: k, column: nPixels)
*/
{
    printf("\n DEBUG: kNN() opend \n");
    printf(" DEBUG: Inputs: mat: MatrixXf(rows: %i, cols: %i)> \n", mat.rows(), mat.cols());
    printf(" DEBUG: Inputs: k=%i \n", k);


    Eigen::MatrixXf sortedMat(mat.rows(), k);
    // sort all entries in each column of the matrix
    for (size_t i = 0; i < mat.rows(); ++i)
    {
        Eigen::VectorXf vec = mat.row(i);
        std::partial_sort(vec.data(), vec.data() + k, vec.data() + vec.size());
        sortedMat.row(i) = vec.head(k);
    }

    printf("\n DEBUG: kNN() closed \n");
    // only take the first k neighbours for each pixel induvidually
    return sortedMat;
}


Eigen::VectorXd softMaxFunction(const Eigen::MatrixXf& input)
/*
 * @brief takes the minimal values of the input matrix along the k nearest neighbours and multplies it
 * with the softmax
 *
 * @param IN input: Matrix with shape: (row: k, col: nPixels)
 * @return VectorXf of size:nPixels with the score for every pixel
*/
{
    printf("\n DEBUG: softMaxFunction() opend \n");
    printf(" DEBUG: Inputs: input: MatrixXf(rows: %i, cols: %i)> \n", input.rows(), input.cols());
    // calc the denominaotr: sum over the exp()
    auto cast = input.cast<double>();
    Eigen::MatrixXd softmaxed = cast.array().exp();
    Eigen::VectorXd denoms = softmaxed.rowwise().sum();
    softmaxed = softmaxed.array().colwise() / denoms.array();

    Eigen::VectorXd reduced = softmaxed.rowwise().minCoeff();
    Eigen::VectorXd oneMinus = Eigen::VectorXd::Ones(reduced.size()) - reduced;
    Eigen::VectorXd reducedInputDouble = (input.rowwise().minCoeff()).cast<double>();
    Eigen::VectorXd product = oneMinus.array() * reducedInputDouble.array();

    printf("\n DEBUG: softMaxFunction() closed \n");
    return product;
}

double calcPercentile(Eigen::VectorXd& vec, const float percentile2)
/*
 * @brief takes the value for which $percentile2$ percent are larger than this value
 *
 * @ param IN vec: VectorXf of size=nPixels as anomaly Score for every pixel
 * @ param IN percentile: float constant hyperparameter
 * @ return: float: shows the score of an whole image to be anomalous
*/
{
    printf("\n DEBUG: calcPercentile() opend \n");
    printf(" DEBUG: Inputs: VectorXd(size: %i)> \n", vec.size());
    printf(" DEBUG: Inputs: percetnile2 = %f \n", percentile2);

    int vecIndex = round(percentile2 * vec.size() / 100.0);
    std::partial_sort(std::reverse_iterator(vec.data() + vec.size()), std::reverse_iterator(vec.data() + vecIndex), std::reverse_iterator(vec.data()), std::greater{});

    printf("\n DEBUG: calcPercentile() closed \n");
    return vec(vecIndex);
}

Eigen::VectorXf aggreagteSofmaxeds(std::vector<Eigen::VectorXd>& softmaxeds)
/*
* @brief: returns the elementwise max for each pixel between different layers
*
* @param IN softmaxed: std::vector(k) of  Eigen::Vector(nPixels) with softmaxed values, k=num of selected layers,
* @ return: one Eigen::Vector(nPixels) with the max over multiple(k) layers for each pixel
*/
{
    printf("\n DEBUG: aggreagteSofmaxeds opend \n ");
    printf(" DEBUG: Inputs: softmaxeds: vector(%i) < VectorXd(size: %i)> \n", softmaxeds.size(), softmaxeds[0].size());

    Eigen::VectorXd max = Eigen::VectorXd::Zero(softmaxeds[0].size());

    for (const auto& sofmaxed : softmaxeds)
        max = max.cwiseMax(sofmaxed);


    printf("\n DEBUG: aggreagteSofmaxeds closed \n ");
    return max.cast<float>();
}

float aggregatePercentileds(std::vector<double>& percentileds)
/*
* @brief: max for variable number of layers k
*
* @param IN percentileds: vector(k) of floats, k=numOfLayers
* @ return: one float value which is the maximum ovetr all for k layers
*/
{
    printf("\n DEBUG: aggregatePercentileds opend \n ");
    printf(" DEBUG: Inputs: percentileds: vector(size: %i)<double> \n", percentileds.size());

    double max = 0.0;
    for (const auto& percentiled : percentileds)
    {
        if (max < percentiled)
            max = percentiled;
    }

    printf("\n DEBUG: aggregatePercentileds closed \n ");
    return max;
}

void getAnomalyScore(std::vector<std::vector<Eigen::MatrixXf>>& coresets,
    std::vector<Eigen::MatrixXf>& fMaps,
    std::vector<float>& resultScalers,
    float& anomalyScore,
    Eigen::VectorXf& anomalyMap,
    bool useDistMetricL1)
    /*
    * @brief:   calculate for a given coreset and fMap the anomaly Score and the anomaly Map. Decide wheter to take L1 or L2 norm
    *           1. calculate distances between fMap and each coreset Sample
    *           2. Find k smallest distances between fMap and coreset samples
    *           3. apply softmax
    *           4. Percentile to predict the anomaly Score for the whole image
    *
    * @param IN  fMaps: Vector(k) of Matrixces with output of selected layer in shape (nPixels, nFeatures), k=nLayers
    * @param IN  coresets: vector(k) of vectors(n) of Matrices with the saved coresert with shape (nPixels, nFeatures), k=nLayers, n=nCoresetSamples
    * @param In  useDistMetricL1: true:L1, false L2
    * @param OUT anomalyScore: Vector with nPixels entries
    * @param OUT anomalyMap: float
    *
    */
{
    printf("\n DEBUG: getAnomalyScore() opend \n");
    printf(" DEBUG: Inputs: coresets: vector(%zu) < vector(%zu) < MatrixXf(rows: %i, cols: %i)>> \n", coresets.size(), coresets[0].size(), coresets[0][0].rows(), coresets[0][0].cols());
    printf(" DEBUG: Inputs: fMaps: vector(%zu) < MatrixXf(rows: %i, cols: %i)> \n", fMaps.size(), fMaps[0].rows(), fMaps[0].cols());
    printf(" DEBUG: Inputs: resultScalers:vector(%i) \n", resultScalers.size());
    printf(" DEBUG: Inputs: anomalyMap: VectorXf(size: %i)> \n", anomalyMap.size());


    const size_t nLayers = fMaps.size();
    std::vector<Eigen::VectorXd> softmaxeds;
    std::vector<double> percentileds;

    for (size_t layerIdx = 0; layerIdx < nLayers; layerIdx++)
    {
        // distance calculation
        Eigen::MatrixXf distMat = Eigen::MatrixXf::Zero(coresets[layerIdx].size(), fMaps[layerIdx].rows());
        if (useDistMetricL1)
        {
            distMat = distanceL1(coresets[layerIdx], fMaps[layerIdx]);
        }
        else
        {
            distMat = distanceL2(coresets[layerIdx], fMaps[layerIdx]);
        }

        // kNN
        Eigen::MatrixXf topK = kNN(distMat, N_NEIGBRHS);

        // softmax
        Eigen::VectorXd softmaxed = softMaxFunction(topK);
        softmaxeds.push_back(softmaxed);

        // divide by the resultScaler
        Eigen::VectorXd normalized = softmaxed / resultScalers[layerIdx];

        // percentile
        double percentiled = calcPercentile(normalized, 100.0 - PERCENTILE2);
        percentileds.push_back(percentiled);

    }

    // max over both layers
    anomalyMap = aggreagteSofmaxeds(softmaxeds);
    anomalyScore = aggregatePercentileds(percentileds);

    printf("\n DEBUG: getAnomalyScore() closed \n");
}

void calcNumOfPatches(twoDImgSize imgSize, const twoDImgSize patchSize, twoDImgSize& nPatches)
/*
* @brief: calculates the number of patches in both directions and returns them with nPatchesWidth and nPatchesHeight
*
* @param  IN imgSize: .height and .width of the image
* @param  IN: patchSize: .height and .width of a patch
* @param OUT nPatches: .height and .width as number of patches in each dimension
*/
{
    printf("\n DEBUG: calcNumOfPatches() opend \n");

    nPatches.height = ceil(float(imgSize.height) / float(patchSize.height));
    nPatches.width = ceil(float(imgSize.width) / float(patchSize.width));

    printf("\n DEBUG: calcNumOfPatches() closed \n");
}

void imageToPatches(std::vector<float>& image, const twoDImgSize imgSize, const twoDImgSize nPatches, const twoDImgSize patchSize, std::vector<std::vector<float>>& imagePatches)
/*
* @brief: converts the input image to patches of size (nPatchesWidth x nPatchesHeight) with zero Padding and returns them in imagePatches
*
* @param  IN image: CImg for which patches are calculated
* @param  IN nPatches: .height and .width as number of patches in each dimension
* @param  IN patchSize: .height and .width of a patch
* @param OUT imagePatches: Patches of the image with the defined dimension (nPatchesWidth, nPatchesHeight)
*/
{
    printf("\n DEBUG: imageToPatches() opend \n");
    std::cout << "imgSize.width: " << imgSize.width << " imgSize.height: " << imgSize.height << std::endl;
    std::cout << "nPatches.width: " << nPatches.width << " nPatches.height: " << nPatches.height << std::endl;
    std::cout << "patchSize.width: " << patchSize.width << " patchSize.height: " << patchSize.height << std::endl;

    
    // calculate the difference in size
    int newImageWidth = nPatches.width * patchSize.width;
    int newImageHeight = nPatches.height * patchSize.height;

    int widthDiff = newImageWidth - imgSize.width;
    int heightDiff = newImageHeight - imgSize.height;
    
    std::cout << "newImageWidth: " << newImageWidth << " newImageHeight: " << newImageHeight << std::endl;
    std::cout << "widthDiff: " << widthDiff << " heightDiff: " << heightDiff << std::endl;
    

    auto paddingStartTime = std::chrono::steady_clock:: now();
    // add 0s to right edge
    int idx = 0;
    int numZeros = widthDiff;
    std::vector<float> zerosRight(numZeros, 0.0);
    for (size_t y = 0; y < imgSize.height; y++)
    {
        for (size_t ch = 0; ch < INPUT_IMAGE_CHANNELS; ch++)
        {
            idx = ch + (INPUT_IMAGE_CHANNELS * imgSize.width) + (y * INPUT_IMAGE_CHANNELS * newImageWidth);
            // insert numZeros zeros at idx to the vector image
            image.insert(image.begin() + idx, zerosRight.begin(), zerosRight.end());
        }
    }

    // add 0s to the bottom
    numZeros = INPUT_IMAGE_CHANNELS * heightDiff * newImageWidth;  // 3 = nChannels
    std::vector<float> zerosBottom(numZeros, 0.0);
    image.insert(image.end(), zerosBottom.begin(), zerosBottom.end());
    
    auto paddingEndTime = std::chrono::steady_clock::now();
    std::cout << "Elapsed padding time in microseconds: "
        << std::chrono::duration_cast<std::chrono::microseconds>(paddingEndTime - paddingStartTime).count()  << " µs" << std::endl;


    // patching
    auto patchingStartTime = std::chrono::steady_clock::now();
    
    int ptchX = 0;                                              // horizontal patch ID
    int ptchY = 0;                                              // vertical patch ID
    int ptchIdx = 0;                                            // flatted(row, than column) patch ID
    int rowIdx = 0;                                             // inside an patch how many rows there are in
    int startIdx = 0;                                           // indx in the image where inserting into an patch starts
    int nRowElems = PATCH_WIDTH * INPUT_IMAGE_CHANNELS;         // number of elements from the flattet image which should be copied to a patch
    int endIdx = startIdx + nRowElems;                          // indx in the image where inserting into an patch ends

    int nPatchRows = nPatches.width * nPatches.height * PATCH_HEIGHT;  // how many rows from the image are copied into patches

    // loop over every patch row which is copied from the image to a specific patch
    for (size_t patchRow = 0; patchRow < nPatchRows; patchRow++)
    {
        ptchIdx = ptchX + (ptchY * nPatches.width);
        //std::cout << "Current patch: " << ptchIdx << " (" << ptchX << "," << ptchY << ")"
        //    " and start and end idxs: " << startIdx << "-" << endIdx << std::endl;

        std::vector<float>& curPatch = imagePatches[ptchIdx];
        curPatch.insert(curPatch.end(), image.begin() + startIdx, image.begin() + endIdx);

        // update startIdx and endIdx
        startIdx = endIdx;
        endIdx = startIdx + nRowElems;
        ptchX++;

        // exit patch at the right boarder
        if (ptchX >= nPatches.height)
        {
            ptchX = 0;
            rowIdx++;
        } 
        // added a comment

        // exit patch at the bottom boarder
        if (rowIdx >= PATCH_HEIGHT)
        {
            rowIdx = 0;
            ptchY++;
        }
    }
    auto patchingEndTime = std::chrono::steady_clock::now();
    std::cout << "Elapsed patching time in microseconds: "
        << std::chrono::duration_cast<std::chrono::microseconds>(patchingEndTime - patchingStartTime).count() << " µs" << std::endl;

    printf("\n DEBUG: imageToPatches() closed \n");
}

std::vector<float> addBlackBoarder(std::vector<float>& imagePatch, const int boarderWidth)
/*
* @brief:   adds an black boarder to an imagePatch at each size and returns an image which has in each direction
*           2*boarderWidth pixels more
*
* @param  IN boarderWidth: int defines how many pixels are added in each direction
* @param  IN: imagePatch: CImg of a patch where the black boarders should be added
*
* @ return: CImg with larger size and black boarder
*/
{
    int patchWidth = PATCH_WIDTH;
    int patchHeight = PATCH_HEIGHT;
    int newWidth = 2 * boarderWidth + patchWidth;
    int newHeight = 2 * boarderWidth + patchHeight;
    // ToDo std::vector<flaot>  newImage(newWidth, newHeight, 1, 3, 0.0);
    std::vector<float>  newImage;
    for (size_t y = 0; y < patchWidth; y++)
    {
        for (size_t x = 0; x < patchHeight; x++)
        {
            size_t newX = x + boarderWidth;
            size_t newY = y + boarderWidth;
            for (size_t i = 0; i < 3; i++)
            {
                // ToDonewImage(newX, newY, 0, i) = imagePatch(x, y, 0, i);
            }
        }
    }
    return newImage;
}

void displayPatches(std::vector<std::vector<float>>& imagePatches, twoDImgSize nPatches)
/*
* @brief: displays all the patches with added black boarders
*
* @param  IN imagePatches: all image patches of an image as vector of CImgs
* @param  IN nPatches: .height and .width as num of pixels in a patch
*
*/
{
    printf("\n DEBUG: displayPatches() opend \n");
    std::vector<float> grid;
    std::vector<float> col;
    size_t idx = 0;
    for (size_t y = 0; y < nPatches.height; y++)
    {
        col.clear();
        for (size_t x = 0; x < nPatches.width; x++)
        {
            idx = (x * nPatches.height) + y;
            std::vector<float> patch = addBlackBoarder(imagePatches[idx], 5);
            // ToDo col.append(patch, 'y');
        }
        // ToDo grid.append(col, 'x');
    }

    printf("\n DEBUG: displayPatches() closed \n");
}

void patchesToImage(std::vector<float>& image, std::vector<std::vector<float>>& imagePatches, twoDImgSize nPatches)
/*
* @brief: puts all patches together in one image
*
* @param  IN imagePatches: vector of patches
* @param  IN nPatches: .height and .width as num of pixels in a patch
* @param OUT image: all patches added together as one image
*/
{
    printf("\n DEBUG: patchesToImage() opend \n");

    image.clear();
    std::vector<float> col;
    size_t idx = 0;
    for (size_t x = 0; x < nPatches.width; x++)
    {
        col.clear();
        for (size_t y = 0; y < nPatches.height; y++)
        {
            idx = (x * nPatches.height) + y;
            std::vector<float> patch = imagePatches[idx];
            // ToDo col.append(patch, 'y');
        }
        // ToDo image.append(col, 'x');
    }
    printf("\n DEBUG: patchesToImage() closed \n");
}

std::vector<float> mobileNetPreprocessing(std::vector<uint8_t>& img)
/*
* @brief normalizes the image by dividing by 172.5 and subtracting -1.0
*
* @param IN/OUT img: CImg in in range[0, 255] out in range [-1.0, 1.0]
*/
{
    printf("\n DEBUG: mobileNetPreprocessing() opend \n");
    std::vector<float> floatImg;
    float value;
    for (uint8_t& elem : img)
    {
        value = float(elem) / 127.5 - 1.0;
        floatImg.push_back(float(elem) / 127.5 - 1.0);
    }

    for (size_t x = 0; x < INPUT_IMAGE_WIDTH; x++)
    {
        for (size_t y = 0; y < INPUT_IMAGE_HEIGHT ; y++)
        {
            for (size_t clr = 0; clr < 3; clr++)
            {
                // ToDo img.atXYZC(x, y, 0, clr) = img.atXYZC(x, y, 0, clr) / 127.5 - 1.0;
            }
        }
    }
    printf("\n DEBUG: mobileNetPreprocessing() closed \n");
    return floatImg;
}

void displayAnomlayMap(Eigen::VectorXf& anomalyMap, twoDImgSize nPatches)
/*
* @brief: displays te anomaly map as Cimg
*
* @param IN anomalyMap: Eigen::VectorXf with size=numOfPixels
* @param IN nPatches: height and width shows the number of patches the map consists of
*/
{
    size_t patchWidth = 14;
    size_t patchHeight = 14;
    size_t nPtchPixels = patchWidth * patchHeight;
    size_t anomalyMapImgWidth = patchWidth * nPatches.width;
    size_t anomalyMapImgHeight = patchHeight * nPatches.height;
    size_t vecIdx = 0;
    size_t ptchIdx = 0;
    size_t xImxIdx = 0;
    size_t yImgIdx = 0;

    // ToDo std::vector<float> anomalyMapImg(70, 70, 1, 1);
    printf("\n DEBUG: displayAnomlayMap() opend \n");
    for (size_t xPtchIdx = 0; xPtchIdx < nPatches.width; xPtchIdx++)
    {
        for (size_t yPtchIdx = 0; yPtchIdx < nPatches.height; yPtchIdx++)
        {
            ptchIdx = (yPtchIdx * nPatches.width) + xPtchIdx;
            int counter = 0;
            for (size_t y = 0; y < patchHeight; y++)
            {
                for (size_t x = 0; x < patchWidth; x++)
                {
                    vecIdx = x + (y * patchWidth) + (nPtchPixels * ptchIdx);
                    xImxIdx = x + (xPtchIdx * patchWidth);
                    yImgIdx = y + (yPtchIdx * patchHeight);
                    // ToDo anomalyMapImg.atXYZC(xImxIdx, yImgIdx, 0, 0) = anomalyMap[vecIdx];
                }
            }
        }
    }

    printf("\n DEBUG: displayAnomlayMap() closed \n");
}

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
    int height, int channels, bool top_down) {
    std::vector<uint8_t> output(height * width * channels);
    for (int i = 0; i < height; i++) {
        int src_pos;
        int dst_pos;

        for (int j = 0; j < width; j++) {
            if (!top_down) {
                src_pos = ((height - 1 - i) * row_size) + j * channels;
            }
            else {
                src_pos = i * row_size + j * channels;
            }

            dst_pos = (i * width + j) * channels;

            switch (channels) {
            case 1:
                output[dst_pos] = input[src_pos];
                break;
            case 3:
                // BGR -> RGB
                output[dst_pos] = input[src_pos + 2];
                output[dst_pos + 1] = input[src_pos + 1];
                output[dst_pos + 2] = input[src_pos];
                break;
            case 4:
                // BGRA -> RGBA
                output[dst_pos] = input[src_pos + 2];
                output[dst_pos + 1] = input[src_pos + 1];
                output[dst_pos + 2] = input[src_pos];
                output[dst_pos + 3] = input[src_pos + 3];
                break;
            default:
                std::cout << "Unexpected number of channels: " << channels << std::endl;
                break;
            }
        }
    }
    return output;
}

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
    int* height, int* channels) {
    int begin, end;

    std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
    if (!file) {
        std::cout << "input file " << input_bmp_name << " not found" << std::endl;
        exit(-1);
    }

    begin = file.tellg();
    file.seekg(0, std::ios::end);
    end = file.tellg();
    size_t len = end - begin;

    std::cout << "len: " << len << std::endl;

    std::vector<uint8_t> img_bytes(len);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(img_bytes.data()), len);
    const int32_t header_size =
        *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
    *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
    *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
    const int32_t bpp =
        *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
    *channels = bpp / 8;

    std::cout << "width, height, channels: " << *width << ", " << *height
        << ", " << *channels << std::endl;

    // there may be padding bytes when the width is not a multiple of 4 bytes
    // 8 * channels == bits per pixel
    const int row_size = (8 * *channels * *width + 31) / 32 * 4;

    // if height is negative, data layout is top down
    // otherwise, it's bottom up
    bool top_down = (*height < 0);

    // Decode image, allocating tensor once the image size is known
    const uint8_t* bmp_pixels = &img_bytes[header_size];
    return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
        top_down);
}

void compareCImgAndVector(CImg<float> imageC, std::vector<uint8_t> image)
{
    std::vector<float> floatImage;
    for (uint8_t& pixel : image)
    {
        floatImage.push_back(float(pixel));
    }
    
    int y = 131;
    int x = 221;
    std::cout << "(" << y << "," << x << ") ="
              << " R" << (int)imageC(x, y, 0, 0)
              << " G" << (int)imageC(x, y, 0, 1)
              << " B" << (int)imageC(x, y, 0, 2);

    std::cout << " compare: R" << getXYC(floatImage, x, y, 0, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_CHANNELS)
        << " G" << getXYC(floatImage, x, y, 1, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_CHANNELS)
        << " B" << getXYC(floatImage, x, y, 2, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_CHANNELS) << std::endl;

}


int main()
{
    auto mainStartTime = std::chrono::steady_clock::now();
    // 1. load Image
    CImg<float> imageC = CImg<float>(INPUT_IMAGE_PATH_C);
    std::vector<uint8_t> image = read_bmp(INPUT_IMAGE_PATH, &INPUT_IMAGE_WIDTH, &INPUT_IMAGE_HEIGHT, &INPUT_IMAGE_CHANNELS);
    twoDImgSize imgSize{INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH }; // swapped imange dimensions
    

    // 2. Preprocess the image
    twoDImgSize patchSize{ PATCH_HEIGHT, PATCH_WIDTH };
    twoDImgSize nPatches;
    
    std::vector<float> floatImage = mobileNetPreprocessing(image);
    calcNumOfPatches(imgSize, patchSize, nPatches);
    std::vector<std::vector<float>> imagePatches(nPatches.width * nPatches.height); // initialize the vector of patches with the number of patches(=nPatches.width * nPatches.height)
    imageToPatches(floatImage, imgSize, nPatches, patchSize, imagePatches);
       

    // 3. load coreset and fmaps
    std::vector<std::vector<Eigen::MatrixXf>> coresets;
    std::vector<Eigen::MatrixXf> fMaps;
    std::vector<float> resultScalers;
    std::vector<int> layerFeatures = { N_FEATURES_L5, N_FEATURES_L6 };

    for (const auto& nFeatures : layerFeatures)  // initialize coreset and fmpas
    {
        fMaps.push_back(Eigen::MatrixXf::Zero(N_PIXELS, nFeatures));
        std::vector<Eigen::MatrixXf> coreset;
        for (size_t i = 0; i < N_CORESET_SAMPLES; i++)
        {
            coreset.push_back(Eigen::MatrixXf::Zero(N_PIXELS, nFeatures));
        }
        coresets.push_back(coreset);
    }
    loadTfLiteModel(coresets, fMaps, resultScalers, TFLITE_MODEL_PATH, imagePatches);


    // 4. calculate Anomaly Score and display the anomalyMap
    float anomalyScore = -1.0;
    Eigen::VectorXf anomalyMap(N_PIXELS);
    getAnomalyScore(coresets, fMaps, resultScalers, anomalyScore, anomalyMap, false);
    std::cout << "anomalyScore: " << anomalyScore << std::endl;
    //displayAnomlayMap(anomalyMap, nPatches);

    auto mainEndTime = std::chrono::steady_clock::now();
    std::cout << "Elapsed padding time in microseconds: "
        << std::chrono::duration_cast<std::chrono::microseconds>(mainEndTime - mainStartTime).count() << " µs" << std::endl;

    return 0;
}

/*
// OPTIONAL
    // display the input image
    if (false)
    {
        CImgDisplay dsp(image, "Image");
        dsp.display(image);
        while (!dsp.is_closed() && !dsp.is_keyESC())
        {
            dsp.wait();
        }
    }

    // check the patches
    if (false)
    {
        displayPatches(imagePatches, nPatches);
    }

    // repatched image
    std::vector<uint8_t> repatchedImg;
    patchesToImage(repatchedImg, imagePatches, nPatches);
    if (false)
    {
        CImgDisplay dsp(repatchedImg, "repatchedImg");
        dsp.display(repatchedImg);
        while (!dsp.is_closed() && !dsp.is_keyESC())
        {
            dsp.wait();
        }
    }
    const unsigned int redColor[] = {255, 0, 0};
    const unsigned int greenColor[] = { 0, 255, 0 };
    const unsigned int blueColor[] = { 0, 0, 255 };
    const unsigned int yellowColor[] = { 0, 255, 255 };
    const unsigned int turkColor[] = { 255, 255, 0 };
    const unsigned int lilaColor[] = { 255, 0, 255 };

    std::vector<uint8_t> image(800, 500, 1, 3, 70);
    image.draw_circle(130, 130, 105, redColor);
    image.draw_circle(400, 130, 105, greenColor);
    image.draw_circle(670, 130, 105, blueColor);
    image.draw_circle(130, 370, 105, yellowColor);
    image.draw_circle(400, 370, 105, turkColor);
    image.draw_circle(670, 370, 105, lilaColor);
    */


    // Anomaly test map
    /*if (false)
    {
        Eigen::VectorXf anomalyTestMap(4900);
        for (size_t i = 0; i < 25; i++)
        {
            for (size_t j = 0; j < 196; j++)
            {
                anomalyTestMap(i * 196 + j) = 1.0 / 25.0 * i;
            }
        }
        //displayAnomlayMap(anomalyTestMap, nPatches);
    }*/
    /*
    if (false)
    {
        printf("\n\n --------------- ANALYZE DSIATNCES START ------------ \n\n");
        printf("distMat for layer %zu \n", layerIdx);
        printf("topLeft: \n");
        std::cout << distMat.topLeftCorner(10, 4) << std::endl;
        printf("topRight: \n");
        std::cout << distMat.topRightCorner(10, 4) << std::endl;
        printf("bottomLeft: \n");
        std::cout << distMat.bottomLeftCorner(10, 4) << std::endl;
        printf("bottomRightl: \n");
        std::cout << distMat.bottomRightCorner(10, 4) << std::endl;
        printf("\n\n --------------- ANALYZE DSIATNCES END------------ \n\n");
    }

    if (false)
    {
        printf("\n\n --------------- ANALYZE topK START ------------ \n\n");
        printf("topK for layer %zu \n", layerIdx);
        printf("topLeft: \n");
        std::cout << topK.topLeftCorner(3, 10) << std::endl;
        printf("topRight: \n");
        std::cout << topK.topRightCorner(3, 10) << std::endl;
        printf("\n\n --------------- ANALYZE topK END------------ \n\n");
    }

    if (false)
    {
        printf("\n\n --------------- ANALYZE softmaxed START ------------ \n\n");
        printf("softmaxed for layer %i \n", layerIdx);
        printf("Head: \n");
        std::cout << softmaxed.head(10) << std::endl;
        printf("Tail: \n");
        std::cout << softmaxed.tail(10) << std::endl;
        printf("\n\n --------------- ANALYZE softmaxed END------------ \n\n");
    }

    if (false)
    {
        printf("\n\n --------------- ANALYZE NORMALIZED START ------------ \n\n");
        printf("Normalized for layer %i \n", layerIdx);
        printf("Sum for normalized: %f \n", normalized.sum());
        printf("ResultScaler: %f\n", resultScalers[layerIdx]);

        printf("Head: \n");
        std::cout << normalized.head(10) << std::endl;

        printf("segment: \n");
        std::cout << normalized.segment(100, 10) << std::endl;

        printf("segment: \n");
        std::cout << normalized.segment(1000, 10) << std::endl;


        printf("Tail: \n");
        std::cout << normalized.tail(10) << std::endl;
        printf("\n\n --------------- ANALYZE NORMALIZED END------------ \n\n");
    }
    */