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
//#include "tensorflow/lite/optional_debug_tools.h" 
//using namespace cimg_library;



constexpr int iN_NEIGBRHS = 3;
constexpr float fPERCENTILE2 = 1;
constexpr int iN_CORESET_SAMPLES = 10;

constexpr int iN_PIXELS = 4900;
constexpr int iPATCH_WIDTH = 224;
constexpr int iPATCH_HEIGHT = 224;
constexpr int iN_FEATURES_L0 = 16;
constexpr int iN_FEATURES_L1 = 24;
constexpr int iN_FEATURES_L2 = 24;
constexpr int iN_FEATURES_L3 = 40;
constexpr int iN_FEATURES_L4 = 40;
constexpr int iN_FEATURES_L5 = 40;
constexpr int iN_FEATURES_L6 = 48;
constexpr int iN_FEATURES_L7 = 48;
constexpr int iN_FEATURES_L8 = 96;
constexpr int iN_FEATURES_L9 = 96;
constexpr int iN_FEATURES_L10 = 96;

int iINPUT_IMAGE_WIDTH = 1024;
int iINPUT_IMAGE_HEIGHT = 1024;
int iINPUT_IMAGE_CHANNELS = 3;

const char* cINPUT_IMAGE_PATH_C = "C:\\SensoPart\\AnomalyDetection\\Datasets\\MVTec\\cable\\train\\good\\000trn.png";
const char* cINPUT_IMAGE_PATH = "C:\\SensoPart\\AnomalyDetection\\Board\\Code\\inference\\data\\000trn.bmp";
const char* cTFLITE_MODEL_PATH = "C:\\SensoPart\\AnomalyDetection\\model_tmp\\MobilenetV3_AnomalyDtc_model__post_int8.tflite";
const char* cTFLITE_CORESET_PATH = "C:\\SensoPart\\AnomalyDetection\\model_tmp\\CoresetModel_model_coresetModel_float.tflite";
//const char* INPUT_IMAGE_PATH = R"(C:\Users\Paul.Hilt\untracked_desktop\source\Data\MVTec_Anomaly_Dataset\cable\train\good\000trn.png)";
//const char* TFLITE_MODEL_PATH = R"(C:\Users\Paul.Hilt\untracked_desktop\source\tensorflow_visor\anomaly_experiments\model_tmp\MobilenetV3_AnomalyDtc_logs_cmp_model_float.tflite)";


struct twoDImgSize
{
    void print(std::string prefix) const
    {
        printf("%s: height: %i and width: %i \n", prefix.c_str(), iHeight, iWidth);
    }
    int iHeight;
    int iWidth;
};


uint8_t getXYC(std::vector<uint8_t> iImage, unsigned int x, unsigned int y, unsigned int c, int& iImageHeight, int iImageWidth, int iNChannels)
{
    int idx = c + (x * iNChannels) + (y * iImageWidth * iNChannels);
    return iImage[idx];
}


void printModelInAndOutput(std::unique_ptr<tflite::Interpreter>& interpreter)
{
    // INPUT
    std::vector<int32_t> iInput = interpreter->inputs();
    printf("INPUT \n");
    for (int inputIdx = 0; inputIdx < iInput.size(); inputIdx++)
    {
        int iNumOfDims = interpreter->tensor(iInput[inputIdx])->dims->size;
        TfLiteType type = interpreter->tensor(iInput[inputIdx])->type;
        TfLiteIntArray* iDims = interpreter->tensor(iInput[inputIdx])->dims;
        printf("  input %i with type %s and dimension: [", inputIdx, TfLiteTypeGetName(type));
        for (int dimIdx = 0; dimIdx < iNumOfDims; dimIdx++)
        {
            printf(" %i ", iDims->data[dimIdx]);
        }
        printf("]\n");
    }

    // OUTPUT
    printf("\nOUTPUT \n");
    std::vector<int32_t> iOutput = interpreter->outputs();
    for (int outputIdx = 0; outputIdx < iOutput.size(); outputIdx++)
    {
        int iNumOfDims = interpreter->tensor(iOutput[outputIdx])->dims->size;
        TfLiteType type = interpreter->tensor(iOutput[outputIdx])->type;
        TfLiteIntArray* iDims = interpreter->tensor(iOutput[outputIdx])->dims;
        printf("  output %i with type %s and dimension: [", outputIdx, TfLiteTypeGetName(type));
        for (int dimIdx = 0; dimIdx < iNumOfDims; dimIdx++)
        {
            printf(" %i ", iDims->data[dimIdx]);
        }
        printf("]\n");
    }
}


int loadTfLiteModel(std::vector<std::vector<Eigen::MatrixXf>>& fCoresets,
    std::vector<Eigen::MatrixXi>& iFMaps,
    std::vector<float>& fResultScalers,
    const std::vector<std::vector<uint8_t>> iImagePatches)
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
    printf(" DEBUG: Inputs: coresets: vector(%i) < vector(%i) < MatrixXf(rows: %i, cols: %i)>> \n", fCoresets.size(), fCoresets[0].size(), fCoresets[0][0].rows(), fCoresets[0][0].cols());
    printf(" DEBUG: Inputs: fMaps: vector(%i) < MatrixXf(rows: %i, cols: %i)> \n", iFMaps.size(), iFMaps[0].rows(), iFMaps[0].cols());
    printf(" DEBUG: Inputs: imagePatches: vector(%i) < CImg(height: %i, width: %i)> \n", iImagePatches.size(), iPATCH_HEIGHT, iPATCH_WIDTH);
    printf(" DEBUG: Inputs: resultScalers:vector(%i) \n", fResultScalers.size());

    // load the Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(cTFLITE_MODEL_PATH);
    std::unique_ptr<tflite::FlatBufferModel> coresetModel = tflite::FlatBufferModel::BuildFromFile(cTFLITE_CORESET_PATH);
    if (model == nullptr || coresetModel == nullptr)
    {
        fprintf(stderr, "failed to load the model or the coresetModel \n");
        exit(-1);
    }

    // Build the interprteter
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::Interpreter> coresetInterpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::ops::builtin::BuiltinOpResolver coresetResolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    tflite::InterpreterBuilder(*coresetModel, coresetResolver)(&coresetInterpreter);
    
    if (interpreter == nullptr || coresetInterpreter == nullptr)
    {
        fprintf(stderr, "Failed to initiate the interpreter \n");
        exit(-1);
    }

    // allocate memory
    if (interpreter->AllocateTensors() != kTfLiteOk && 
        coresetInterpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Failed to allocate tensor \n");
        exit(-1);
    }

    printModelInAndOutput(interpreter);
    printModelInAndOutput(coresetInterpreter);

    std::vector<int32_t> iInput = interpreter->inputs();
    std::vector<int32_t> iOutput = interpreter->outputs();
    std::vector<int32_t> iCoresetOutput = coresetInterpreter->outputs();

    // get the properties of the layers
    int iFMapIdcs[] = { 0, 1 };
    int iCoresetIdcs[] = { 1, 2 };
    int iResultScalerIdcs[] = { 3, 3 };
    const size_t iNLayers = iFMaps.size();

    // load the coresets
    for (size_t layerIdx = 0; layerIdx < iNLayers; layerIdx++)
    {
        TfLiteIntArray* iCsetDim = coresetInterpreter->tensor(iCoresetOutput[iCoresetIdcs[layerIdx]])->dims;
        auto nCrsetSmpls = iCsetDim->data[0];
        auto nPixels = iCsetDim->data[1];
        auto nFeatures = iCsetDim->data[2];
        float* fCSet = coresetInterpreter->typed_output_tensor<float>(iCoresetIdcs[layerIdx]);
        size_t idx = 0;

        for (size_t smplIdx = 0; smplIdx < nCrsetSmpls; smplIdx++)
        {
            for (size_t pixelIdx = 0; pixelIdx < nPixels; pixelIdx++)
            {
                for (size_t featurIdx = 0; featurIdx < nFeatures; featurIdx++)
                {
                    idx = (smplIdx * nPixels * nFeatures) + (pixelIdx * nFeatures) + featurIdx;
                    fCoresets[layerIdx][smplIdx](pixelIdx, featurIdx) = fCSet[idx];
                }
            }
        }
    }
    printf("\n DEBUG: Coresets loaded \n");



    for (size_t layerIdx = 0; layerIdx < iNLayers; layerIdx++)
    {
        TfLiteIntArray* iFMapDim = interpreter->tensor(iOutput[iFMapIdcs[layerIdx]])->dims;

        auto layerWidth = iFMapDim->data[1];
        auto layerHeight = iFMapDim->data[2];
        auto nFeatures = iFMapDim->data[3];
        auto nLayerPixels = layerWidth * layerHeight;

        size_t outputIdx = 0;
        size_t fMapIdx = 0;

        for (size_t ptchIdx = 0; ptchIdx < iImagePatches.size(); ptchIdx++)
        {
            int8_t* iNetInput = interpreter->typed_input_tensor<int8_t>(0);
            std::vector<uint8_t> iCurPatch = iImagePatches[ptchIdx];

            // generate net output for the curPatch
            interpreter->typed_tensor<int>(iInput[0]);
            for (size_t elemIdx = 0; elemIdx < iCurPatch.size(); elemIdx++)
            {
                iNetInput[elemIdx] = iCurPatch[elemIdx];
            }
            printf("Output for patch %i generated \n", ptchIdx);


            if (interpreter->Invoke() != kTfLiteOk) {
                printf("Failed to run model! \n");
            }

            // load the fMap
            int8_t* iNetOutput = interpreter->typed_output_tensor<int8_t>(iFMapIdcs[layerIdx]);
            for (size_t pxlIdx = 0; pxlIdx < nLayerPixels; pxlIdx++)
            {
                fMapIdx = pxlIdx + (ptchIdx * nLayerPixels);
                for (size_t ftrIdx = 0; ftrIdx < nFeatures; ftrIdx++)
                {
                    outputIdx = ftrIdx + (pxlIdx * nFeatures);
                    iFMaps[layerIdx](fMapIdx, ftrIdx) = iNetOutput[outputIdx];
                }
            }
        }
    }
    printf("\n DEBUG: fMaps laoded \n");


    // load the result scalers
    for (size_t layerIdx = 0; layerIdx < iNLayers; layerIdx++)
    {
        float* fNetOutput = coresetInterpreter->typed_output_tensor<float>(iResultScalerIdcs[layerIdx]);
        fResultScalers.push_back(fNetOutput[0]);
    }

    printf("\n DEBUG: loadTfLiteModel() closed \n");
}


Eigen::MatrixXf distanceL1(const std::vector<Eigen::MatrixXf>& fCoreset, const Eigen::MatrixXf& fFMap)
/*
 * @brief calculates L1 Distance between every corset sample and the fMap
 *
 * @param IN coresets: vector(n) of coresets: n=nCoresetSamples and Matrices of shape (row: nPixels, col: nFeatures)
 * @param IN fMaps: Matrix of Shape (row: nPixels, col: nFeatures)
 * @return: L1 Distance as Matrix of shape(row: nCoresetSamples, nPixels)
*/
{
    printf("\n DEBUG: distanceL1() opend \n");
    printf(" DEBUG: Inputs: coreset: vector(%i) < MatrixXf(rows: %i, cols: %i)> \n", fCoreset.size(), fCoreset[0].rows(), fCoreset[0].cols());
    printf(" DEBUG: Inputs: fMap: MatrixXf(rows: %i, cols: %i)> \n", fFMap.rows(), fFMap.cols());


    int iNCoresetSamples = fCoreset.size();

    Eigen::MatrixXf fDistMat(iNCoresetSamples, fFMap.rows());
    for (size_t cSetIdx = 0; cSetIdx < iNCoresetSamples; cSetIdx++)
    {
        Eigen::MatrixXf fDiff = fCoreset[cSetIdx] - fFMap;
        Eigen::MatrixXf fAbsDiff = fDiff.array().abs();
        Eigen::VectorXf fDistVec(fFMap.rows());
        fDistVec = fAbsDiff.rowwise().sum();
        fDistMat.row(cSetIdx) = fDistVec;
    }

    printf("\n DEBUG: distanceL1() closed \n");
    return fDistMat;
}

Eigen::MatrixXf distanceL2(const std::vector<Eigen::MatrixXf>& fCoreset, const Eigen::MatrixXf& fFMap)
/*
 * @brief calculates L2 Distance between every corset sample and the fMap
 *
 * @param IN coreset: vector(n): n=nCoresetSamples and Matrices of shape (row: nPixels, col: nFeatures)
 * @param IN fMaps: Matrix of Shape (row: nPixels, col: nFeatures)
 * @return: L2 Distance as Matrix of shape(row: nCoresetSamples, nPixels)
*/
{
    printf("\n DEBUG: distanceL2() opend \n");
    printf(" DEBUG: Inputs: coreset: vector(%i) < MatrixXf(rows: %i, cols: %i)> \n", fCoreset.size(), fCoreset[0].rows(), fCoreset[0].cols());
    printf(" DEBUG: Inputs: fMap: MatrixXf(rows: %i, cols: %i)> \n", fFMap.rows(), fFMap.cols());
    int iNCoresetSamples = fCoreset.size();

    Eigen::MatrixXf fDistMat(fFMap.rows(), iNCoresetSamples);
    for (size_t cSetIdx = 0; cSetIdx < iNCoresetSamples; cSetIdx++)
    {
        Eigen::MatrixXf fDiff = fCoreset[cSetIdx] - fFMap;
        Eigen::MatrixXf fSqrDiff = fDiff.array().square();
        Eigen::VectorXf fDistVec(fFMap.rows());
        fDistVec = fSqrDiff.rowwise().sum();
        fDistVec = fDistVec.array().sqrt();
        fDistMat.col(cSetIdx) = fDistVec;
    }
    printf("\n DEBUG: distanceL2() closed \n");
    return fDistMat;
}

Eigen::MatrixXf colMin(const Eigen::MatrixXf& fMat)
/*
 * @brief calculate for every column (pixelDimension) the minimal value over all samples in mat (nCoresetSamples)
 *
 * @param IN mat: Matrix with shape: (row: nCoresetSamples, col: nPixels)
 * @return vector with all columnwise minmums: VectorXf of shape (size: nPixels)
*/
{
    printf("\n DEBUG: colMin() opend \n");
    printf(" DEBUG: Inputs: mat: MatrixXf(rows: %i, cols: %i)> \n", fMat.rows(), fMat.rows());
    printf("\n DEBUG: colMin() closed \n");

    return fMat.colwise().minCoeff();
}


Eigen::MatrixXf kNN(const Eigen::MatrixXf& fMat)
/*
 * @brief calculates for every column (pixelDimension) the k minimal values over all samples in mat (nCoresetSamples)
 *
 * @param IN mat: Matrix with shape: (row: nCoresetSamples, col: nPixels)
 * @param IN k: number of coreset Smaples to take
 * @return Matrix with all columnwise k minimal values: Matrix of shape (row: k, column: nPixels)
*/
{
    printf("\n DEBUG: kNN() opend \n");
    printf(" DEBUG: Inputs: mat: MatrixXf(rows: %i, cols: %i)> \n", fMat.rows(), fMat.cols());
    printf(" DEBUG: Inputs: k=%i \n", iN_NEIGBRHS);


    Eigen::MatrixXf fSortedMat(fMat.rows(), iN_NEIGBRHS);
    // sort all entries in each column of the matrix
    for (size_t i = 0; i < fMat.rows(); ++i)
    {
        Eigen::VectorXf fVec = fMat.row(i);
        std::partial_sort(fVec.data(), fVec.data() + iN_NEIGBRHS, fVec.data() + fVec.size());
        fSortedMat.row(i) = fVec.head(iN_NEIGBRHS);
    }

    printf("\n DEBUG: kNN() closed \n");
    // only take the first k neighbours for each pixel induvidually
    return fSortedMat;
}


Eigen::VectorXd softMaxFunction(const Eigen::MatrixXf& fInput)
/*
 * @brief takes the minimal values of the input matrix along the k nearest neighbours and multplies it
 * with the softmax
 *
 * @param IN input: Matrix with shape: (row: k, col: nPixels)
 * @return VectorXf of size:nPixels with the score for every pixel
*/
{
    printf("\n DEBUG: softMaxFunction() opend \n");
    printf(" DEBUG: Inputs: input: MatrixXf(rows: %i, cols: %i)> \n", fInput.rows(), fInput.cols());
    // calc the denominaotr: sum over the exp()
    auto cast = fInput.cast<double>();
    Eigen::MatrixXd dSoftmaxed = cast.array().exp();
    Eigen::VectorXd dDenoms = dSoftmaxed.rowwise().sum();
    dSoftmaxed = dSoftmaxed.array().colwise() / dDenoms.array();

    Eigen::VectorXd dReduced = dSoftmaxed.rowwise().minCoeff();
    Eigen::VectorXd dOneMinus = Eigen::VectorXd::Ones(dReduced.size()) - dReduced;
    Eigen::VectorXd dReducedInputDouble = (fInput.rowwise().minCoeff()).cast<double>();
    Eigen::VectorXd dProduct = dOneMinus.array() * dReducedInputDouble.array();

    printf("\n DEBUG: softMaxFunction() closed \n");
    return dProduct;
}

double calcPercentile(Eigen::VectorXd& dVec)
/*
 * @brief takes the value for which $percentile2$ percent are larger than this value
 *
 * @ param IN vec: VectorXf of size=nPixels as anomaly Score for every pixel
 * @ param IN percentile: float constant hyperparameter
 * @ return: float: shows the score of an whole image to be anomalous
*/
{
    printf("\n DEBUG: calcPercentile() opend \n");
    printf(" DEBUG: Inputs: VectorXd(size: %i)> \n", dVec.size());
    printf(" DEBUG: Inputs: percetnile2 = %f \n", fPERCENTILE2);

    int vecIndex = round((100.0 - fPERCENTILE2) * dVec.size() / 100.0);
    std::partial_sort(std::reverse_iterator(dVec.data() + dVec.size()), std::reverse_iterator(dVec.data() + vecIndex), std::reverse_iterator(dVec.data()), std::greater{});

    printf("\n DEBUG: calcPercentile() closed \n");
    return dVec(vecIndex);
}


Eigen::VectorXf aggreagteSofmaxeds(std::vector<Eigen::VectorXd>& dSoftmaxeds)
/*
* @brief: returns the elementwise max for each pixel between different layers
*
* @param IN softmaxed: std::vector(k) of  Eigen::Vector(nPixels) with softmaxed values, k=num of selected layers,
* @ return: one Eigen::Vector(nPixels) with the max over multiple(k) layers for each pixel
*/
{
    printf("\n DEBUG: aggreagteSofmaxeds opend \n ");
    printf(" DEBUG: Inputs: softmaxeds: vector(%i) < VectorXd(size: %i)> \n", dSoftmaxeds.size(), dSoftmaxeds[0].size());

    Eigen::VectorXd dMax = Eigen::VectorXd::Zero(dSoftmaxeds[0].size());

    for (const auto& sofmaxed : dSoftmaxeds)
        dMax = dMax.cwiseMax(sofmaxed);


    printf("\n DEBUG: aggreagteSofmaxeds closed \n ");
    return dMax.cast<float>();
}


float aggregatePercentileds(std::vector<double>& dPercentileds)
/*
* @brief: max for variable number of layers k
*
* @param IN percentileds: vector(k) of floats, k=numOfLayers
* @ return: one float value which is the maximum ovetr all for k layers
*/
 {
    printf("\n DEBUG: aggregatePercentileds opend \n ");
    printf(" DEBUG: Inputs: percentileds: vector(size: %i)<double> \n", dPercentileds.size());

    float dMax = 0.0;
    for (const auto& percentiled : dPercentileds)
    {
        if (dMax < percentiled)
            dMax = percentiled;
    }

    printf("\n DEBUG: aggregatePercentileds closed \n ");
    return dMax;
}
 

void getAnomalyScore(std::vector<std::vector<Eigen::MatrixXf>>& fCoresets,
    std::vector<Eigen::MatrixXf>& iFMaps,
    std::vector<float>& fResultScalers,
    float& fAnomalyScore,
    Eigen::VectorXf& fAnomalyMap,
    bool bUseDistMetricL1)
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
    printf(" DEBUG: Inputs: coresets: vector(%zu) < vector(%zu) < MatrixXf(rows: %i, cols: %i)>> \n", fCoresets.size(), fCoresets[0].size(), fCoresets[0][0].rows(), fCoresets[0][0].cols());
    printf(" DEBUG: Inputs: fMaps: vector(%zu) < MatrixXf(rows: %i, cols: %i)> \n", iFMaps.size(), iFMaps[0].rows(), iFMaps[0].cols());
    printf(" DEBUG: Inputs: resultScalers:vector(%i) \n", fResultScalers.size());
    printf(" DEBUG: Inputs: anomalyMap: VectorXf(size: %i)> \n", fAnomalyMap.size());

    const size_t iNLayers = iFMaps.size();
    std::vector<Eigen::VectorXd> dSoftmaxeds;
    std::vector<double> dPercentileds;

    
    for (size_t layerIdx = 0; layerIdx < iNLayers; layerIdx++)
    {
        // distance calculation
        Eigen::MatrixXf fDistMat = Eigen::MatrixXf::Zero(fCoresets[layerIdx].size(), iFMaps[layerIdx].rows());
        if (bUseDistMetricL1)
        {
            fDistMat = distanceL1(fCoresets[layerIdx], iFMaps[layerIdx]);
        }
        else
        {
            fDistMat = distanceL2(fCoresets[layerIdx], iFMaps[layerIdx]);
        }

        // kNN
        Eigen::MatrixXf fTopK = kNN(fDistMat);

        // softmax
        Eigen::VectorXd dSoftmaxed = softMaxFunction(fTopK);
        dSoftmaxeds.push_back(dSoftmaxed);

        // divide by the resultScaler
        Eigen::VectorXd dNormalized = dSoftmaxed / fResultScalers[layerIdx];

        // percentile
        double dPercentiled = calcPercentile(dNormalized);
        dPercentileds.push_back(dPercentiled);

    }

    // max over both layers
    fAnomalyMap = aggreagteSofmaxeds(dSoftmaxeds);
    fAnomalyScore = aggregatePercentileds(dPercentileds);

    printf("\n DEBUG: getAnomalyScore() closed \n");
    
}


void calcNumOfPatches(twoDImgSize iImgSize, const twoDImgSize iPatchSize, twoDImgSize& iNPatches)
/*
* @brief: calculates the number of patches in both directions and returns them with nPatchesWidth and nPatchesHeight
*
* @param  IN imgSize: .height and .width of the image
* @param  IN: patchSize: .height and .width of a patch
* @param OUT nPatches: .height and .width as number of patches in each dimension
*/
{
    printf("\n DEBUG: calcNumOfPatches() opend \n");

    iNPatches.iHeight = ceil(float(iImgSize.iHeight) / float(iPatchSize.iHeight));
    iNPatches.iWidth = ceil(float(iImgSize.iWidth) / float(iPatchSize.iWidth));

    printf("\n DEBUG: calcNumOfPatches() closed \n");
 }
    

void imageToPatches(std::vector<uint8_t>& iImage, const twoDImgSize iImgSize, const twoDImgSize iNPatches, const twoDImgSize iPatchSize, std::vector<std::vector<uint8_t>>& iImagePatches)
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
    std::cout << "imgSize.width: " << iImgSize.iWidth << " imgSize.height: " << iImgSize.iHeight << std::endl;
    std::cout << "nPatches.width: " << iNPatches.iWidth << " nPatches.height: " << iNPatches.iHeight << std::endl;
    std::cout << "patchSize.width: " << iPatchSize.iWidth << " patchSize.height: " << iPatchSize.iHeight << std::endl;

    
    // calculate the difference in size
    int iNewImageWidth = iNPatches.iWidth * iPatchSize.iWidth;
    int iNewImageHeight = iNPatches.iHeight * iPatchSize.iHeight;

    int iWidthDiff = iNewImageWidth - iImgSize.iWidth;
    int iHeightDiff = iNewImageHeight - iImgSize.iHeight;
    
    std::cout << "newImageWidth: " << iNewImageWidth << " newImageHeight: " << iNewImageHeight << std::endl;
    std::cout << "widthDiff: " << iWidthDiff << " heightDiff: " << iHeightDiff << std::endl;
    

    auto paddingStartTime = std::chrono::steady_clock:: now();
    // add 0s to right edge
    int idx = 0;
    int iNumZeros = iWidthDiff;
    std::vector<float> fZzerosRight(iNumZeros, 0.0);
    for (size_t y = 0; y < iImgSize.iHeight; y++)
    {
        for (size_t ch = 0; ch < iINPUT_IMAGE_CHANNELS; ch++)
        {
            idx = ch + (iINPUT_IMAGE_CHANNELS * iImgSize.iWidth) + (y * iINPUT_IMAGE_CHANNELS * iNewImageWidth);
            // insert numZeros zeros at idx to the vector image
            iImage.insert(iImage.begin() + idx, fZzerosRight.begin(), fZzerosRight.end());
        }
    }

    // add 0s to the bottom
    iNumZeros = iINPUT_IMAGE_CHANNELS * iHeightDiff * iNewImageWidth;  // 3 = nChannels
    std::vector<float> fZerosBottom(iNumZeros, 0.0);
    iImage.insert(iImage.end(), fZerosBottom.begin(), fZerosBottom.end());
    
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
    int iNRowElems = iPATCH_WIDTH * iINPUT_IMAGE_CHANNELS;         // number of elements from the flattet image which should be copied to a patch
    int endIdx = startIdx + iNRowElems;                          // indx in the image where inserting into an patch ends

    int iNPatchRows = iNPatches.iWidth * iNPatches.iHeight * iPATCH_HEIGHT;  // how many rows from the image are copied into patches

    // loop over every patch row which is copied from the image to a specific patch
    for (size_t patchRow = 0; patchRow < iNPatchRows; patchRow++)
    {
        ptchIdx = ptchX + (ptchY * iNPatches.iWidth);
        //std::cout << "Current patch: " << ptchIdx << " (" << ptchX << "," << ptchY << ")"
        //    " and start and end idxs: " << startIdx << "-" << endIdx << std::endl;

        std::vector<uint8_t>& curPatch = iImagePatches[ptchIdx];
        curPatch.insert(curPatch.end(), iImage.begin() + startIdx, iImage.begin() + endIdx);

        // update startIdx and endIdx
        startIdx = endIdx;
        endIdx = startIdx + iNRowElems;
        ptchX++;

        // exit patch at the right boarder
        if (ptchX >= iNPatches.iHeight)
        {
            ptchX = 0;
            rowIdx++;
        } 
        // added a comment

        // exit patch at the bottom boarder
        if (rowIdx >= iPATCH_HEIGHT)
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


void patchesToImage(std::vector<uint8_t>& iImage, std::vector<std::vector<uint8_t>>& iImagePatches, twoDImgSize iNPatches)
/*
* @brief: puts all patches together in one image
*
* @param  IN imagePatches: vector of patches
* @param  IN nPatches: .height and .width as num of pixels in a patch
* @param OUT image: all patches added together as one image
*/
{
    printf("\n DEBUG: patchesToImage() opend \n");

    iImage.clear();
    std::vector<uint8_t> iCol;
    size_t idx = 0;
    for (size_t x = 0; x < iNPatches.iWidth; x++)
    {
        iCol.clear();
        for (size_t y = 0; y < iNPatches.iHeight; y++)
        {
            idx = (x * iNPatches.iHeight) + y;
            std::vector<uint8_t> patch = iImagePatches[idx];
            // ToDo col.append(patch, 'y');
        }
        // ToDo image.append(col, 'x');
    }
    printf("\n DEBUG: patchesToImage() closed \n");
}


std::vector<uint8_t> mobileNetPreprocessing(std::vector<uint8_t>& iImg)
/*
* @brief normalizes the image by dividing by 172.5 and subtracting -1.0
*
* @param IN/OUT img: CImg in in range[0, 255] out in range [-1.0, 1.0]
*/
{
    printf("\n DEBUG: mobileNetPreprocessing() opend \n");
    std::vector<uint8_t> iOutputImg;
    int iValue;
    for (uint8_t& elem : iImg)
    {
        iValue = elem; // / 127.5 - 1.0;
        iOutputImg.push_back(iValue);
    }
    
    printf("\n DEBUG: mobileNetPreprocessing() closed \n");
    return iOutputImg;
}



std::vector<uint8_t> decode_bmp(const uint8_t* iInput, int iRow_size, int iWidth,
    int iHeight, int iChannels, bool bTop_down) {
    std::vector<uint8_t> iOutput(iHeight * iWidth * iChannels);
    for (int i = 0; i < iHeight; i++) {
        int iSrc_pos;
        int iDst_pos;

        for (int j = 0; j < iWidth; j++) {
            if (!bTop_down) {
                iSrc_pos = ((iHeight - 1 - i) * iRow_size) + j * iChannels;
            }
            else {
                iSrc_pos = i * iRow_size + j * iChannels;
            }

            iDst_pos = (i * iWidth + j) * iChannels;

            switch (iChannels) {
            case 1:
                iOutput[iDst_pos] = iInput[iSrc_pos];
                break;
            case 3:
                // BGR -> RGB
                iOutput[iDst_pos] = iInput[iSrc_pos + 2];
                iOutput[iDst_pos + 1] = iInput[iSrc_pos + 1];
                iOutput[iDst_pos + 2] = iInput[iSrc_pos];
                break;
            case 4:
                // BGRA -> RGBA
                iOutput[iDst_pos] = iInput[iSrc_pos + 2];
                iOutput[iDst_pos + 1] = iInput[iSrc_pos + 1];
                iOutput[iDst_pos + 2] = iInput[iSrc_pos];
                iOutput[iDst_pos + 3] = iInput[iSrc_pos + 3];
                break;
            default:
                std::cout << "Unexpected number of channels: " << iChannels << std::endl;
                break;
            }
        }
    }
    return iOutput;
}

std::vector<uint8_t> read_bmp(const std::string& sInput_bmp_name, 
                              int* iWidth,
                              int* iHeight, int* channels) 
{
    int iBegin, iEnd;
    std::ifstream file(sInput_bmp_name, std::ios::in | std::ios::binary);
    if (!file) {
        std::cout << "input file " << sInput_bmp_name << " not found" << std::endl;
        exit(-1);
    }

    iBegin = file.tellg();
    file.seekg(0, std::ios::end);
    iEnd = file.tellg();
    size_t len = iEnd - iBegin;

    std::cout << "len: " << len << std::endl;

    std::vector<uint8_t> iImg_bytes(len);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(iImg_bytes.data()), len);
    const int32_t iHeader_size =
        *(reinterpret_cast<const int32_t*>(iImg_bytes.data() + 10));
    *iWidth = *(reinterpret_cast<const int32_t*>(iImg_bytes.data() + 18));
    *iHeight = *(reinterpret_cast<const int32_t*>(iImg_bytes.data() + 22));
    const int32_t iBpp =
        *(reinterpret_cast<const int32_t*>(iImg_bytes.data() + 28));
    *channels = iBpp / 8;

    std::cout << "width, height, channels: " << *iWidth << ", " << *iHeight
        << ", " << *channels << std::endl;

    // there may be padding bytes when the width is not a multiple of 4 bytes
    // 8 * channels == bits per pixel
    const int iRow_size = (8 * *channels * *iWidth + 31) / 32 * 4;

    // if height is negative, data layout is top down
    // otherwise, it's bottom up
    bool bTop_down = (*iHeight < 0);

    // Decode image, allocating tensor once the image size is known
    const uint8_t* iBmp_pixels = &iImg_bytes[iHeader_size];
    return decode_bmp(iBmp_pixels, iRow_size, *iWidth, abs(*iHeight), *channels,
        bTop_down);
}


int main()
{
    auto mainStartTime = std::chrono::steady_clock::now();
    // 1. load Image
    std::vector<uint8_t> iImage = read_bmp(cINPUT_IMAGE_PATH, &iINPUT_IMAGE_WIDTH, &iINPUT_IMAGE_HEIGHT, &iINPUT_IMAGE_CHANNELS);
    twoDImgSize iImgSize{iINPUT_IMAGE_HEIGHT, iINPUT_IMAGE_WIDTH }; // swapped imange dimensions
    
    
    // 2. Preprocess the image
    twoDImgSize iPatchSize{ iPATCH_HEIGHT, iPATCH_WIDTH };
    twoDImgSize iNPatches;
    
    std::vector<uint8_t> iPreProcessedImage = mobileNetPreprocessing(iImage);
    calcNumOfPatches(iImgSize, iPatchSize, iNPatches);
    std::vector<std::vector<uint8_t>> iImagePatches(iNPatches.iWidth * iNPatches.iHeight); // initialize the vector of patches with the number of patches(=nPatches.width * nPatches.height)
    imageToPatches(iPreProcessedImage, iImgSize, iNPatches, iPatchSize, iImagePatches);
       

    // 3. load coreset and fmaps
    std::vector<std::vector<Eigen::MatrixXf>> fCoresets;
    std::vector<Eigen::MatrixXi> iFMaps;
    std::vector<float> iResultScalers;
    std::vector<uint8_t> iLayerFeatures = { iN_FEATURES_L5, iN_FEATURES_L6 };

    for (const auto& nFeatures : iLayerFeatures)  // initialize coreset and fmpas
    {
        iFMaps.push_back(Eigen::MatrixXi::Zero(iN_PIXELS, nFeatures));
        std::vector<Eigen::MatrixXf> fCoreset;
        for (size_t i = 0; i < iN_CORESET_SAMPLES; i++)
        {
            fCoreset.push_back(Eigen::MatrixXf::Zero(iN_PIXELS, nFeatures));
        }
        fCoresets.push_back(fCoreset);
    }
    loadTfLiteModel(fCoresets, iFMaps, iResultScalers, iImagePatches);

    return 0;
    /*// 4. calculate Anomaly Score and display the anomalyMap
    float fAnomalyScore = -1.0;
    Eigen::VectorXf fAnomalyMap(iN_PIXELS);
    getAnomalyScore(fCoresets, iFMaps, iResultScalers, fAnomalyScore, fAnomalyMap, false);
    std::cout << "anomalyScore: " << fAnomalyScore << std::endl;
    //displayAnomlayMap(anomalyMap, nPatches);

    auto mainEndTime = std::chrono::steady_clock::now();
    std::cout << "Elapsed padding time in microseconds: "
        << std::chrono::duration_cast<std::chrono::microseconds>(mainEndTime - mainStartTime).count() << " µs" << std::endl;

    return 0;
    */
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