import argparse
import dataLoader.DA.genData
import os
import prepareData


def main():
    parser = argparse.ArgumentParser(description='Random_Forest_RunExperiment')
    args = parser.parse_args()
    dataset = dataLoader.DA.genData.office31(os.path.dirname(os.path.abspath(__file__)))
    prepareData.extractFeatures(dataset, 'resnet18', 1)


if __name__ == '__main__':
    main()