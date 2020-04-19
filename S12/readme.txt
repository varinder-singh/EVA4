
This project contains many folders and there is a sequence to run the files.

First of all the Objective-
1. Download Tiny Imagenet 200 data.
2. Preprocess data by splitting in 70% to 30% train and test split respecively.
3. Train ResNet18 model on this dataset.
4. Achieve 50% accuracy within 50 epochs.

Take Away from here - how to create own data, build it and process it for own use.

A very good approach is taken here to write a custom DataSet (inherited by pytorch DataSet inbuilt class). There were issues in implemeting especially CPU RAM leakage which is a known issue in pytorch. File - TinyImagenetDataSetBuilder.py is a unique way fo implemeniting a DataSet with object oriented way of solving aforementioned issue. Alongwith this a github link - "https://github.com/pytorch/pytorch/issues/13246" (open issue till date) solution posted as - "https://github.com/pytorch/pytorch/issues/13246#issuecomment-615846051". Encourage readers to go through this.

Part 2 is assignment - 

1. collection of 50 dog images.
2. used an online tool to draw Bounding Boxes
3. With IOU and Kmeans find the cluster.
4. Cluster centers provides the anchor boxes on the 50 images.

Objective of part B is to understand what YOLO V1,V2 does behind the scenes.
