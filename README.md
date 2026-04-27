---
title: deep-learning-eurosat
app_file: app.py
sdk: gradio
sdk_version: 6.13.0
---
# DAT255 Project Group 7

## Title:

Engineering and evaluating multispectral convolutional neural networks for land use and land cover classification

## Description:

This project investigates land cover classification using Sentinel-2 satellite imagery and CNNs. We will train and compare models on RGB images and full 13-band multispectral inputs to evaluate performance differences and computational trade-offs. We will also analyze the importance of individual spectral bands through controlled experiments to understand their contribution to classification performance. If there is time we would like to make a simple web-app that either allows a user to upload a satellite image or integrate with an online map service.

## Data:

The EuroSAT Sentinel-2 land cover classification dataset (https://zenodo.org/records/7711810), which contains 27,000 labeled satellite images across 10 land cover classes in both RGB and 13-channel multispectral formats.

## Models:

We will implement and train CNNs from randomly initialized weights for both RGB and 13-channel inputs, and use systematic experiment tracking (with one of the recommended tools from the lecture) to compare architectures, training strategies, and spectral band configurations.
